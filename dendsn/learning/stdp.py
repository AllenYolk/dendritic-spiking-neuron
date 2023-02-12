from typing import Callable, Union

import torch
import torch.nn as nn
from spikingjelly.activation_based import monitor

from dendsn.learning.base import BaseLearner
from dendsn.model import synapse
from dendsn.model import synapse_conn
from dendsn.model import neuron


def stdp_linear_single_step(
    fc: nn.Linear, dsn: neuron.BaseDendNeuron,
    pre_spike: torch.Tensor, post_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None], 
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float, tau_post: float, 
    f_w_pre_post: Callable, f_w_post_pre: Callable,
    f_trace_pre: Callable, f_trace_post: Callable
):
    weight = fc.weight.data # [N_out, N_in]

    if trace_pre is None:
        trace_pre = 0.
    if trace_post is None:
        trace_post = 0.
    trace_pre = trace_pre - trace_pre / tau_pre + pre_spike # [B, N_in]
    trace_post = trace_post - trace_post / tau_post + post_spike # [B, N_soma]

    n_input_comp = dsn.dend.wiring.n_input # N_out = N_soma * n_input_comp
    B = pre_spike.shape[0]
    if n_input_comp == 1:
        trace_post_syn = trace_post
        post_spike_syn = post_spike
    else:
        trace_post_syn = (
            trace_post.unsqueeze(-1).repeat(1, 1, n_input_comp).view(B, -1)
        ) # [B, N_out]
        post_spike_syn = (
            post_spike.unsqueeze(-1).repeat(1, 1, n_input_comp).view(B, -1)
        ) # [B, N_out]

    # [B, N_out, N_in] -> [N_out, N_in]
    delta_w_post_pre = (
        -f_w_post_pre(weight) * 
        ((f_trace_post(trace_post_syn).unsqueeze(-1) * pre_spike.unsqueeze(1))
        .sum(0))
    )
    delta_w_pre_post = (
        f_w_pre_post(weight) * 
        ((f_trace_pre(trace_pre).unsqueeze(1) * post_spike_syn.unsqueeze(-1))
        .sum(0))
    )
    return trace_pre, trace_post, delta_w_post_pre + delta_w_pre_post


def stdp_linear_multi_step(
    fc: nn.Linear, dsn: neuron.BaseDendNeuron,
    pre_spike: torch.Tensor, post_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None], 
    trace_post: Union[float, torch.Tensor, None],
    tau_pre: float, tau_post: float, 
    f_w_pre_post: Callable, f_w_post_pre: Callable,
    f_trace_pre: Callable, f_trace_post: Callable
):
    weight = fc.weight.data
    T = pre_spike.shape[0]
    B = pre_spike.shape[1]

    if trace_pre is None:
        trace_pre = 0.
    if trace_post is None:
        trace_post = 0.
    trace_pre_seq = torch.zeros_like(pre_spike) # [T, B, N_in]
    trace_post_seq = torch.zeros_like(post_spike) # [T, B, N_soma]
    for t in range(T):
        trace_pre = trace_pre - trace_pre / tau_pre + pre_spike[t]
        trace_post = trace_post - trace_post / tau_post + post_spike[t] 
        trace_pre_seq[t] = trace_pre
        trace_post_seq[t] = trace_post

    n_input_comp = dsn.dend.wiring.n_input # N_out = N_soma * n_input_comp
    if n_input_comp == 1:
        trace_post_seq_syn = trace_post_seq
        post_spike_syn = post_spike
    else:
        trace_post_seq_syn = (
            trace_post.unsqueeze(-1).repeat(1, 1, 1, n_input_comp)
            .view(T, B, -1)
        ) # [T, B, N_out]
        post_spike_syn = (
            post_spike.unsqueeze(-1).repeat(1, 1, 1, n_input_comp)
            .view(T, B, -1)
        ) # [T, B, N_out]

    # [T, B, N_out, N_in] -> [N_out, N_in]
    delta_w_post_pre = (
        -f_w_post_pre(weight) * 
        ((f_trace_post(trace_post_seq_syn).unsqueeze(-1) 
        * pre_spike.unsqueeze(2))
        .sum((0, 1)))
    )
    delta_w_pre_post = (
        f_w_pre_post(weight) * 
        ((f_trace_pre(trace_pre_seq).unsqueeze(2)
        * post_spike_syn.unsqueeze(-1))
        .sum((0, 1)))
    )
    return trace_pre, trace_post, delta_w_post_pre + delta_w_pre_post


def stdp_multi_step(f_single: Callable) -> Callable:
    def f_multi(
        layer: nn.Module, dsn: neuron.BaseDendNeuron,
        pre_spike: torch.Tensor, post_spike: torch.Tensor,
        trace_pre: Union[float, torch.Tensor, None], 
        trace_post: Union[float, torch.Tensor, None],
        tau_pre: float, tau_post: float, 
        f_w_pre_post: Callable, f_w_post_pre: Callable,
        f_trace_pre: Callable, f_trace_post: Callable
    ):
        weight = layer.weight.data
        delta_w = torch.zeros_like(weight)
        T = pre_spike.shape[0]

        for t in range(T):
            trace_pre, trace_post, dw = f_single(
                layer, dsn, pre_spike[t], post_spike[t], trace_pre, trace_post,
                tau_pre, tau_post, f_w_pre_post, f_w_post_pre,
                f_trace_pre, f_trace_post
            )
            delta_w += dw

        return trace_pre, trace_post, delta_w

    return f_multi


class STDPLearner(BaseLearner):

    def __init__(
        self, syn: Union[synapse.BaseSynapse, synapse_conn.BaseSynapseConn], 
        dsn: neuron.BaseDendNeuron, tau_pre: float, tau_post: float,
        f_w_pre_post: Callable = lambda w: 1, 
        f_w_post_pre: Callable = lambda w: 1,
        f_trace_pre: Callable = lambda t: t, 
        f_trace_post: Callable = lambda t: t,
        step_mode: str = "s", specified_multi_imp: bool = False
    ):
        super().__init__(step_mode=step_mode)
        self.syn = syn
        self.dsn = dsn
        self.pre_spike_monitor = monitor.InputMonitor(syn)
        self.post_spike_monitor = monitor.OutputMonitor(dsn)
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.f_w_pre_post = f_w_pre_post
        self.f_w_post_pre = f_w_post_pre
        self.f_trace_pre = f_trace_pre
        self.f_trace_post = f_trace_post

        conn = syn
        if isinstance(self.syn, synapse.BaseSynapse):
            conn = syn.conn
        if isinstance(conn, nn.Linear):
            self.f_single = stdp_linear_single_step
            self.f_multi = (
                stdp_linear_multi_step if specified_multi_imp
                else stdp_multi_step(stdp_linear_single_step)
            ) # stdp_multi_step(f_single) is quicker than stdp_linear_multi_step
        else:
            raise NotImplementedError(
                f"STDPLearner doesn't support synapse_conn: {conn}"
            )

        self.register_memory("trace_pre", None)
        self.register_memory("trace_post", None)

    def reset(self):
        super().reset()
        self.pre_spike_monitor.clear_recorded_data()
        self.post_spike_monitor.clear_recorded_data()

    def enable(self):
        self.pre_spike_monitor.enable()
        self.post_spike_monitor.enable()

    def disable(self):
        self.pre_spike_monitor.disable()
        self.post_spike_monitor.disable()

    def step(self, on_grad: bool = True, scale: float = 1.):
        delta_w = None
        l = len(self.pre_spike_monitor.records)

        f = None
        if self.step_mode == "s":
            f = self.f_single
        elif self.step_mode == "m":
            f = self.f_multi
        else:
            raise ValueError(f"step_mode={self.step_mode}")

        syn = self.syn
        if isinstance(self.syn, synapse.BaseSynapse):
            syn = self.syn.conn

        for _ in range(l):
            pre_spike = self.pre_spike_monitor.records.pop(0)
            post_spike = self.post_spike_monitor.records.pop(0)
            self.trace_pre, self.trace_post, dw = f(
                syn, self.dsn,
                pre_spike, post_spike, self.trace_pre, self.trace_post, 
                self.tau_pre, self.tau_post, 
                self.f_w_pre_post, self.f_w_post_pre,
                self.f_trace_pre, self.f_trace_post
            )
            dw *= scale
            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if syn.weight.grad is None:
                syn.weight.grad = -delta_w
            else:
                syn.weight.grad = syn.weight.grad - delta_w
        return delta_w
