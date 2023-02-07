from typing import Callable, Union

import torch
import torch.nn as nn
from spikingjelly.activation_based import monitor

from dendsn.learning.base import BaseLearner
from dendsn.model import synapse
from dendsn.model import synapse_conn
from dendsn.model import neuron


def semistdp_linear_single_step(
    fc: nn.Linear, dsn: neuron.BaseDendNeuron,
    pre_spike: torch.Tensor, post_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None], tau_pre: float,
    f_w_pre_post: Callable, f_trace_pre: Callable,
):
    weight = fc.weight.data # [N_out, N_in]

    if trace_pre is None:
        trace_pre = 0.
    trace_pre = trace_pre - trace_pre / tau_pre + pre_spike # [B, N_in]

    n_input_comp = dsn.dend.wiring.n_input # N_out = N_soma * n_input_comp
    B = pre_spike.shape[0]
    if n_input_comp == 1:
        post_spike_syn = post_spike
    else:
        post_spike_syn = (
            post_spike.repeat(1, 1, n_input_comp).view(B, -1)
        ) # [B, N_out]

    # [B, N_out, N_in] -> [N_out, N_in]
    delta_w_pre_post = (
        f_w_pre_post(weight) * 
        ((f_trace_pre(trace_pre).unsqueeze(1) * post_spike_syn.unsqueeze(-1))
        .sum(0))
    )
    return trace_pre, delta_w_pre_post


def semistdp_linear_multi_step(
    fc: nn.Linear, dsn: neuron.BaseDendNeuron,
    pre_spike: torch.Tensor, post_spike: torch.Tensor,
    trace_pre: Union[float, torch.Tensor, None], tau_pre: float, 
    f_w_pre_post: Callable, f_trace_pre: Callable,
):
    weight = fc.weight.data
    T = pre_spike.shape[0]
    B = pre_spike.shape[1]

    if trace_pre is None:
        trace_pre = 0.
    trace_pre_seq = torch.zeros_like(pre_spike) # [T, B, N_in]
    for t in range(T):
        trace_pre = trace_pre - trace_pre / tau_pre + pre_spike[t]
        trace_pre_seq[t] = trace_pre

    n_input_comp = dsn.dend.wiring.n_input # N_out = N_soma * n_input_comp
    if n_input_comp == 1:
        post_spike_syn = post_spike
    else:
        post_spike_syn = (
            post_spike.repeat(1, 1, 1, n_input_comp).view(T, B, -1)
        ) # [T, B, N_out]

    # [T, B, N_out, N_in] -> [N_out, N_in]
    delta_w_pre_post = (
        f_w_pre_post(weight) * 
        ((f_trace_pre(trace_pre_seq).unsqueeze(2)
        * post_spike_syn.unsqueeze(-1))
        .sum((0, 1)))
    )
    return trace_pre, delta_w_pre_post


def semistdp_multi_step(f_single: Callable) -> Callable:
    def f_multi(
        layer: nn.Module, dsn: neuron.BaseDendNeuron,
        pre_spike: torch.Tensor, post_spike: torch.Tensor,
        trace_pre: Union[float, torch.Tensor, None], tau_pre: float,
        f_w_pre_post: Callable, f_trace_pre: Callable,
    ):
        weight = layer.weight.data
        delta_w = torch.zeros_like(weight)
        T = pre_spike.shape[0]

        for t in range(T):
            trace_pre, dw = f_single(
                layer, dsn, pre_spike[t], post_spike[t], trace_pre, tau_pre, 
                f_w_pre_post, f_trace_pre,
            )
            delta_w += dw

        return trace_pre, delta_w

    return f_multi


class SemiSTDPLearner(BaseLearner):

    def __init__(
        self, syn: Union[synapse.BaseSynapse, synapse_conn.BaseSynapseConn], 
        dsn: neuron.BaseDendNeuron, tau_pre: float,
        f_w_pre_post: Callable = lambda w: 1, 
        f_trace_pre: Callable = lambda t: t, 
        step_mode: str = "s", specified_multi_imp: bool = False
    ):
        super().__init__(step_mode=step_mode)
        self.syn = syn
        self.dsn = dsn
        self.pre_spike_monitor = monitor.InputMonitor(syn)
        self.post_spike_monitor = monitor.OutputMonitor(dsn)
        self.tau_pre = tau_pre
        self.f_w_pre_post = f_w_pre_post
        self.f_trace_pre = f_trace_pre

        conn = syn
        if isinstance(self.syn, synapse.BaseSynapse):
            conn = syn.conn
        if isinstance(conn, nn.Linear):
            self.f_single = semistdp_linear_single_step
            self.f_multi = (
                semistdp_linear_multi_step if specified_multi_imp
                else semistdp_multi_step(semistdp_linear_single_step)
            )
        else:
            raise NotImplementedError(
                f"STDPLearner doesn't support synapse_conn: {conn}"
            )

        self.register_memory("trace_pre", None)

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
            self.trace_pre, dw = f(
                syn, self.dsn,
                pre_spike, post_spike, self.trace_pre, self.tau_pre,
                self.f_w_pre_post, self.f_trace_pre,
            )
            dw *= scale
            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if syn.weight.grad is None:
                syn.weight.grad = -delta_w
            else:
                syn.weight.grad = syn.weight.grad - delta_w
        return delta_w
