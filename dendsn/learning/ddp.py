from typing import Callable, Union

import torch
import torch.nn as nn
from spikingjelly.activation_based import monitor

from dendsn.learning.base import BaseLearner
from dendsn.model import synapse
from dendsn.model import synapse_conn
from dendsn.model import neuron
from dendsn.model import dendrite


def ddp_linear_single_step(
    fc: nn.Linear, dsn: neuron.BaseDendNeuron,
    pre_spike: torch.Tensor, v_dend: torch.Tensor, v_soma: torch.Tensor,
    f_rate: Callable, f_u_pred: Callable, f_w: Callable
):
    weight = fc.weight.data # [N_out, N_in]
    B = pre_spike.shape[0]
    n_comp = dsn.dend.wiring.n_compartment
    # n_input_comp = dsn.dend.wiring.n_input
    v_dend = v_dend.view(B, -1)
    v_soma = v_soma.repeat(1, 1, n_comp).view(B, -1)
    # [B, N_soma * n_comp] = [B, N_soma * n_input_comp] = [B, N_out]

    pred_error = f_rate(v_soma) - f_rate(f_u_pred(v_dend))
    delta_w = (pred_error.unsqueeze(-1) * pre_spike.unsqueeze(1)).sum(0)
    return f_w(weight) * delta_w


def ddp_linear_multi_step(
    fc: nn.Linear, dsn: neuron.BaseDendNeuron,
    pre_spike: torch.Tensor, v_dend: torch.Tensor, v_soma: torch.Tensor,
    f_rate: Callable, f_u_pred: Callable, f_w: Callable
):
    weight = fc.weight.data # [N_out, N_in]
    T, B = pre_spike.shape[0], pre_spike.shape[1]
    n_comp = dsn.dend.wiring.n_compartment
    # n_input_comp = dsn.dend.wiring.n_input
    v_dend = v_dend.view(T, B, -1)
    v_soma = v_soma.repeat(1, 1, 1, n_comp).view(T, B, -1)
    # [T, B, N_soma * n_comp] = [T, B, N_soma * n_input_comp] = [T, B, N_out]

    pred_error = f_rate(v_soma) - f_rate(f_u_pred(v_dend))
    delta_w = (pred_error.unsqueeze(-1) * pre_spike.unsqueeze(2)).sum((0, 1))
    return f_w(weight) * delta_w


def ddp_multi_step(f_single: Callable) -> Callable:
    def f_multi(
        layer: nn.Linear, dsn: neuron.BaseDendNeuron,
        pre_spike: torch.Tensor, v_dend: torch.Tensor, v_soma: torch.Tensor,
        f_rate: Callable, f_u_pred: Callable, f_w: Callable
    ):
        weight = layer.weight.data
        delta_w = torch.zeros_like(weight)
        T = pre_spike.shape[0]

        for t in range(T):
            dw = f_single(
                layer, dsn, pre_spike[t], v_dend[t], v_soma[t], 
                f_rate, f_u_pred, f_w
            )
            delta_w += dw

        return delta_w

    return f_multi


class DDPLearner(BaseLearner):

    def __init__(
        self, syn: Union[synapse.BaseSynapse, synapse_conn.BaseSynapseConn],
        dsn: neuron.BaseDendNeuron, f_rate: Callable, 
        f_w: Callable = lambda w: 1, f_u_pred: Callable = lambda u: u,
        step_mode: str = "s", specified_multi_imp: bool = False
    ):
        super().__init__(step_mode=step_mode)
        self.syn = syn
        self.dsn = dsn
        if not isinstance(self.dsn.dend, dendrite.SegregatedDend):
            raise NotImplementedError(
                f"DDPLearner only supports dsn with SegregatedDend at present, "
                f"but type(dsn.dend)={type(self.dsn.dend)}."
            )

        self.pre_spike_monitor = monitor.InputMonitor(syn)
        self.v_dend_monitor = monitor.AttributeMonitor(
            "v_seq" if step_mode=="m" else "v", 
            pre_forward=False, net=dsn.dend.compartment
        )
        self.v_soma_monitor = monitor.AttributeMonitor(
            "v_seq" if step_mode=="m" else "v", 
            pre_forward=False, net=dsn.soma
        )
        self.f_rate = f_rate
        self.f_w = f_w
        self.f_u_pred = f_u_pred

        conn = syn
        if isinstance(self.syn, synapse.BaseSynapse):
            conn = syn.conn
        if isinstance(conn, nn.Linear):
            self.f_single = ddp_linear_single_step
            self.f_multi = (
                ddp_linear_multi_step if specified_multi_imp
                else ddp_multi_step(ddp_linear_single_step)
            )
        else:
            raise NotImplementedError(
                f"STDPLearner doesn't support synapse_conn: {conn}"
            )

    def reset(self):
        super().reset()
        self.pre_spike_monitor.clear_recorded_data()
        self.v_dend_monitor.clear_recorded_data()
        self.v_soma_monitor.clear_recorded_data()

    def enable(self):
        self.pre_spike_monitor.enable()
        self.v_dend_monitor.enable()
        self.v_soma_monitor.enable()

    def disable(self):
        self.pre_spike_monitor.disable()
        self.v_dend_monitor.disable()
        self.v_soma_monitor.disable()

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
            v_dend = self.v_dend_monitor.records.pop(0)
            v_soma = self.v_soma_monitor.records.pop(0)
            dw = f(
                syn, self.dsn, pre_spike, v_dend, v_soma, 
                self.f_rate, self.f_u_pred, self.f_w
            )
            dw *= scale
            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if syn.weight.grad is None:
                syn.weight.grad = -delta_w
            else:
                syn.weight.grad = syn.weight.grad - delta_w
        return delta_w