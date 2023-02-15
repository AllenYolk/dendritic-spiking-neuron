from typing import Callable

import torch
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import surrogate


class BaseSoma(neuron.BaseNode):

    def __init__(
        self, v_threshold: float = 1., v_reset: float = 0., 
        surrogate_function: Callable = surrogate.Sigmoid(), 
        detach_reset: bool = False, step_mode: str = "s", 
        backend: str = "torch", store_v_seq: bool = False
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode,
            backend, store_v_seq
        )

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        v_pre_spike = self.v
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike, v_pre_spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        v_pre_spike_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y, v_pre_spike = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            v_pre_spike_seq.append(v_pre_spike)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        return torch.stack(y_seq), torch.stack(v_pre_spike_seq)


class LIFSoma(BaseSoma):

    def __init__(
        self, tau: float = 2., decay_input: bool = True, 
        v_threshold: float = 1., v_reset: float = 0., 
        surrogate_function: Callable = surrogate.Sigmoid(), 
        detach_reset: bool = False, step_mode: str = "s", 
        backend: str = "torch", store_v_seq: bool = False
    ):
        if not (isinstance(tau, float) and tau >= 1.):
            return AssertionError(
                f"LIFSoma.tau should be larger than 1., "
                f"but get tau={tau}."
            )
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, 
            step_mode, backend, store_v_seq
        )
        self.tau = tau
        self.decay_input = decay_input

    def extra_repr(self):
        return (
            super().extra_repr() + 
            f", tau={self.tau}, decay_input={self.decay_input}"
        )

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_decay_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_decay(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_no_decay_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v  - v / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_single_eval_hard_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        v_pre_spike = v + (x - (v - v_reset)) / tau
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = spike * v_reset + (1. - spike) * v_pre_spike
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_hard_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        v_pre_spike = v - (v - v_reset) / tau + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = spike * v_reset + (1. - spike) * v_pre_spike
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_soft_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        v_pre_spike = v + (x - v) / tau
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_pre_spike - spike * v_threshold
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_soft_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        v_pre_spike = v - v / tau + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_pre_spike - spike * v_threshold
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_no_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_no_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_no_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_no_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if (self.v_reset is None) or (self.v_reset == 0.):
                self.v = self.jit_neuronal_charge_decay_reset0(
                    x, self.v, self.tau
                )
            else:
                self.v = self.jit_neuronal_charge_decay(
                    x, self.v, self.v_reset, self.tau
                )
        else:
            if (self.v_reset is None) or (self.v_reset == 0.):
                self.v = self.jit_neuronal_charge_no_decay_reset0(
                    x, self.v, self.tau
                )
            else:
                self.v = self.jit_neuronal_charge_no_decay(
                    x, self.v, self.v_reset, self.tau
                )

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            return super().single_step_forward(x)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None: # soft reset
                if self.decay_input:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_soft_decay(
                            x, self.v, self.v_threshold, self.tau
                        )
                    )
                else:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_soft_no_decay(
                            x, self.v, self.v_threshold, self.tau
                        )
                    )
            else:
                if self.decay_input:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_hard_decay(
                            x, self.v, self.v_threshold, self.v_reset, self.tau
                        )
                    )
                else:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_hard_no_decay(
                            x, self.v, self.v_threshold, self.v_reset, self.tau
                        )
                    )
            return spike, v_pre_spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            return super().multi_step_forward(x_seq)
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None: # soft reset
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_soft_decay_v_seq(
                                x_seq, self.v, self.v_threshold, self.tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_soft_decay(
                                x_seq, self.v, self.v_threshold, self.tau
                            )
                        )
                else:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_soft_no_decay_v_seq(
                                x_seq, self.v, self.v_threshold, self.tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_soft_no_decay(
                                x_seq, self.v, self.v_threshold, self.tau
                            )
                        )
            else: # hard reset
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_hard_decay_v_seq(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, self.tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_hard_decay(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, self.tau
                            )
                        )
                else:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_hard_no_decay_v_seq(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, self.tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_hard_no_decay(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, self.tau
                            )
                        )
            return spike_seq, v_pre_spike_seq
