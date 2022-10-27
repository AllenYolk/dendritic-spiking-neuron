import argparse

import torch
from spikingjelly.activation_based import neuron as sj_neuron

from dendsn import dend_compartment, wiring, dendrite, neuron, dend_soma_conn


def dend_compartment_test(T: int = 5, N: int  = 3):
    print("====="*20)
    print("dendritic compartment dynamics test:")
    x_seq = torch.randn(size = [T, N]) + 0.5

    dend = dend_compartment.PassiveDendCompartment(
        decay_input = True, step_mode = "m"
    )
    v_seq = dend(x_seq)
    for t in range(T):
        print(f"t = {t}:")
        print(f"    x = {x_seq[t]}")
        print(f"    v = {v_seq[t]} [decay input]")

    dend.reset()
    dend.decay_input = False
    v_seq = dend(x_seq)
    for t in range(T):
        print(f"t = {t}:")
        print(f"    x = {x_seq[t]}")
        print(f"    v = {v_seq[t]} [not decay input]")  


def wiring_test():
    print("====="*20)
    print("dendritic compartment wiring test:")
    w = wiring.SegregatedDendWiring(
        n_compartment = 10
    )
    w.validation_check()
    print(w.adjacency_matrix)


def dendrite_test(T: int = 5, B: int = 2, N: int = 6, k: int = 3):
    print("====="*20)
    print("dendritic model test:")
    x_seq = torch.randn(size = [T, B, N])

    dend1 = dendrite.SegregatedDend(
        compartment = dend_compartment.PassiveDendCompartment(),
        wiring = wiring.SegregatedDendWiring(n_compartment = N),
        step_mode = "m"
    )
    dend2 = dendrite.VDiffDend(
        compartment = dend_compartment.PassiveDendCompartment(),
        wiring = wiring.SegregatedDendWiring(n_compartment = N),
        step_mode = "m"
    )
    w3 = wiring.SegregatedDendWiring(n_compartment = N)
    w3.adjacency_matrix = torch.eye(n = N, dtype = torch.int32)
    dend3 = dendrite.VDiffDend(
        compartment = dend_compartment.PassiveDendCompartment(),
        wiring = w3, step_mode = "m"
    )
    dend4 = dendrite.VDiffDend(
        compartment = dend_compartment.PassiveDendCompartment(),
        wiring = wiring.Kto1DendWirng(k = k, n_output = N // k, n_input = N),
        step_mode = "m"
    )
    v1_seq = dend1(x_seq)
    v2_seq = dend2(x_seq)
    v3_seq = dend3(x_seq)
    v4_seq = dend4(x_seq)
    for t in range(T):
        print(f"t = {t}:")
        print(f"    x = {x_seq[t, 0]}")
        print(f"    v1 = {v1_seq[t, 0]} [decay input, SegregatedDend]")
        print(f"    v2 = {v2_seq[t, 0]} [decay input, VDiffDend]")
        print(f"    v3 = {v3_seq[t, 0]} [decay input, VDiffDend, eye]")
        print(f"    v4 = {v4_seq[t, 0]} [decay input, VDiffDend, {k}to1]")


def neuron_test(
    T: int = 25, B: int = 2, N: int = 36, k1: int = 4, k2: int = 3,
    tau_dend = 3., tau_soma = 20.
):
    print("====="*20)
    print("neuron model test:")
    x_seq = torch.randn(size = [T, B, N]) + 3

    dn = neuron.VForwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//k1, n_input = N
            ),
            coupling_strength = (tau_dend - 1.) / k1# dynamics analysis
        ),
        soma = sj_neuron.LIFNode(tau = tau_soma),
        dend_soma_conn = dend_soma_conn.Kto1DendSomaConn(
            k = k2, n_soma = N//k1//k2, n_output_compartment = N//k1,
            enable_backward = False
        ),
        step_mode = "m"
    )
    dn_fb = neuron.VForwardSBackwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//k1, n_input = N, bidirection = False
            ),
            coupling_strength = (tau_dend - 1.) / k1
        ),
        soma = sj_neuron.LIFNode(tau = tau_soma),
        dend_soma_conn = dend_soma_conn.Kto1DendSomaConn(
            k = k2, n_soma = N//k1//k2, n_output_compartment = N//k1,
            enable_backward = True
        ),
        step_mode = "m"
    )
    bdn = neuron.VForwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//k1, n_input = N, bidirection = True
            ),
            coupling_strength = (tau_dend - 1.) / k1# dynamics analysis
        ),
        soma = sj_neuron.LIFNode(tau = tau_soma),
        dend_soma_conn = dend_soma_conn.Kto1DendSomaConn(
            k = k2, n_soma = N//k1//k2, n_output_compartment = N//k1,
            enable_backward = False
        ),
        step_mode = "m"
    )
    bdn_fb = neuron.VForwardSBackwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//k1, n_input = N, bidirection = True
            ),
            coupling_strength = (tau_dend - 1.) / k1
        ),
        soma = sj_neuron.LIFNode(tau = tau_soma),
        dend_soma_conn = dend_soma_conn.Kto1DendSomaConn(
            k = k2, n_soma = N//k1//k2, n_output_compartment = N//k1,
            enable_backward = True
        ),
        step_mode = "m"
    )

    y_seq = dn(x_seq)
    y_seq_fb = dn_fb(x_seq)
    by_seq = bdn(x_seq)
    by_seq_fb = bdn_fb(x_seq)
    dn.step_mode = "s"
    dn_fb.step_mode = "s"
    bdn.step_mode = "s"
    bdn_fb.step_mode = "s"
    dn.reset()
    dn_fb.reset()
    bdn.reset()
    bdn_fb.reset()

    for t in range(T):
        y = dn(x_seq[t])
        y_fb = dn_fb(x_seq[t])
        by = bdn(x_seq[t])
        by_fb = bdn_fb(x_seq[t])
        print(f"t = {t}:")
        print(
            f"    v_dend = "
            f"{dn.dend.compartment.v[0, dn.dend.wiring.output_index]}"
            f" [VF, decay input]"
        )
        print(
            f"    v_dend = "
            f"{dn_fb.dend.compartment.v[0, dn.dend.wiring.output_index]}"
            f" [VFSB, decay input]"
        )
        print(
            f"    v_dend = "
            f"{bdn.dend.compartment.v[0, dn.dend.wiring.output_index]}"
            f" [BiVF, decay input]"
        )
        print(
            f"    v_dend = "
            f"{bdn_fb.dend.compartment.v[0, dn.dend.wiring.output_index]}"
            f" [BiVFSB, decay input]"
        )
        print(f"    v_soma = {dn.soma.v[0]} [VF, decay input]")
        print(f"    v_soma = {dn_fb.soma.v[0]} [VFSB, decay input]")
        print(f"    v_soma = {bdn.soma.v[0]} [BiVF, decay input]")
        print(f"    v_soma = {bdn_fb.soma.v[0]} [BiVFSB, decay input]")
        print(f"    spike = {y[0]} [VF, decay input, single-step]")
        print(f"    spike = {y_seq[t, 0]} [VF, decay input, multi-step]")
        print(f"    spike = {y_fb[0]} [VFSB, decay input, single-step]")
        print(f"    spike = {y_seq_fb[t, 0]} [VFSB, decay input, multi-step]")
        print(f"    spike = {by[0]} [BiVF, decay input, single-step]")
        print(f"    spike = {by_seq[t, 0]} [BiVF, decay input, multi-step]")
        print(f"    spike = {by_fb[0]} [BiVFSB, decay input, single-step]")
        print(f"    spike = {by_seq_fb[t, 0]} [BiVFSB, decay input, multi-step]")


def main():
    parser = argparse.ArgumentParser(description = "dendsj test")
    parser.add_argument(
        "--mode", "-m", type = str, default = "all",
        help = "the mode of test.py (specifying the feature to be tested)"
    )
    args = parser.parse_args()

    if args.mode == "dend_compartment":
        dend_compartment_test()
    elif args.mode == "wiring":
        wiring_test()
    elif args.mode == "dendrite":
        dendrite_test()
    elif args.mode == "neuron":
        neuron_test()
    elif args.mode == "all":
        dend_compartment_test()
        wiring_test()
        dendrite_test()
        neuron_test()
    else:
        raise ValueError(f"Invalid argument: mode = {args.mode}")


if __name__ == "__main__":
    main()