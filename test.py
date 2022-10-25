import argparse

import torch
from dendsn import dend_compartment, wiring, dendrite


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


def dendrite_test(T: int = 5, B: int = 2, N: int = 6):
    print("====="*20)
    print("dendritic model test:")
    x_seq = torch.randn(size = [T, B, N])

    dend1 = dendrite.SegregatedDend(
        compartment = dend_compartment.PassiveDendCompartment(),
        wiring = wiring.SegregatedDendWiring(n_compartment = N),
        step_mode = "m"
    )
    dend2 = dendrite.VoltageDiffDend(
        compartment = dend_compartment.PassiveDendCompartment(),
        wiring = wiring.SegregatedDendWiring(n_compartment = N),
        step_mode = "m"
    )
    w3 = wiring.SegregatedDendWiring(n_compartment = N)
    w3.adjacency_matrix = torch.eye(n = N, dtype = torch.int32)
    dend3 = dendrite.VoltageDiffDend(
        compartment = dend_compartment.PassiveDendCompartment(),
        wiring = w3, step_mode = "m"
    )
    dend4 = dendrite.VoltageDiffDend(
        compartment = dend_compartment.PassiveDendCompartment(),
        wiring = wiring.Kto1DendWirng(k = 3, n_output = 2, n_input = 6),
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
        print(f"    v2 = {v2_seq[t, 0]} [decay input, VoltageDiffDend]")
        print(f"    v3 = {v3_seq[t, 0]} [decay input, VoltageDiffDend, eye]")
        print(f"    v4 = {v4_seq[t, 0]} [decay input, VoltageDiffDend, 2to1]")


def neuron_test():
    pass


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