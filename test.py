import argparse

import torch
from dendsn import dend_compartment, neuron, wiring


def dend_compartment_test(T: int = 10, N: int  = 3):
    x_seq = torch.randn(size = [T, N]) + 0.5

    dend = dend_compartment.PassiveDendCompartment(step_mode = "m", decay_input = True)
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
    w = wiring.SingleDendLayerWiring(
        n_compartment = 10
    )
    w.validation_check()
    print(w.adjacency_matrix)


def dendrite_test():
    pass


def neuron_test():
    dn = neuron.DendNeuron()


def main():
    parser = argparse.ArgumentParser(description = "dendsj test")
    parser.add_argument(
        "--mode", "-m", type = str, default = "neuron",
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
    else:
        raise ValueError(f"Invalid argument: mode = {args.mode}")


if __name__ == "__main__":
    main()