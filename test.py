import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based import layer
import matplotlib.pyplot as plt
from tqdm import tqdm

from dendsn.model import dend_compartment, wiring, dendrite 
from dendsn.model import synapse
from dendsn.model import neuron
from dendsn import stochastic_firing


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
    T: int = 25, B: int = 2, N: int = 24, k1: int = 4, n_soma: int = 3,
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
                k = k1, n_output = N//n_soma//k1, n_input = N//n_soma
            ),
            coupling_strength = (tau_dend - 1.) / k1# dynamics analysis
        ),
        soma = sj_neuron.LIFNode(tau = tau_soma,),
        soma_shape = [n_soma],
        step_mode = "m"
    )
    dn_fb = neuron.VForwardSBackwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//n_soma//k1, n_input = N//n_soma,
                bidirection = False
            ),
            coupling_strength = (tau_dend - 1.) / k1
        ),
        soma = sj_neuron.LIFNode(tau = tau_soma),
        soma_shape = [n_soma],
        step_mode = "m"
    )
    bdn = neuron.VForwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//n_soma//k1, n_input = N//n_soma, 
                bidirection = True
            ),
            coupling_strength = (tau_dend - 1.) / k1# dynamics analysis
        ),
        soma = sj_neuron.LIFNode(tau = tau_soma),
        soma_shape = [n_soma],
        step_mode = "m"
    )
    bdn_fb = neuron.VForwardSBackwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//n_soma//k1, n_input = N//n_soma,
                bidirection = True
            ),
            coupling_strength = (tau_dend - 1.) / k1
        ),
        soma = sj_neuron.LIFNode(tau = tau_soma),
        soma_shape = [n_soma],
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
            f"{dn.dend.compartment.v[0, :, dn.dend.wiring.output_index]}"
            f" [VF, decay input]"
        )
        print(
            f"    v_dend = "
            f"{dn_fb.dend.compartment.v[0, :, dn.dend.wiring.output_index]}"
            f" [VFSB, decay input]"
        )
        print(
            f"    v_dend = "
            f"{bdn.dend.compartment.v[0, :, dn.dend.wiring.output_index]}"
            f" [BiVF, decay input]"
        )
        print(
            f"    v_dend = "
            f"{bdn_fb.dend.compartment.v[0, :, dn.dend.wiring.output_index]}"
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


def synapse_test(
    T: int = 5, B: int = 2, in_features = 20, out_features = 12,
    in_channels = 4, out_channels = 1, kernel_size = 2
):
    print("====="*20)
    print("synapse model test - MaskedLinear:")
    x_seq = torch.randn(size = [T, B, in_features])
    syn = synapse.MaskedLinearIdenditySynapse(
        in_features, out_features, bias = True, step_mode = "m"
    )
    y_seq = syn(x_seq)
    syn.step_mode = "s"
    syn.reset()
    print(f"mask = {syn.conn.weight_mask.data}")
    for t in range(T):
        y = syn(x_seq[t])
        print(f"t = {t}:")
        print(f"    x = {x_seq[t, 0]}")
        print(f"    y1 = {y_seq[t, 0]} [multi-step]")
        print(f"    y2 = {y[0]} [single-step]")

    print("====="*20)
    print("synapse model test - Conv:")
    x_seq = torch.randn(size = [T, B, in_channels, 4, 4])
    syn = synapse.Conv2dIdentitySynapse(
        in_channels, out_channels, kernel_size, step_mode = "m"
    )
    y_seq = syn(x_seq)
    syn.step_mode = "s"
    syn.reset()
    for t in range(T):
        y = syn(x_seq[t])
        print(f"t = {t}:")
        print(f"    x = {x_seq[t, 0]}")
        print(f"    y1 = {y_seq[t, 0]} [multi-step]")
        print(f"    y2 = {y[0]} [single-step]")


def stochastic_test(
    T: int = 25, B: int = 1, N: int = 96, k1: int = 4, n_soma: int = 3,
    tau_dend = 3., tau_soma = 20.
):
    print("====="*20)
    print("stochastic firing test:")
    x_seq = torch.randn(size = [T, B, N]) + 3

    stf = stochastic_firing.ExpStochasticFiring(
        f_thres = 0.9, beta = 5., spiking = True
    )
    stf2 = stochastic_firing.LogisticStochasticFiring(
        f_thres = 0.9, beta = 5., spiking = True
    )

    dn = neuron.VForwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//n_soma//k1, n_input = N//n_soma
            ),
            coupling_strength = (tau_dend - 1.) / k1# dynamics analysis
        ),
        soma = sj_neuron.LIFNode(
            tau = tau_soma, surrogate_function = stf,
        ),
        soma_shape = [n_soma],
        step_mode = "s"
    )
    dn2 = neuron.VForwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//n_soma//k1, n_input = N//n_soma
            ),
            coupling_strength = (tau_dend - 1.) / k1# dynamics analysis
        ),
        soma = sj_neuron.LIFNode(
            tau = tau_soma, surrogate_function = stf2,
        ),
        soma_shape = [n_soma],
        step_mode = "s"
    )

    for t in range(T):
        y = dn(x_seq[t])
        y2 = dn2(x_seq[t])
        print(f"t = {t}")
        print(
            f"    v_dend = "
            f"{dn.dend.compartment.v[0, :, dn.dend.wiring.output_index]}"
            f"[exp, single step]"
        )
        print(
            f"    v_dend = "
            f"{dn2.dend.compartment.v[0, :, dn.dend.wiring.output_index]}"
            f"[logistic, single step]"
        )
        print(f"    v_soma = {dn.soma.v[0]} [exp, single step]")
        print(f"    v_soma = {dn2.soma.v[0]} [logistic, single step]")
        print(
            f"    firing rate = {stf.rate_function(dn.soma.v[0] - 1.)}"
            f"[exp, single step]"
        )
        print(
            f"    firing rate = {stf2.rate_function(dn2.soma.v[0] - 1.)}"
            f"[logistic, single step]"
        )
        print(f"    spike = {y[0]} [exp, single step]")
        print(f"    spike = {y2[0]} [logistic, single step]")


def rate_plot_test():
    print("====="*20)
    print("rate plot test:")
    _, ax = plt.subplots()
    stf_exp = stochastic_firing.ExpStochasticFiring(
        f_thres = 1., beta = 5., theta = 1.
    )
    stf_logistic = stochastic_firing.LogisticStochasticFiring(
        f_thres = 1., beta = 5., theta = 1.
    )
    stf_exp.plot_firing_rate(ax = ax)
    stf_logistic.plot_firing_rate(ax = ax)
    ax.vlines(x = [1.], ymin = 0, ymax = 5., linestyles = "dotted", colors = "r")
    plt.show()


def gradient_test(
    T: int = 25, B: int = 32, M: int = 32, N: int = 24, 
    k1: int = 4, n_soma: int = 3, tau_dend = 3., tau_soma = 20.,
    iterations: int = 10000, lr = 1.,
    surrogate_function = surrogate.Sigmoid()
):
    print("====="*20)
    print("single-layer gradient BP test:")
    syn = nn.Linear(in_features = M, out_features = N, bias = False)
    #init.uniform_(syn.weight.data)
    dn = neuron.VForwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = N//n_soma//k1, n_input = N//n_soma
            ),
            coupling_strength = (tau_dend - 1.) / k1# dynamics analysis
        ),
        soma = sj_neuron.LIFNode(
            tau = tau_soma,
            surrogate_function = surrogate_function
        ),
        soma_shape = [n_soma],
        step_mode = "m"
    )
    net = nn.Sequential(syn, dn)

    net2 = nn.Sequential(
        nn.Linear(in_features = M, out_features = N, bias = False),
        sj_neuron.LIFNode(
            tau = tau_soma, 
            surrogate_function = surrogate_function,
            step_mode = "m"
        ),
        nn.Linear(in_features = N, out_features = n_soma, bias = False),
        sj_neuron.LIFNode(
            tau = tau_soma, 
            surrogate_function = surrogate_function,
            step_mode = "m"
        ),
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(params = net.parameters(), lr = lr)
    optimizer2 = optim.SGD(params = net2.parameters(), lr = lr)

    l1 = []
    l2 = []
    for iteration in tqdm(range(iterations)):
        x_seq = torch.rand(size = [T, B, M])
        functional.reset_net(net)
        functional.reset_net(net2)
        spike_seq = net(x_seq)
        spike_seq2 = net2(x_seq)
        target = torch.ones_like(spike_seq)
        target2 = torch.ones_like(spike_seq2)
        loss = criterion(spike_seq[1:], target[1:])
        loss2 = criterion(spike_seq2[1:], target2[1:])
        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        loss2.backward()
        optimizer.step()
        optimizer2.step()

        #print(
        #    f"iteration {iteration}: "
        #    f"loss1 = {loss.item()}, loss2 = {loss2.item()}"
        #)
        l1.append(loss.item())
        l2.append(loss2.item())

    print(spike_seq.mean(dim = [1, 2]))
    print(spike_seq2.mean(dim = [1, 2]))

    plt.style.use("ggplot")
    f, (ax0, ax1) = plt.subplots(ncols = 2)
    f.set_size_inches(w = 12, h = 4)
    ax0.plot(range(iterations), l1, label = "dendritic neuron")
    ax0.plot(range(iterations), l2, label = "point neuron")
    ax0.set(xlabel = "iterations", ylabel = "loss")
    ax0.legend()
    im = ax1.matshow(net[0].weight.data)
    ax1.figure.colorbar(im)
    plt.show()


def conv_test(
    T: int = 25, B: int = 32, C1: int = 3, C2: int = 24,
    h: int = 4, w: int = 5,
    k1: int = 4, soma_channel: int = 3, tau_dend = 3., tau_soma = 20.,
    iterations: int = 10000, lr = 1.,
    surrogate_function = surrogate.Sigmoid()
):
    print("====="*20)
    print("convolution test:")
    syn = layer.Conv2d(C1, C2, kernel_size = 2, stride = 1, step_mode = "m")
    #init.uniform_(syn.weight.data)
    dn = neuron.VForwardDendNeuron(
        dend = dendrite.VDiffDend(
            compartment = dend_compartment.PassiveDendCompartment(
                tau = tau_dend, decay_input = True
            ),
            wiring = wiring.Kto1DendWirng(
                k = k1, n_output = C2//k1//soma_channel,
                n_input = C2//soma_channel
            ),
            coupling_strength = (tau_dend - 1.) / k1# dynamics analysis
        ),
        soma = sj_neuron.LIFNode(
            tau = tau_soma,
            surrogate_function = surrogate_function
        ),
        soma_shape = [soma_channel, h-1, w-1],
        step_mode = "m"
    )
    net = nn.Sequential(syn, dn)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(params = net.parameters(), lr = lr)

    l1 = []
    for iteration in tqdm(range(iterations)):
        x_seq = torch.rand(size = [T, B, C1, h, w])
        functional.reset_net(net)
        spike_seq = net(x_seq)
        target = torch.ones_like(spike_seq)
        loss = criterion(spike_seq[1:], target[1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(
        #    f"iteration {iteration}: "
        #    f"loss1 = {loss.item()}, loss2 = {loss2.item()}"
        #)
        l1.append(loss.item())

    print(spike_seq.shape)
    print(spike_seq.mean(dim = [1, 2, 3, 4]))

    plt.style.use("ggplot")
    f, ax0 = plt.subplots()
    f.set_size_inches(w = 6, h = 4)
    ax0.plot(range(iterations), l1, label = "dendritic neuron")
    ax0.set(xlabel = "iterations", ylabel = "loss")
    ax0.legend()
    plt.show()

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
    elif args.mode == "synapse":
        synapse_test()
    elif args.mode == "stochastic":
        stochastic_test()
    elif args.mode == "rate_plot":
        rate_plot_test()
    elif args.mode == "gradient":
        gradient_test()
    elif args.mode == "conv":
        conv_test()
    elif args.mode == "all":
        dend_compartment_test()
        wiring_test()
        dendrite_test()
        neuron_test()
        synapse_test()
        stochastic_test()
        rate_plot_test()
        gradient_test()
        conv_test()
    else:
        raise ValueError(f"Invalid argument: mode = {args.mode}")


if __name__ == "__main__":
    main()