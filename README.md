# dendritic-spiking-neuron

**Author**: [AllenYolk](mailto:allen.yfhuang@gmail.com)

A new dendritic computing package `dendsn` based on [PyTorch](https://github.com/pytorch/pytorch) and [SpikingJelly](https://github.com/fangwei123456/spikingjelly).
Draw inspiration from the nonlinear nature of biological dendritic neurons!

**Table of Contents:**

* [Installation](#installation)
* [Modules in `dendsn`](#modules-in-dendsn)
* [TODOs](#todos)

## Installation

### Install from source code

From [Github](https://github.com/AllenYolk/dendritic-spiking-neuron)

```shell
git clone https://github.com/AllenYolk/dendritic-spiking-neuron.git
cd dendritic-spiking-neuron
pip install .
```

## Modules in `dendsn`

### Dendritic Neuron Models

Dendritic neurons are built in a bottom-up manner in this package, and each component is implemented in a separate python script:

* **dendritic neuron**: `model/neuron.py`
    * dendrite: `model/dendrite.py`
        * dendritic compartments: `model/dend_compartment.py`
        * wiring of the compartments: `model/wiring.py`
    * soma: `model/soma.py`
* **synapse**: `model/synapse.py`
    * synaptic connection and weights: `model/synapse_conn.py`
        * Both linear and conv layers are available!
    * synaptic filter: `model/synapse_filter.py`

The basic assumption is: **all the dendritic neurons in the same layer share exactly the same morphology**!

### Learning Rules

A series of plasticity rules are available in `dendsn.learning`, whose implementation is based on "monitors" in `spikingjelly`.

* **STDP**: `learning/stdp.py`
* **Semi-STDP**: `learning/semi_stdp.py`
    * A simplified form of STDP: `trace_post` and `delta_w_post_pre` will be neglected.
* **Dendritic Prediction Plasticity**: `learning/dpp.py`
    * See (Urbanczik & Senn, 2014).

Now, these learning rules can only support fully connected weight layers.

### Other Modules: 

* useful functions: `functional.py`
* stochastic spiking autograd functions: `stochastic_firing.py`

## TODOs

* [ ] Add docstrings and comments to `dendsn.learning`.
* [ ] Extend plasticity rules in `dendsn.learning` to convolutional layers.

## References

* Urbanczik, R., & Senn, W. (2014). Learning by the Dendritic Prediction of Somatic Spiking. Neuron, 81(3), 521–528. [https://doi.org/10.1016/j.neuron.2013.11.030](https://doi.org/10.1016/j.neuron.2013.11.030)
