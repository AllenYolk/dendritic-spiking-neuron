# dendritic-spiking-neuron

**Author**: [AllenYolk](mailto:huang2627099045@gmail.com)

A new dendritic computing package `dendsn` based on [pytorch](https://github.com/pytorch/pytorch) and [spikingjelly](https://github.com/fangwei123456/spikingjelly).
Draw inspiration from the nonlinear nature of biological dendritic neurons!

**Table of Contents:**
* [Installation](#installation)
* [Modules in `dendsn`](#modules-in-dendsn)

## Installation

### Install from source code
From [Github](https://github.com/AllenYolk/dendritic-spiking-neuron)
```shell
git clone https://github.com/AllenYolk/dendritic-spiking-neuron.git
cd dendritic-spiking-neuron
pip install .
```

## Modules in `dendsn`

Dendritic neurons are built in a bottom-up manner in this package, and each component is implemented in a separate python script:
* **dendritic neuron**: `model/neuron.py` 
    * dendrite: `model/dendrite.py`
        * dendritic compartments: `model/dend_compartment.py`
        * wiring of the compartments: `model/wiring.py`
    * soma: `model/soma.py`
* **synapse**: `model/synapse.py`
    * synaptic connection and weights: `model/synapse_conn.py`
    * synaptic filter: `model/synapse_filter.py`
* other modules: 
    * encapsulated jit operations: `functional.py`: 
    * stochastic spiking autograd functions: `stochastic_firing.py`

The basic assumption is: **all the dendritic neurons in the same layer share exactly the same morphology**!