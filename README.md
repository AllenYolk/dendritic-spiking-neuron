# dendritic-spiking-neuron

**Author**: [AllenYolk](mailto:huang2627099045@gmail.com)

A new dendritic computing package `dendsn` based on [pytorch](https://github.com/pytorch/pytorch) and [spikingjelly](https://github.com/fangwei123456/spikingjelly).
Draw inspiration from the nonlinear nature of biological dendritic neurons!

The model in `dendsn` is based on:
* (Legenstein & Maass, 2011)
*

### Modules in `dendsn`

Dendritic neurons are built in a bottom-up manner in this package.

* **dendritic neuron**: `model/neuron.py` 
    * dendrite: `model/dendrite.py`
        * dendritic compartments: `model/dend_compartment.py`
        * wiring of the compartments: `model/wiring.py`
    * connection from dendrite to soma: `model/dend_soma_conn.py`
    * soma: `spikingjelly.activation_based.neuron.py`
* **synapse**: `model/synapse.py`
    * synaptic connection and weights: `model/synapse_conn.py`
    * synaptic filter: `model/synapse_filter.py`
* other modules: 
    * encapsulated jit operations: `functional.py`: 
    * stochastic spiking autograd functions: `stochastic_firing.py`



### References:
* Legenstein, R., & Maass, W. (2011). Branch-specific plasticity enables self-organization of nonlinear computation in single neurons. The Journal of Neuroscience: The Official Journal of the Society for Neuroscience, 31(30), 10787–10802. https://doi.org/10.1523/JNEUROSCI.5684-10.2011

* 