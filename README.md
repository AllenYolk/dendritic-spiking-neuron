# dendritic-spiking-neuron

**Author**: [AllenYolk](mailto:2627099045@qq.com)

A new dendritic computing package `dendsn` based on [pytorch](https://github.com/pytorch/pytorch) and [spikingjelly](https://github.com/fangwei123456/spikingjelly).
Draw inspiration from the nonlinear nature of biological dendritic neurons!

The model in `dendsn` is based on:
* (Legenstein & Maass, 2011)
*

### Modules in `dendsn`

Dendritic neurons are built in a bottom-up manner in this package.

* **dendritic neuron**: `neuron.py`
    * synapse: ???
    * dendrite: `dendrite.py`
        * dendritic compartments: `dend_compartment.py`
        * wiring of the compartments: `wiring.py`
    * connection from dendrite to soma: `dend_soma_conn.py`
    * soma: `spikingjelly.activation_based.neuron.py`



### References:
* Legenstein, R., & Maass, W. (2011). Branch-specific plasticity enables self-organization of nonlinear computation in single neurons. The Journal of Neuroscience: The Official Journal of the Society for Neuroscience, 31(30), 10787–10802. https://doi.org/10.1523/JNEUROSCI.5684-10.2011

* 