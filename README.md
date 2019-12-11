# How to accelerate deep spiking neural networks with recurrent API from Keras/Tensorflow
### by Lynn K.A. Sörensen, Bojian Yin, H.Steven Scholte & Sander M. Bohté

## Background
Spiking deep neural networks are attractive for deep learning for their biological realism and sparse activations, making them more efficient in how they represent activation. Recently, it has become possible to convert common deep convolutional neural networks trained with a specific activation function instead of a ReLU into a deep spiking network after training (e.g. [Zambrano et al. 2018](https://www.frontiersin.org/articles/10.3389/fnins.2018.00987/full); [Rueckauer et al. 2016](https://arxiv.org/abs/1612.04052)). Developments like this fuel the surge in new applications using spiking deep neural networks for artificial vision and event-based processing (e.g. [Mueggler, Huber & Scaramuzza, 2014](https://ieeexplore.ieee.org/abstract/document/6942940)). 

While there is a range of APIs that excel at capturing the complexity of spiking neurons such as [GeNN](http://genn-team.github.io/genn/), [Nest](https://www.nest-simulator.org/) or [Nengo](https://www.nengo.ai/), these frameworks are not yet integrated with the most commonly used deep learning APIs. Here, we show how you can implement any spiking neuron in a deep neural network leveraging recurrent layer routines from established deep learning APIs such as [Keras](https://keras.io/)/[Tensorflow](https://www.tensorflow.org/) to effectively speed-up computation time. 

![alt text](https://github.com/lynnsoerensen/SpikingRecurrencyAPI/blob/master/Figure.png "Overview")

## Approach
### Neuron models

A spiking neuron maintains a number of parameters over time such as the integrated current and its firing threshold in the case of an [adaptive firing neuron](https://papers.nips.cc/paper/4698-efficient-spike-coding-with-multiplicative-adaptation-in-a-spike-response-model.pdf). Because all of these parameters decay over time as well as respond to new incoming activations, the states of these parameters have to be maintained from one timesteps to another. Algorithmically, this can result in a for-loop in which every time step is being computed sequentially (but see causal convolutions). These kind of computations are not well-suited for GPUs since they cannot be parallelized and require a lot of memory. 

<details><summary>This is an example of one such implementation for a spiking neuron in Numpy.</summary>
<p>
    
```python
class ASN:
    """ Adaptive spiking neuron class """
    def __init__(self, mf = 0.1, bias = 0):
        # Params of a spiking neuron
        # membrane filter
        self.tau_phi =2.5
        self.dPhi = np.exp(-1 / self.tau_phi)
        # threshold decay filter
        self.tau_gamma = 15.0
        self.dGamma = np.exp(-1 / self.tau_gamma)
        # refractory decay  filter
        self.tau_eta = 50.0
        self.dEta = np.exp(-1 / self.tau_eta)
        self.dBeta = self.dEta

        self.m_f = mf 
        self.theta0 = self.m_f  # Resting threshold

        self.S_bias = bias,
        self.S = self.S_bias  # filtered activation, initialized with bias
        self.S_dyn = 0
        self.theta =self.theta0  # Start value of thresehold
        self.theta_dyn = 0  # dynamic part of the threshold
        self.S_hat = 0  # refractory response, internal approximation

        self.I = 0
        self.spike = 0

    def update(self ,current ,spike_train = True):
        """inject current for one moment in time at once"""
        # Membrane filter
        if spike_train == True:
            self.I = self.I * self.dBeta + current
        else:
            self.I = current
        self.S_dyn =(1 - self.dPhi) * self.I + self.dPhi * self.S_dyn
        self.S = self.S_bias + self.S_dyn
        # Decay
        self.S_hat = self.S_hat * self.dEta
        
        # Spike?
        if self.S - self.S_hat > 0.5 * self.theta:
            self.spike = 1  # Code spike

            # Update refractory response
            self.S_hat = self.S_hat + self.theta

            # Update threshold
            self.theta_dyn = self.theta_dyn + self.m_f * self.theta  # adaptive part based on the paper

         else:
            self.spike = 0

        # Decay
        self.theta_dyn = self.theta_dyn * self.dGamma
        self.theta = self.theta0 + self.theta_dyn


    def call(self, input, spike_train=True, mf=0.1, bias=0):
        timesteps = input.shape[1]
        batch_size = input.shape[0]
        spikes = np.zeros(input.shape)

        for b in range(batch_size):
            self.__init__(mf=mf, bias=bias)
            for t in range(timesteps):  # loop over timesteps
                self.update(input[b, t, :], spike_train=spike_train)
                spikes[b, t, 0] = self.spike

        return spikes 
```
</p>
</details>

### Recurrent API
Recurrent layer processing routines found in deep learning toolboxes such as Keras/Tensorflow and PyTorch offer a more optimized way to perform these computations on a GPU. The key here is that only the preceding state of the parameters has to be kept in memory on the GPU making it a more light-weight operation and that by expressing the spiking computations in a recurrent routine, they can be optimized for GPU computation. 

<details><summary>Below is the call function for a Spiking neuron layer in Tensorflow/Keras.</summary>
<p>

```python
    def call(self, inputs, mask=None):

        batch_size = K.shape(inputs)[0]

        # Preallocate states
        I = tf.zeros((batch_size, self.units))
        S_dyn = tf.zeros((batch_size, self.units))  # dynamic part of the activation
        theta_dyn = tf.zeros((batch_size, self.units))  # dynamic part of the threshold
        S_hat = tf.zeros((batch_size, self.units))  # refractory response, internal approximation

        # Loop over all time points
        last_output, outputs, states = K.rnn(self.update,
                                             inputs,
                                             [I, theta_dyn, S_dyn, S_hat],
                                             unroll=False,
                                             input_length=K.int_shape(inputs)[1])

        return outputs
        
```
</p>
</details>


<details><summary>The core of the spiking neuron can be found in the step function, which is passed to keras.backend.rnn().</summary>
<p>
    
```python

    def update(self, current, states):
        """inject current for one moment in time at once"""
        # states: [I, theta_dyn, S_dyn, S_hat]

        I = states[0]
        theta_dyn = states[1]
        S_dyn = states[2]
        S_hat = states[3]

        theta = self.theta0 + theta_dyn

        # Apply dense weights
        current = tf.matmul(current, self.kernel)

        # Membrane filter
        if self.input_layer == True:  # in the case when the input to the neuron is already a current, e.g. pixel values
            I = current

        else:  # when the input is a spiking sequence
            I = I * self.dBeta + current

        # Membrane filter
        S_dyn = (1 - self.dPhi) * I + self.dPhi * S_dyn

        if self.use_bias:
            S = self.bias + S_dyn
        else:
            S = S_dyn

        # Decay
        S_hat = S_hat * self.dEta

        # Spike?
        spike = tf.cast(S - S_hat > 0.5 * theta, tf.float32)  # Code spike

        # Update refractory response
        S_hat = S_hat + tf.multiply(theta, spike)

        # Update threshold
        theta_dyn = theta_dyn + tf.multiply(tf.multiply(theta, spike), self.mf)

        # Decay
        theta_dyn = theta_dyn * self.dGamma

        if self.last_layer == True:
            out = self.activation(S * self.h)  # for the last layer give out the S instead of spikes
        else:
            out = spike * self.h  # if it is a spike scale by h

        return out, [I, theta_dyn, S_dyn, S_hat] 
```
</p>
</details>

Comparing these two approaches above over 300 timesteps shows that the Tensorflow implementation deals well with an increase in neurons computed over 300 timesteps, while the Numpy implementation quickly incurs long delays. 

![alt text](https://github.com/lynnsoerensen/SpikingRecurrencyAPI/blob/master/Performance.png "Performance comparison")

Please see the [Demo](https://github.com/lynnsoerensen/SpikingRecurrencyAPI/blob/master/Demo.py) for more details. 

## Conclusion

We here showed how any spiking neuron model can be used with current deep learning APIs for recurrent neural network routines. This is a simple and straightforward way to implement spiking neurons that can scale to large-scale architectures such as Resnet18. 
