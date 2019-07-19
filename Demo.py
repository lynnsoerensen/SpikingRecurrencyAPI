"""
This demo illustrates how you can use recurrent API from tf-keras to speed up spiking neuron computation time.
For more information on the neuron model used see Bohte, 2012
(https://papers.nips.cc/paper/4698-efficient-spike-coding-with-multiplicative-adaptation-in-a-spike-response-model.pdf)

Numpy - 1.16.4
Tensorflow-gpu - 1.10
Keras - 2.2.4

L. Sörensen - 11.07.2019
"""

import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
import time

class ASN:
    """ Adaptive spiking neuron class """
    def __init__(self, mf = 0.1, bias = 0):
        # Params
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

        self.m_f = mf  # **2 would be the old matlab code
        self.theta0 = self.m_f  # Resting threshold

        self.S_bias = bias,
        self.S = self.S_bias  # filtered activation, initialized with bias
        self.S_dyn = 0
        self.theta =self.theta0  # Start value of thresehold
        self.theta_dyn = 0  # dynamic part of the threshold
        self.S_hat = 0  # refractory response, internal approximation

        #self.current_next = 0  # incoming current in next neuron
        #self.S_next = 0  # and filtered by the membrane potential.
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
        #self.current_next = self.current_next * self.dEta
        # Spike?
        if self.S - self.S_hat > 0.5 * self.theta:
            self.spike = 1  # Code spike

            # Update refractory response
            self.S_hat = self.S_hat + self.theta

            # Update threshold
            self.theta_dyn = self.theta_dyn + self.m_f * self.theta  # adaptive part based on the paper

            #self.current_next = self.current_next + 1
        else:
            self.spike = 0

        # Decay
        self.theta_dyn = self.theta_dyn * self.dGamma
        self.theta = self.theta0 + self.theta_dyn

        # Signal in next neuron
        #self.S_next = (1 - self.dPhi) * self.current_next + self.dPhi * self.S_next

    def call(self, input, spike_train=True, mf=0.1, bias=0):
        timesteps = input.shape[1]
        batch_size = input.shape[0]
        #S_next = np.zeros(input.shape)
        spikes = np.zeros(input.shape)

        for b in range(batch_size):
            self.__init__(mf=mf, bias=bias)
            for t in range(timesteps):  # loop over timesteps
                self.update(input[b, t, :], spike_train=spike_train)
                #S_next[b, t, 0] = self.S_next
                spikes[b, t, 0] = self.spike

        return spikes #, S_next


class ASN_1D(keras.layers.Dense):
    """ Adaptive spiking neuron class
    This was the first version of the ASN, which only computes simple activation values.

    Adapted from a regular densely-connected NN layer (Keras documentation below):
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).

        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

        last_layer: Key command for a read-out ASN neuron
        # Input shape
            nD tensor with shape: `(batch_size, ..., input_dim)`.
            The most common situation would be
            a 2D input with shape `(batch_size, input_dim)`.
        # Output shape
            nD tensor with shape: `(batch_size, ..., units)`.
            For instance, for a 2D input with shape `(batch_size, input_dim)`,
            the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='ones',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_layer=False,
                 last_layer=False,
                 mf=0.1,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ASN_1D, self).__init__(units, **kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_spec = keras.layers.InputSpec(min_ndim=3)
        self.supports_masking = True
        self.input_layer = input_layer
        self.last_layer = last_layer

        # Params for ASN neuron
        # membrane filter
        self.tau_phi = 2.5
        if self.last_layer == True:
            self.tau_phi = 50.  # Longer temporal integration for last layer
        self.dPhi = K.constant(np.exp(-1 / self.tau_phi))
        # threshold decay filter
        self.tau_gamma = 15.0
        self.dGamma = K.constant(np.exp(-1 / self.tau_gamma))
        # refractory decay  filter
        self.tau_eta = 50.0
        self.dEta = K.constant(np.exp(-1 / self.tau_eta))
        self.dBeta = self.dEta

        self.mf = mf
        self.h = 1
        self.theta0 = self.mf  # Resting threshold


    def build(self, input_shape):
        assert len(input_shape) >= 3
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = keras.layers.InputSpec(min_ndim=3, axes={-1: input_dim})
        self.built = True

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

    def call(self, inputs, mask=None):

        batch_size = K.shape(inputs)[0]

        # Preallocate states
        I = tf.zeros((batch_size, self.units)) # current
        S_dyn = tf.zeros((batch_size, self.units))  # dynamic part of the integrated current
        theta_dyn = tf.zeros((batch_size, self.units))  # dynamic part of the threshold
        S_hat = tf.zeros((batch_size, self.units))  # refractory response

        # Loop over all time points
        last_output, outputs, states = K.rnn(self.update,
                                             inputs,
                                             [I, theta_dyn, S_dyn, S_hat],
                                             unroll=False,
                                             input_length=K.int_shape(inputs)[1])

        return outputs

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super(keras.layers.Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


#sess = tf.InteractiveSession()
sess = K.get_session()
#init = tf.global_variables_initializer()
n_neurons = [10, 100, 1000, 2000, 4000, 8000, 10000, 20000]

times = np.zeros((2,len(n_neurons)))
# Create stimulus
time_steps = 300
for i,n in enumerate(n_neurons):
    t1 = time.time()
    S = np.repeat(np.linspace(0,4,n)[:,np.newaxis,np.newaxis],time_steps, axis=1)
    # Evaluate numpy implementation
    neuron = ASN()
    spikes_neuron = neuron.call(S, spike_train=False)
    t2 = time.time()
    print('Numpy- Total execution time for ' + str(n_neurons[i]) + ' neurons: ' + str(t2 - t1))
    times[0, i] = t2 - t1

    # Convert stimulus to a tensor
    S_tf = tf.Variable(S, dtype='float32')

    # Evaluate keras-tf implementation
    layer = ASN_1D(1,use_bias=False,mf=0.1, input_layer=True) # integrating current
    layer.build(input_shape=(None, time_steps, 1))
    spikes_layer = layer.call(S_tf)

    init = tf.global_variables_initializer()

    t1 = time.time()
    with tf.Session() as sess:
        init.run()
        spikes_opt = spikes_layer.eval()
    t2 = time.time()
    times[1, i] = t2 - t1
    print('Keras-Tensorflow- Total execution time for ' + str(n_neurons[i]) + ' neurons: ' + str(t2 - t1))

    del S, neuron, spikes_neuron, S_tf,layer,spikes_layer, spikes_opt


import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '15'
plt.rcParams['legend.frameon'] = False
fig,ax  = plt.subplots()
ax.plot(n_neurons, times.T, marker='.')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True)
ax.set_ylabel('Execution time (sec)')
ax.set_xlabel('# neurons')

fig.legend(['Numpy spiking neurons', 'Tensorflow/Keras\nspiking neurons'], loc='upper center', bbox_to_anchor=(0.4,1))

fig.show()


#%%


