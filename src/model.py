import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer




class A3C_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):

            # Create ICM
            self.ICM = ICM(s_size, a_size, scope, trainer)

            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            self.conv1 = slim.conv2d(self.imageIn, 32, [3, 3], stride=[2, 2],
                activation_fn=tf.nn.elu, padding='SAME')
            self.conv2 = slim.conv2d(self.conv1, 32, [3, 3], stride=[2, 2],
                activation_fn=tf.nn.elu, padding='SAME')
            self.conv3 = slim.conv2d(self.conv2, 32, [3, 3], stride=[2, 2],
                activation_fn=tf.nn.elu, padding='SAME')
            self.conv4 = slim.conv2d(self.conv3, 32, [3, 3], stride=[2, 2],
                activation_fn=tf.nn.elu, padding='SAME')


            #self.conv3 = slim.conv2d(self.conv2, 64, [3, 3], stride=[1, 1],
            #    activation_fn=tf.nn.elu, padding='SAME')
            hidden = slim.fully_connected(slim.flatten(self.conv4), 256, activation_fn=tf.nn.elu)

            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 50.0)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

def generic_input(x, nconvs = 4):
    for i in range(nconvs):
        x = slim.conv2d(x, 32, [3,3], stride=[2,2],
                        activation_fn=tf.nn.elu, padding="SAME")
    x = slim.flatten(x)
    return x
def linear(x, size, name, init=None, bias_init=0):
    w = tf.get_variable(name + '/W', [x.get_shape()[1], size], tf.float32,
                        initializer=init)
    b = tf.get_variable(name + '/b', [size], tf.float32,
                        tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

class ICM:
    def __init__(self, ob_space, ac_space, scope, trainer):
        with tf.variable_scope(scope):
            self.s1 = s1= tf.placeholder(tf.float32, [None, ob_space])
            self.s2 = s2 = tf.placeholder(tf.float32, [None, ob_space])
            self.act_sample = tf.placeholder(tf.float32, shape=[None, ac_space])

            units = 256
            s1 = tf.reshape(self.s1, shape=[-1, 84, 84, 1])
            s2 = tf.reshape(self.s2, shape=[-1, 84, 84, 1])
            s1 = generic_input(s1)
            s2 = generic_input(s2)

            # Inverse Model
            x = tf.concat([s1,s2],1)
            x = slim.fully_connected(x,256, activation_fn=tf.nn.elu)

            # one hot encoded action vector
            a_index = tf.argmax(self.act_sample)
            logits = linear(x, ac_space, 'last', normalized_columns_initializer())
            print(logits)
            print(a_index)
            

            self.inv_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                               labels = a_index))
            print("LOSS SUCCESS")
            self.a_inv_probs = tf.nn.softmax(logits)

            # Forward model without backprop
            f = tf.concat([s1, self.act_sample],1)
            f = tf.nn.relu(linear(f, units, 'f1', normalized_columns_initializer()))
            f = linear(f, s1.get_shape()[1].value, 'flast', normalized_columns_initializer())
            self.forward_loss = 0.5 * tf.reduce_mean(
                tf.square(tf.subtract(f, s2)),name='forward_loss')
            # From paper
            self.forward_loss *= 288


            # Worker network ops
            if scope != 'global':
                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients_inv = tf.gradients(self.inv_loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients_inv, 40.0)

                #apply to global_network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))




