import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
# def normalized_col_init(std=1.0):
#             def _init(shape, dtype=None, partition_info=None):
#                 out = np.random.randn(*shape).astype(np.float32)
#                 out *= std/np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#                 return tf.constant(out)
#             return _init

# class A3C_Network:

#     def __init__(self, obv_space, action_space, scope, trainer):
#         with tf.variable_scope(scope):
#             # Here the input is handled
#             # input will take in a batch of state observations
#             self.inputs = tf.placeholder(shape=[None] + list(obv_space), dtype=tf.float32)
#             self.image_in = tf.reshape(self.inputs,shape=[-1, 84, 84, 1])

#             # image encoding
#             self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
#                                      inputs=self.image_in,
#                                      num_outputs=16,
#                                      kernel_size=[8,8],
#                                      stride=[4,4],
#                                      padding='VALID')

#             self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
#                                      inputs=self.conv1,
#                                      num_outputs=32,
#                                      kernel_size=[4,4],
#                                      stride=[2,2],
#                                      padding='VALID')
#             hidden = slim.fully_connected(slim.flatten(self.conv2),
#                                           256, activation_fn=tf.nn.elu)

#             # Recurrent layer
#             lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256,state_is_tuple=True)
#             c_init = np.zeros((1,lstm_cell.state_size.c), np.float32)
#             h_init = np.zeros((1,lstm_cell.state_size.h), np.float32)
#             self.state_init = [c_init, h_init]

#             # LSTM inputs
#             c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
#             h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
#             self.state_in = (c_in, h_in)
#             rnn_in = tf.expand_dims(hidden, [0])
#             # TODO document shape image shape
#             step_size = tf.shape(self.image_in)[:1]
#             state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
#             lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in,
#                                                         initial_state=state_in,
#                                                         sequence_length=step_size,
#                                                         time_major=False)
#             lstm_c, lstm_h = lstm_state
#             # TODO document these state ouputs
#             self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
#             rnn_out = tf.reshape(lstm_output, [-1, 256])

#             # Output layers
#             self.policy = slim.fully_connected(rnn_out, action_space,
#                                                activation_fn = tf.nn.elu,
#                                                weights_initializer=normalized_col_init(0.01),
#                                                biases_initializer=None)
#             self.value = slim.fully_connected(rnn_out,1,
#                                               activation_fn=tf.nn.elu,
#                                               weights_initializer=normalized_col_init(),
#                                               biases_initializer=None)

#             # Worker ops
#             if scope != 'global':
#                 self.actions = tf.placeholder(tf.int32, shape=[None])
#                 self.actions_onehot = tf.one_hot(self.actions, action_space,
#                                                  dtype=tf.float32)
#                 self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
#                 self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
#                 self.responsible_outs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

#                 # Loss
#                 self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
#                 self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
#                 self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outs)*self.advantages)
#                 self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

#                 # Get Gradients and update variables
#                 local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
#                 self.grads = tf.gradients(self.loss, local_vars)
#                 self.var_norm = tf.global_norm(local_vars)
#                 # TODO Document why we clip 40.0
#                 grads, self.grad_norms = tf.clip_by_global_norm(self.grads, 40.0)

#                 # Apply grads
#                 global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
#                 self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer




class A3C_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
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

