import tensorflow as tf
import numpy as np
import scipy.signal

def update_target_graph(from_scope, target):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target)

    op_holder = []
    # Creates a new op to assign target with value from_var 
    for from_var,target_var, in zip(from_vars, target_vars):
        op_holder.append(target_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
