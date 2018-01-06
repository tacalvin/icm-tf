import gym
import numpy as np
import multiprocessing
import threading
import time
import os
import tensorflow as tf
from time import sleep
from worker import *
from model import *
from env import *
MAX_EPS = 400
gamma = .99

ENV = 'mario'
# env_test = create_atari_env(ENV,0,0)
env_test = create_env(ENV,0,0)
obv_space = 7056 # env_test.observation_space.shape
action_space = env_test.action_space.n
load_model = False
model_path = './model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.mkdir(model_path)

with tf.device('/cpu:0'):
    global_eps = tf.Variable(0, dtype=tf.int32, name='global_episode',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    global_network = A3C_Network(obv_space, action_space,'global', None)
    num_workers = 4 #multiprocessing.cpu_count() -1
    workers = []
    print("Creating workerse")
    for i in range(num_workers):
        workers.append(Worker(create_env(ENV,i+1,i+1),i+1,obv_space,
                              action_space,
                              trainer,
                              model_path,
                              global_eps))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    print("Beginning training")
    if load_model:
        print("Loading Model")
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(ckpt)
    else:
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(MAX_EPS, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
