import tensorflow as tf
import scipy
import numpy as np
from model import A3C_Network
from helper import *
from vizdoom import *
import skimage 
from skimage import transform, color, exposure
from skimage.transform import rotate

def process_frame(frame):
    s = skimage.color.rgb2gray(frame)
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

class Worker:

    def __init__(self, env, name, obv_space, action_space,
                 trainer, model_path, global_eps, render=True):
        self.name = "worker_" + str(name)
        self.number = name
        self.render = True
        self.model = model_path
        self.trainer = trainer
        self.global_eps = global_eps
        self.increment = self.global_eps.assign_add(1)
        self.eps_rew = []
        self.eps_len = []
        self.eps_mean = []
        self.env = env
        self.summary_writer = tf.summary.FileWriter("train" + str(name))

        self.local_net = A3C_Network(obv_space, action_space, self.name, trainer)
        # Update the local variables to match global network
        self.update_local_ops = update_target_graph('global', self.name)

    def train(self, rollout, sess, gamma, bootstrap_val):
        rollout = np.array(rollout)
        # Ideally rollout batch will be [:,0-3] with each row being an observation
        observations = np.array(rollout[:,0])
        actions = np.array(rollout[:,1])
        rewards = np.array(rollout[:,2])
        next_obs = np.array(rollout[:,3])
        values = np.array(rollout[:,5])

        # Generate advantage and discounted rewards
        self.rewards = np.asarray(rewards.tolist() + [bootstrap_val])
        discount_r = discount(self.rewards,gamma)[:-1]
        self.values = np.asarray(values.tolist() + [bootstrap_val])
        advantages = rewards + gamma * self.values[1:] - self.values[:-1]
        advantages = discount(advantages, gamma)

        # update global network from loss
        # Generate network stats
        feed_dict = {
            self.local_net.target_v:discount_r,
            self.local_net.inputs:np.vstack(observations),
            self.local_net.actions:actions,
            self.local_net.advantages:advantages,
            self.local_net.state_in[0]:self.batch_rnn_state[0],
            self.local_net.state_in[1]:self.batch_rnn_state[1]
        }

        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run(
            [self.local_net.value_loss,
             self.local_net.policy_loss,
             self.local_net.entropy,
             self.local_net.grad_norms,
             self.local_net.var_norms,
             self.local_net.state_out,
             self.local_net.apply_grads],
            feed_dict=feed_dict)

        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n

    def work(self, max_eps, gamma, sess, coord, saver):
        eps_count = sess.run(self.global_eps)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.reset()
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_net.state_init
                self.batch_rnn_state = rnn_state
                while not d:
                    # Get action using policy
                    # print("state size {}".format(s.shape))
                    # quit()
                    a_dist, v, rnn_state = sess.run([self.local_net.policy,
                                                     self.local_net.value,
                                                     self.local_net.state_out],
                                                    feed_dict = {
                                                        self.local_net.inputs:[s],
                                                        self.local_net.state_in[0]:rnn_state[0],
                                                        self.local_net.state_in[1]:rnn_state[1]
                                                    })
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    # run through env with actions
                    s_, r, d, info = self.env.step(a)
                    s_ = process_frame(s_)
                    if self.render:
                        self.env.render()

                    if d:
                        s_ = s
                    # Append to rollout
                    episode_buffer.append([s,a,r,s_,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s_
                    total_steps += 1
                    episode_step_count += 1

                    # If episode is still running and buffer is full than update
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_eps - 1:
                        # Value estimation
                        v1 = sess.run(self.local_net.value,
                                      feed_dict={
                                          self.local_net.inputs:[s],
                                          self.local_net.state_in[0]:self.batch_rnn_state[0],
                                          self.local_net.state_in[1]:self.batch_rnn_state[1]
                                      })[0,0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d:
                        break
                self.eps_rew.append(episode_reward)
                self.eps_len.append(episode_step_count)
                self.eps_mean.append(np.mean(episode_values))

                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)

                # Training diagnostics
                if eps_count % 5 == 0 and eps_count != 0:
                    if eps_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model+'/model-'+str(eps_count)+'.cptk')
                        print("Saved Model")
                    mean_reward = np.mean(self.eps_rew[-5:])
                    print("MEAN REWARDS {}".format(mean_reward))
                    mean_len = np.mean(self.eps_len[-5:])
                    mean_val = np.mean(self.eps_mean[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_len))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_val))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, eps_count)
                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                eps_count += 1


