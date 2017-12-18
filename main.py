#coding=utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gym
from compiler.ast import flatten
from gym import wrappers


STATE_DIM = 4
ACTION_DIM = 2
STEP = 2000
SAMPLE_NUMS = 30

class ActorNetwork():
    def __init__(self,hidden_size,action_size):
        with tf.variable_scope("Actor"):
            self.inputs = tf.placeholder(tf.float32,[None,STATE_DIM],'s')
            self.fc1 = tf.layers.dense(inputs = self.inputs, units = hidden_size, use_bias = True, activation = tf.nn.relu)
            self.fc2 = tf.layers.dense(inputs = self.fc1, units = hidden_size, use_bias = True, activation = tf.nn.relu)
            self.fc3 = tf.layers.dense(inputs = self.fc2, units = ACTION_DIM, use_bias = True, activation = None)

            self.out = tf.nn.log_softmax(self.fc3)

            self.actions = tf.placeholder(tf.float32,[None,ACTION_DIM])
            self.prob = tf.reduce_sum(self.out * self.actions, 1)
            self.advantages = tf.placeholder(tf.float32,[None])
            self.loss = - tf.reduce_mean(self.prob * self.advantages)

            actor_network_optim = tf.train.AdamOptimizer(learning_rate=0.01)
            actor_network_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor")
            self.optimizer = actor_network_optim.minimize(self.loss, var_list=actor_network_var)


class ValueNetwork():
    def __init__(self,hidden_size,output_size):
        with tf.variable_scope("Critic"):
            self.inputs = tf.placeholder(tf.float32,[None, STATE_DIM],'s')
            self.fc1 = tf.layers.dense(inputs = self.inputs, units = hidden_size, use_bias = True, activation = tf.nn.relu)
            self.fc2 = tf.layers.dense(inputs = self.fc1, units = hidden_size, use_bias = True, activation = tf.nn.relu)
            self.fc3 = tf.layers.dense(inputs = self.fc2, units = output_size, use_bias = True, activation = None)
            self.out = tf.reduce_sum(self.fc3,1)
            self.target_values = tf.placeholder(tf.float32,[None],'target_values')
            self.loss = tf.reduce_mean(tf.square(self.target_values - self.out))

            value_network_optim = tf.train.AdamOptimizer(learning_rate=0.01)
            value_network_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic")
            self.optimizer = value_network_optim.minimize(self.loss,var_list=value_network_var)



def roll_out(sess,actor_network,task,sample_nums,value_network,init_state):
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for j in range(sample_nums):
#        task.render()
        states.append(state)
        state = state.reshape(1,-1)
        log_softmax_action = sess.run(actor_network.out,feed_dict = {actor_network.inputs:state})
        softmax_action = sess.run(tf.exp(log_softmax_action))     # 转化为概率表示
        action = np.random.choice(ACTION_DIM, p = softmax_action[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state, reward, done, _ = task.step(action)
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = task.reset()
            break

    if not is_done:
        final_state = final_state.reshape(1,-1)
        final_r = sess.run(value_network.out, feed_dict = {value_network.inputs:final_state})

    return states, actions, rewards, final_r,state

def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():
    task = gym.make("CartPole-v0")
#    task = wrappers.Monitor(task, "CartPole-v0" + "/experiment-1", force=True)
    init_state = task.reset()
    modelpath = './model/model.ckpt'
    modelpath_ = './model/'
    results = []

#init value network
    value_network = ValueNetwork(40,1)

#init actor network
    actor_network = ActorNetwork(40,ACTION_DIM)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(modelpath_)
    if ckpt and ckpt.model_checkpoint_path:
        print "old vars"
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print "new vars"

    for step in range(STEP):
        print step
        states_var, actions_var, rewards, final_r, current_state = roll_out(sess,actor_network, task,SAMPLE_NUMS,value_network,init_state)
        init_state = current_state

        vs = sess.run(value_network.out, feed_dict = {value_network.inputs:states_var})

        qs = discount_reward(rewards, 0.99, final_r)

        advantages = list(map(lambda x: x[0]-x[1], zip(qs, vs)))

        loss,_ = sess.run((actor_network.loss,actor_network.optimizer),feed_dict={actor_network.inputs:states_var,actor_network.actions:actions_var,actor_network.advantages:advantages})

        sess.run(value_network.optimizer,feed_dict={value_network.inputs:states_var,value_network.target_values:qs})

        if (step + 1) % 50 == 0:
            result = 0
            saver.save(sess,modelpath, global_step=step)
            test_task = gym.make("CartPole-v0")
            test_task = wrappers.Monitor(test_task, "CartPole-v0" + "/experiment-1", force=True)
            for test_epi in range(10):
                print test_epi
                state = test_task.reset()
                for test_step in range(200):
                    test_task.render()
                    state = state.reshape(1,-1)
                    out = sess.run(actor_network.out, feed_dict={actor_network.inputs:state})
                    softmax_action = sess.run(tf.exp(out))
                    action = np.argmax(softmax_action[0])
                    next_state, reward, done,_ = test_task.step(action)
                    result += reward
                    state = next_state
                    if done:
                        break
            print("step:",step + 1,"test result:",result/10.0)
            results.append(result/10.0)
    print results

if __name__ == '__main__':
    main()





