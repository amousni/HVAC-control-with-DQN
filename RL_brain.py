#library&const
"""
This part of code is the Deep Q Network (DQN) brain.

Using:
Tensorflow: r1.2
"""
# #问题点：
#     -参数：
#           - Q‘网络参数更新iteration
#           - epsilon每次增加的量
#     -与Prototype不同之处：s,r,a,s_在每次获得s_时，获得（s,r,a,s_） 作为参数传入store_transition(self, s, a, r, s_)

import numpy as np
import pandas as pd
import tensorflow as tf
import time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
from Data_interaction import DataInteraction 
from tensorflow.python import pywrap_tensorflow
import warnings
warnings.filterwarnings("ignore")



# define random number generator
np.random.seed(1)
tf.set_random_seed(1)

# * @dev Deep Q-learning network training and use it to output control
class DeepQNetwork:
	def __init__(
		self,
		n_actions ,
		n_features,
		learn_step_count,
		learning_rate=0.003,
		reward_decay=0.99,
		# e_greedy=0.1,
		epsilon_max=0.99,
		replace_target_iter=50, 
		memory_size=500,
		batch_size=48,
		e_greedy_increment=0.01, 
		output_graph=False,
	):

		self.action_space = np.array([30,24,25,25,26])
		# action_dict = {22:297,23:298,24:299,25:300,30:301,26:349}
		# counts of action
		self.n_actions = n_actions

		# counts of environmental variable
		self.n_features = n_features

		# learning rate of Q-learning
		self.lr = learning_rate

		# discount rate  
		self.gamma = reward_decay

		# max epsilon in e-greedy
		self.epsilon_max = epsilon_max

		# iterations for updating target Q network with evaluation Q network
		self.replace_target_iter = replace_target_iter

		# data size of memory
		self.memory_size = memory_size

		# data size of mini-batch
		self.batch_size = batch_size

		# increment value of epsilon in a step
		self.epsilon_increment = e_greedy_increment

		# initial value of epsilon
		self.epsilon = 0.1

		# count of step
		if learn_step_count >= 50:
			self.learn_step_counter = int((learn_step_count - 50)/5 + 1)
		else:
			self.learn_step_counter = 0

		# initial memory to zeros
		self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

		# build target_net & evaluate_net (BP，e：square error of every output neuron)
		self._build_net()
		t_params = tf.get_collection('target_net_params') # 从一个集合中取出全部变量，是一个列表
		e_params = tf.get_collection('eval_net_params')
		self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # 把e赋值到t

		#create saving session
		self.saver = tf.train.Saver(max_to_keep=10)

		# create a seesion
		self.sess = tf.Session() 

		# send event including graph to log
		if output_graph:
			tf.summary.FileWriter("../../data/logs/", self.sess.graph) 
		
		# restore parameter trained
		print("first learn_step_count is {}".format(learn_step_count))
		# if (self.learn_step_counter > 0):
		# 	self.learn_step_counter = 0
		# 	self.learn_step_count=0
		# 	time_str = time.strftime('%Y-%m-%d',time.localtime(time.time()))
		# 	year_str = time_str.split('-')[0][-2:]
		# 	month_str = time_str.split('-')[1]
		# 	day_str = time_str.split('-')[2]
		# 	filename_model = ""
		# 	# print('\n\ndebug here\n\n\n')
		# 	for month in range(int(month_str), 0, -1):
		# 		for day in range(31, 0, -1):
		# 			month = '0' + str(month) if len(str(month)) == 1 else str(month)
		# 			day = '0' + str(day) if len(str(day)) == 1 else str(day)
		# 			filename_model = 'dqn_model_' + year_str + month + day +'.ckpt.meta'
		# 			filename = 'dqn_model_' + year_str + month + day +'.ckpt.index'
		# 			print(filename_model, filename)
		# 			print(os.path.exists("../../data/dqn_model/"+filename))
		# 			if os.path.exists("../../data/dqn_model/"+filename):
		# 				print("dqn model file found\n")
						
		# 				# saver = tf.train.import_meta_graph("../../data/dqn_model/"+filename_model)
		# 				self.saver.restore(self.sess,"../../data/dqn_model/"+filename_model[:-5])
		# 				print("##################### DQN model {} restored..\n".format(filename_model))
		# 				break
		# 		if os.path.exists("../../data/dqn_model/"+filename):
		# 			break 
		# else:
		# # initialize variable
		self.sess.run(tf.global_variables_initializer()) 
		print('RLRunForOneTime')
		self.cost_his = [] 
		

	def _get_time_string(self):
		return time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
	
	# ** 
	# * @dev build targret NN & evaluation NN
	def _build_net(self):

		# all inputs
		self.s = tf.placeholder(tf.float32,[None, self.n_features],name='s')
		self.s_ = tf.placeholder(tf.float32,[None, self.n_features], name = 's_')
		self.r = tf.placeholder(tf.float32, [None, ], name = 'r')
		self.a = tf.placeholder(tf.int32, [None, ], name = 'a')

		# w_initializer, b_initializer = tf.random_normal_initializer(0.,0.3), tf.constant_initializer(0.1)
		ckp_path = "../../data/dqn_model/saved_variable"
		reader = pywrap_tensorflow.NewCheckpointReader(ckp_path)
		var_to_shape_map = reader.get_variable_to_shape_map()
		W1 = tf.Variable(reader.get_tensor('E/kernel/RMSProp'))
		b1 = tf.Variable(reader.get_tensor('E/bias/RMSProp'))

		# build_evaluate_net
		with tf.variable_scope('eval_net'):
			e1 = tf.add(tf.matmultiply(self.s, W1),b1,name='e1')
			# e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer,name='e1')
			self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer, bias_initializer=b_initializer,name='q')

		# build_target_net
		with tf.variable_scope('target_net'):
			t1 = tf.add(tf.matmultiply(self.s, W1),b1,name='t1')
			# t1 = tf.layers.dense(self.s_,20,tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer,name='t1')
			self.q_next = tf.layers.dense(t1,self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

		with tf.variable_scope('q_target'):
			# q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
			# self.q_target = tf.stop_gradient(q_target) 
			def maxmize_a(tensor_a):
				maxmize_tensor_a = tf.py_func(_maximize_a, [tensor_a],tf.float32)
				return maxmize_tensor_a
			
			def _maximize_a(a):
				if a[0]>0:
					maxmize_a = a*1.25
				else:
					maxmize_a = a*0.75
				maxmize_a = maxmize_a.astype(np.float32)
				print('maximize a is', maxmize_a)
				return maxmize_a

			q_target = tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
			def _print_tensor(tensor):
				print(a[0].eval())
				return 1
			tf.py_func(_print_tensor, [q_target],tf.int32)
			# print('q_target is', q_target.eval())
			q_target = maxmize_a(q_target)
			self.q_target = tf.stop_gradient(q_target)

		with tf.variable_scope('q_eval'):
			a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1) # indices shape(46,179)
			self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices) # 根据指定的参数indices来提取params的元素重建出一个tensor
			def _print_tensor(tensor):
				print(a[0].eval())
				return 1
			tf.py_func(_print_tensor, [self.q_eval_wrt_a],tf.int32)
			# print('q_eval_wrt_a is', self.q_eval_wrt_a.eval())

		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name = 'TD_error'))

		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
		print(self.s)

	# ** 
	# * @dev store transition in the memory and keep latest data in the memory
	#
	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s, [a,r], s_))
		index = self.memory_counter % self.memory_size
		# replace the old memory with new memory
		self.memory[index, :] = transition
		self.memory_counter += 1

	def _get_time_hour():
		time_str = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
		hour = int(time_str.split('-')[3])
		return hour

	# ** 
	# * @dev choose an action by inputing observation into evaluation NN
	# * @pram observation The variables to be input into evalution to output an action
	# * @return action The action output
	# *
	def choose_action(self, observation, env_for_ac_choose):
		
		action_dict = {22:297,23:298,24:299,25:300,29:300,26:349}
		action_output2tem = {297:22,298:23,299:24,300:25,301:30,349:26,}
		
		if env_for_ac_choose[1]<=23.5:
			action_value = 4
			action_tem = self.action_space[action_value]
			return action_value, 349
		t1,t2 = 25.6, 26
		time_str = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
		hour = int(time_str.split('-')[3])

		tem_in = env_for_ac_choose[0]
		tem_out = env_for_ac_choose[1]
		tem_out_pre = env_for_ac_choose[2:-2]
		action_output_last = int(env_for_ac_choose[-1])

		action_tem_last = action_output2tem[action_output_last]
		if  np.random.uniform() < self.epsilon:
			observation = np.array(observation)
			observation = observation[np.newaxis,:]
			# action_value: 0-5
			action_value = np.argmax(self.sess.run(self.q_eval, feed_dict={self.s: observation}))
			action_tem = self.action_space[action_value]
			action_output = action_dict[action_tem]
		else:
			action_value = random.randint(0,4)
			action_tem = self.action_space[action_value]
			action_output = action_dict[action_tem]
			return action_value, action_output		


	# ** 
	# * @dev train parameters of evaluation NN
	# *
	def learn(self):
		# replace network prameters
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.target_replace_op)
			print('\ntargrt_params_replaced\n')

		# sample batch memory from all memory
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		# 
		_, cost = self.sess.run(
			[self._train_op, self.loss],
			feed_dict={
				self.s: batch_memory[:, :self.n_features],
				self.a: batch_memory[:, self.n_features],
				self.r: batch_memory[:, self.n_features+1],
				self.s_: batch_memory[:, -self.n_features:],
			})

		self.cost_his.append(cost)
		time_str = time.strftime('%Y-%m-%d',time.localtime(time.time()))
		year_str = time_str.split('-')[0][-2:]
		month_str = time_str.split('-')[1]
		day_str = time_str.split('-')[2]
		save_path = self.saver.save(self.sess,"../../data/dqn_model/dqn_model_{}{}{}.ckpt".format(year_str,month_str,day_str),write_meta_graph=False)
		
		# increase epsilon
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1
	
	# ** 
	# * @dev plot the cost of a episode
	# *
	def plot_cost(self):
		import matplotlib.pyplot as plt 
		plt.plot(np.arange(len(self.cost_his)),self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()

if __name__ == '__main__':
	DQN = DeepQNetwork(n_actions =65 ,n_features = 8, output_graph = True)
####if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。