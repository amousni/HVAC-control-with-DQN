import numpy as np
import time
import sys
from Data_interaction import DataInteraction
from prediction_algo import Prediction 
from caiyun_reg import Regression
import warnings
warnings.filterwarnings("ignore")
# * @dev get observation, reward of a transition
class env(object):
	def __init__(self):
		
		self.n_features = 13
		self.DI = DataInteraction()
		self.Pre = Prediction()
		self.Reg = Regression()
		self.n_actions = 5 #self.DI.getActionCount()
	# **
	# * @dev  
	# * @pram
	# * @pram
	# * @pram col_tem, col_hum, col_tem_out, col_hum_out, col_tem_pre, col_hum_pre, col_action The columns of 			variables
	# * @return
	# * @
	def step(self, id, step, data, col_tem, 
			col_hum, col_tem_out, col_hum_out, col_sol_out, 
			col_tem_pre, col_hum_pre, col_sol_pre, col_action, action_output):
		
		# predict tem&hum outside
		pre_15min = self.Pre.get_res()
		pre_tem_15min = pre_15min[0]
		pre_hum_15min = pre_15min[1]
		self.DI.pushData(id, pre_tem_15min, 'tem_pre')
		self.DI.pushData(id, pre_hum_15min, 'hum_pre')
		print('15min prediction data get and tem is {}, hum is {}\n'.format(pre_tem_15min, pre_hum_15min))
		# get caiyun api realtime and 3h prediction data
		caiyun_data = self.Reg.get_res()
		# caiyun_rl = caiyun_data[0].reshape([2]).tolist()
		caiyun_3h = caiyun_data[1][:3,:].reshape([6]).tolist()
		self.DI.pushData(id, caiyun_3h[0], 'T_1h')
		self.DI.pushData(id, caiyun_3h[1], 'H_1h')
		self.DI.pushData(id, caiyun_3h[2], 'T_2h')
		self.DI.pushData(id, caiyun_3h[3], 'H_2h')
		self.DI.pushData(id, caiyun_3h[4], 'T_3h')
		self.DI.pushData(id, caiyun_3h[5], 'H_3h')
		print('caiyun 3h prediction data get and tem is {}... , hum is {}...\n'.format(caiyun_3h[0], caiyun_3h[1]))
		# initial variables of observation
		s_tem = data[0][col_tem]
		s_hum = data[0][col_hum]
		s_tem_out = data[0][col_tem_out]
		s_hum_out = data[0][col_hum_out]
		s_sol_out = data[0][col_sol_out]

		env_for_ac_choose = np.array([s_tem,s_tem_out, pre_tem_15min,caiyun_3h[0],caiyun_3h[2],caiyun_3h[4], action_output])

		observation_ = [s_tem,s_hum,s_tem_out,s_hum_out,s_sol_out,\
						pre_tem_15min,pre_hum_15min] + caiyun_3h 
		# print('observation data generated and length is {}'.format(len(observation_)))
		reward_tem = s_tem - 25.5 if (s_tem -25.5) > 0 else 0
		reward_ener = self.DI.cal_power(id)
		reward =-(reward_ener / 0.175 /2 + reward_tem / 2 / 2)

		if step < 10000:
			done = False
		else:
			done = True 

		return observation_, reward, done, env_for_ac_choose

if __name__ == '__main__':
	env = env()
