from Env import env
import numpy as np
import pandas as pd
import time
from RL_brain import DeepQNetwork
from prediction_algo import Prediction
from Data_interaction import DataInteraction
from caiyun_reg import Regression
import warnings
warnings.filterwarnings("ignore")
# **
# * @dev run the whole algorithm of prediction control of HAVC
# * @param num The number of dataset for prediction.
# *
def run_env(num, step):
    step0 = step
    step = step0
    # get start timestamp
    startId = DI.getStartId()
    col_tem = 8
    col_hum = 9
    col_tem_out = 3
    col_hum_out = 4
    col_sol_out = 6
    col_tem_pre = 20 # col_name is "tem_pre"
    col_hum_pre = 21 # col_name is "hum_pre"
    col_sol_pre = 22 # col_name is "sol_pre"
    col_action = 24  # col_name is "CTL_1" (80w-400w)
    col_timeStep = 23
    col_ElecP = 31

    id_his = []
    # get 2 rows data if time margin from step0 over (step-step0)*15min
    data = DI.getData(step,step0,col_tem,startId,num)
    id = data[0][0]
    id_his.append(id)

    DI.pushData(id, step, 'timeStep')
    df_step = pd.DataFrame(data = [step], columns=['step'])
    df_step.to_csv('../../data/step.csv', mode='a', header=False, index=False)
    print("In step {} step index saved..\n".format(step))

    # predict tem&hum outside
    pre_15min = Pre.get_res()
    pre_tem_15min = pre_15min[0]
    pre_hum_15min = pre_15min[1]
    DI.pushData(id, pre_tem_15min, 'tem_pre')
    DI.pushData(id, pre_hum_15min, 'hum_pre')
    print('15min prediction data get and tem is {}, hum is {}\n'.format(pre_tem_15min, pre_hum_15min))
    # get caiyun api realtime and 3h prediction data
    caiyun_data = Reg.get_res()
    # caiyun_rl = caiyun_data[0].reshape([2]).tolist()
    caiyun_3h = caiyun_data[1][:3,:].reshape([6]).tolist()
    DI.pushData(id, caiyun_3h[0], 'T_1h')
    DI.pushData(id, caiyun_3h[1], 'H_1h')
    DI.pushData(id, caiyun_3h[2], 'T_2h')
    DI.pushData(id, caiyun_3h[3], 'H_2h')
    DI.pushData(id, caiyun_3h[4], 'T_3h')
    DI.pushData(id, caiyun_3h[5], 'H_3h')
    print('caiyun 3h prediction data get and tem is {}... , hum is {}...\n'.format(caiyun_3h[0], caiyun_3h[1]))
    # initial variables of observation
    s_tem = data[0][col_tem]
    s_hum = data[0][col_hum]
    s_tem_out = data[0][col_tem_out]
    s_hum_out = data[0][col_hum_out]
    s_sol_out = data[0][col_sol_out]
    action_output = 349
    env_for_ac_choose = np.array([s_tem, s_tem_out, pre_tem_15min, caiyun_3h[0],caiyun_3h[2],caiyun_3h[4],action_output])

    observation = [s_tem,s_hum,s_tem_out,s_hum_out,s_sol_out,\
                    pre_tem_15min,pre_hum_15min] + caiyun_3h 
    # print('observation data generated and length is {}'.format(len(observation)))
    # feed observation into eval net and generate action control
    action_, action_output = RL.choose_action(observation, env_for_ac_choose)
    time_str = time.strftime('%Y-%m-%d-%H-%M: ', time.localtime(time.time()))
    print(time_str+'step %d: got action %d\n'%(step,action_output))
    # write action into sql
    DI.pushData(id, action_output, 'CTL_1')
    df_test = pd.DataFrame(data = {'id':[id],'time':[time_str]})
    df_test.to_csv('../../data/test.csv', mode='a', header=False, index=False)
    
    step += 1
    df_step = pd.DataFrame(data = [step], columns=['step'])
    df_step.to_csv('../../data/step.csv', mode='a', header=False, index=False)
    print("move to step {}\n".format(step))
    startId = DI.getStartId()
    while True:
        data = DI.getData(step, step0,col_tem, startId, num)
        time_str = time.strftime('%Y-%m-%d-%H-%M: ', time.localtime(time.time()))
        print(time_str+'step %d: got data\n'%(step))
        # get observation in this step and reward, action, observation last step
        id = data[0][0]
        id_his.append(id)
        DI.pushData(id, step, 'timeStep')
        observation_, reward, done, env_for_ac_choose = env.step(id, step, data, 
                                            col_tem, col_hum, col_tem_out, 
                                            col_hum_out, col_sol_out, col_tem_pre, 
                                            col_hum_pre, col_sol_pre, col_action,int(np.array(action_output)))

        # store data of transition in mini-batch memory
        RL.store_transition(observation, int(np.array(action_)), reward, observation_)
        time_str = time.strftime('%Y-%m-%d-%H-%M: ', time.localtime(time.time()))
        print(time_str+'step %d:transaction stored\n'%(step))
        # after 50 steps, neutral network learns once five steps
        if (step >= 50) and (step % 5 == 0): 
            RL.learn()

        # generate action control in current step
        action_, action_output = RL.choose_action(observation_, env_for_ac_choose)
        time_str = time.strftime('%Y-%m-%d-%H-%M: ', time.localtime(time.time()))
        print(time_str+'step %d: got action %d\n'%(step,action_output))
        df_test = pd.DataFrame(data =  {'id':[id],'time':[time_str]})
        df_test.to_csv('../../data/test.csv', mode='a', header=False, index=False)
        # write action into sql
        DI.pushData(id, action_output, 'CTL_1')
        observation = observation_.copy()
        # break loop when end of this episode
        if done:
            break
        step += 1
        df_step = pd.DataFrame(data = [step], columns=['step'])
        df_step.to_csv('../../data/step.csv', mode='a', header=False, index=False)

    print('\nepisode over')

if __name__ == "__main__":
    # load step
    isRestart = int(input("is start from 0 step please input 0 or 1,\n if you not sure, input 0!\n:"))
    if isRestart == 1:
        step_start=0
    else: step_start = int(pd.read_csv('../../data/step.csv').iloc[-1].step)
    print("start step is: {}\n".format(step_start))
    env = env()
    DI = DataInteraction()
    # step = DI.getStartStep()
    Pre = Prediction()
    Reg = Regression()
    RL = DeepQNetwork(env.n_actions, env.n_features, step_start
                      # learning_rate=0.01,
                      # reward_decay=0.9,
                      # e_greedy=0.9,
                      # replace_target_iter=200,
                      # memory_size=2000,
                      # output_graph=True
                      )
    run_env(2, step_start)
    # RL.plot_cost()
