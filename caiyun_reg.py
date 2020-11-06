import math
from math import sqrt
from numpy import concatenate
from pandas import concat
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time, os
from pandas import read_csv
from datetime import datetime
import numpy as np
import pandas as pd
from Data_interaction import DataInteraction
import random
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
import xgboost as xgb
from sklearn.metrics import f1_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

class Regression():   
    def __init__(self):    
        self.params  = {'eta': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.7, 'alpha': 0.2,
                        'objective':'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': -1,
                        'scale_pos_weight ':1,
                        }    
        self.DI = DataInteraction()
        
    def get_res(self):
        self.train_model()
        res_now, res_pre =  self.predict() # 输出：温度 - col0, 湿度 - col1
        # print('regression outdoor indicators:{}\n'.format(res_now))
        # print('predicted 6 hours outdoor indicators:{}\n'.format(res_pre))
        return res_now, res_pre


    def train_model(self):
        time_str = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
        year_str = time_str.split('-')[0][-2:]
        month_str = time_str.split('-')[1]
        day_str = time_str.split('-')[2]
        hour_str = time_str.split('-')[3]
        min_str = time_str.split('-')[4]
        # day in 2, 9, 16,23 update prediction model
        if (int(day_str) in [2,9,16,23]) and (int(hour_str) == 0) and (int(min_str)<=20):
            self._train_regression()


# @dev load xgboost models and generate output based on these models
    def predict(self):
#         DI = DataInteraction
        time_str = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
        year_str = time_str.split('-')[0][-2:]
        month_str = time_str.split('-')[1]
        day_str = time_str.split('-')[2]

        df =  self.DI.getDf(1000)
        data = df.iloc[:,[0,1,3,4,6]]
        data.columns = ['id', 'date','To','Ho','So']
        data.loc[:,'timestamp'] = data.loc[:,'date'].apply(lambda x: x.timestamp())
        data.loc[:,'year'] = data.loc[:,'date'].apply(lambda x: x.strftime('%Y')).astype('int')
        data.loc[:,'month'] = data.loc[:,'date'].apply(lambda x: x.strftime('%m')).astype('int')
        data.loc[:,'day'] = data.loc[:,'date'].apply(lambda x: x.strftime('%d')).astype('int')
        data.loc[:,'hour'] = data.loc[:,'date'].apply(lambda x: x.strftime('%H')).astype('int')
        data.loc[:,'minute'] = data.loc[:,'date'].apply(lambda x: x.strftime('%M')).astype('int')
        # df_hour = data[data['minute']==0].reset_index(drop=True)
        rl_sql_1h = data[data['minute'] == 0].reset_index(drop=True).drop_duplicates(subset=['hour'])
        rl_sql_1h['To_1']=rl_sql_1h['To'].shift(-1)
        rl_sql_1h['Ho_1']=rl_sql_1h['Ho'].shift(-1)
        rl_sql_1h['So_1']=rl_sql_1h['So'].shift(-1)
        rl_sql_1h['To_2']=rl_sql_1h['To'].shift(-2)
        rl_sql_1h['Ho_2']=rl_sql_1h['Ho'].shift(-2)
        rl_sql_1h['So_2']=rl_sql_1h['So'].shift(-2)
        rl_sql_1h=rl_sql_1h.iloc[0:1].reset_index(drop=True)
        #         df_1min_sql = pd.read_csv('../../data/env/1min_190502.csv')[['To','Ho','So','year','month','day','hour','minute']]
        #         rl = pd.read_csv('../../data/caiyun/30sec_realtime_cy.csv')
        for_6h = pd.read_csv('../../data/caiyun/6h_forcast_cy.csv').iloc[-1:]

        for_1h = for_6h[['month','day','hour','To_forecast1','Ho_forecast1', 'Co_forecast1', 
                            'So_forecast1']].reset_index(drop=True)
        for_2h = for_6h[['month','day','hour','To_forecast2','Ho_forecast2', 'Co_forecast2', 
                            'So_forecast2']].reset_index(drop=True)
        for_3h = for_6h[['month','day','hour','To_forecast3','Ho_forecast3', 'Co_forecast3', 
                            'So_forecast3']].reset_index(drop=True)

        model_To = []
        model_Ho = []
        for month in range(int(month_str), 0, -1):
            for day in range(31, 0, -1):
                month = '0' + str(month) if len(str(month)) == 1 else str(month)
                day = '0' + str(day) if len(str(day)) == 1 else str(day)
                filename_model = '1min_To_0_0_' + year_str + month + day +'.model'
                if os.path.exists("../../data/xgb_model/"+filename_model):
                    for i in range(3):
                        for j in range(3):
                            model_To.append(xgb.Booster(model_file='../../data/xgb_model/1min_To_{}_{}_{}{}{}.model'.format(i,j,year_str,month,day)))
                            model_Ho.append(xgb.Booster(model_file='../../data/xgb_model/1min_Ho_{}_{}_{}{}{}.model'.format(i,j,year_str,month,day)))

        for_1h = pd.concat([for_1h, rl_sql_1h[['To','So','Ho','To_1','So_1','Ho_1','To_2',
                                               'So_2','Ho_2']]],axis=1)

        output_now = np.zeros([1,2])
        output_pre = np.zeros([3,2])

        input = xgb.DMatrix(for_1h[['month','day','hour','To_forecast1','Ho_forecast1','Co_forecast1',\
                                   'So_forecast1','To','Ho','So','To_1','Ho_1','So_1','To_2','Ho_2','So_2']])
        for i in range(len(model_To)):
                output_pre[0,0] += model_To[i].predict(input)/len(model_To)
        for i in range(len(model_Ho)):
                output_pre[0,1] += model_Ho[i].predict(input)/len(model_Ho)

        for_2h['To'] = output_pre[0,0]
        for_2h['Ho'] = output_pre[0,1]
        for_2h['So'] = for_2h['So_forecast2']/(for_1h['So_forecast1'] +1)*for_1h['So']
        for_2h['To_1'] = for_1h['To']
        for_2h['Ho_1'] = for_1h['Ho']
        for_2h['So_1'] = for_1h['So']

        for_2h['To_2'] = for_1h['To_1']
        for_2h['Ho_2'] = for_1h['Ho_1']
        for_2h['So_2'] = for_1h['So_1']
        for_2h = for_2h.rename(columns=lambda x:x.replace('st2','st1'))

        input_2 = xgb.DMatrix(for_2h[['month','day','hour','To_forecast1','Ho_forecast1','Co_forecast1','So_forecast1','To','Ho','So','To_1','Ho_1','So_1','To_2','Ho_2','So_2']])
        for i in range(len(model_To)):
                output_pre[1,0] += model_To[i].predict(input_2)/len(model_To)
        for i in range(len(model_Ho)):
                output_pre[1,1] += model_Ho[i].predict(input_2)/len(model_Ho)

        for_3h['To'] = output_pre[1,0]
        for_3h['Ho'] = output_pre[1,1]
        for_3h['So'] = for_3h['So_forecast3']/(for_2h['So_forecast1'] +1)*for_2h['So']
        for_3h['To_1'] = for_2h['To']
        for_3h['Ho_1'] = for_2h['Ho']
        for_3h['So_1'] = for_2h['So']

        for_3h['To_2'] = for_2h['To_1']
        for_3h['Ho_2'] = for_2h['Ho_1']
        for_3h['So_2'] = for_2h['So_1']
        for_3h = for_3h.rename(columns=lambda x:x.replace('st3','st1'))

        input_3 = xgb.DMatrix(for_3h[['month','day','hour','To_forecast1','Ho_forecast1','Co_forecast1','So_forecast1','To','Ho','So','To_1','Ho_1','So_1','To_2','Ho_2','So_2']])
        for i in range(len(model_To)):
                output_pre[2,0] += model_To[i].predict(input_3)/len(model_To)
        for i in range(len(model_Ho)):
                output_pre[2,1] += model_Ho[i].predict(input_3)/len(model_Ho)
        return output_now, output_pre

    def _train_xgb_model(self, trainset, label, en_amount=3, NFOLDS=3
                ):
        time_str = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
        year_str = time_str.split('-')[0][-2:]
        month_str = time_str.split('-')[1]
        day_str = time_str.split('-')[2]
        for seed in range(en_amount):
            train_data_use = trainset.drop(columns=[label]).reset_index(drop=True)
            train_label = trainset[label].reset_index(drop=True)
            train_label_index = train_label.astype('int') ### ?

            kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
            kf = kfold.split(train_data_use, train_label_index)
            for i, (trn_idx, val_idx) in enumerate(kf):
                trn_data = xgb.DMatrix(train_data_use.iloc[trn_idx,:], train_label[trn_idx])
                val_data = xgb.DMatrix(train_data_use.iloc[val_idx,:], train_label[val_idx])
                watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
                clf = xgb.train(dtrain=trn_data, num_boost_round=10000, evals=watchlist, early_stopping_rounds=200,
                                verbose_eval=1000, params=self.params)
                clf.save_model('../../data/xgb_model/1min_{}_{}_{}_{}{}{}.model'.format(label,seed,i,year_str,month_str,day_str))
        print('xgb model saved..\n')

    # **
    # * @dev train SVR prediction model and use it to predict
    # * @pram col The column of variable to be predicted
    # * @pram data The dataset for training and prediction
    # * @return linear_svr_y_predict The outcome of prediction
    #
    def _train_regression(self):
        time_str = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        year_str = time_str.split('-')[0][-2:]
        month_str = time_str.split('-')[1]
        day_str = time_str.split('-')[2]
        filename_data_ = ""
        dataset = pd.DataFrame()
        for month in range(int(month_str), 0, -1):
            for day in range(31, 0, -1):
                month = '0' + str(month) if len(str(month)) == 1 else str(month)
                day = '0' + str(day) if len(str(day)) == 1 else str(day)
                filename_data = '1min_' + year_str + month + day +'.csv' 
                # print(filename_data)
                try:    
                    dataset = read_csv('../../data/env/'+ filename_data,  index_col=0)
                    filename_data_ = filename_data
                    break
                except:
                    pass
            if len(dataset) > 0:
                break

        df_1min_sql = dataset[['To','Ho','So','year','month','day','hour','minute']]
        rl = pd.read_csv('../../data/caiyun/30sec_realtime_cy.csv')
        rl_1min = rl_1min = rl[rl['minute']%1 == 0].iloc[:,-23000:].reset_index(drop=True)
        rows = [x for x in range(len(rl_1min)-1) if rl_1min.iloc[x]['minute'] == rl_1min.iloc[x+1]['minute'] ]
        rl_1min.drop(rows, inplace=True)
        rl_1min.reset_index(drop=True)
        rl_sql_1min = rl_1min.merge(df_1min_sql, how='left', on=['year','month','day','hour','minute'])
        rl_sql_1min.dropna(inplace = True)
        df_for_merge_1h = rl_sql_1min[['timestamp','To','Ho','So']].rename(columns={'To':'To_1','Ho':'Ho_1','So':'So_1'})
        df_for_merge_1h['timestamp'] = (df_for_merge_1h['timestamp'] + 3600)/60
        df_for_merge_1h['timestamp'] = df_for_merge_1h['timestamp'].astype('int')

        df_for_merge_2h = rl_sql_1min[['timestamp','To','Ho','So']].rename(columns={'To':'To_2','Ho':'Ho_2','So':'So_2'})
        df_for_merge_2h['timestamp'] = (df_for_merge_2h['timestamp'] + 3600*2)/60
        df_for_merge_2h['timestamp'] = df_for_merge_2h['timestamp'].astype('int')

        df_for_merge_3h = rl_sql_1min[['timestamp','To','Ho','So']].rename(columns={'To':'To_3','Ho':'Ho_3','So':'So_3'})
        df_for_merge_3h['timestamp'] = (df_for_merge_3h['timestamp'] + 3600*3)/60
        df_for_merge_3h['timestamp'] = df_for_merge_3h['timestamp'].astype('int')

        rl_sql_1min['timestamp'] = (rl_sql_1min['timestamp'])/60
        rl_sql_1min['timestamp'] = rl_sql_1min['timestamp'].astype('int')
        rl_sql_1min = pd.merge(rl_sql_1min,df_for_merge_1h,on=['timestamp'],how='left')
        rl_sql_1min = pd.merge(rl_sql_1min,df_for_merge_2h,on=['timestamp'],how='left')
        rl_sql_1min = pd.merge(rl_sql_1min,df_for_merge_3h,on=['timestamp'],how='left')
        rl_sql_1min.dropna(inplace=True)

        dataset_To = rl_sql_1min[['month','day','hour','To_cy','Ho_cy','Co_cy','So_cy','To_1','Ho_1','So_1','To_2','Ho_2','So_2','To_3','Ho_3','So_3','To']]
        dataset_Ho = rl_sql_1min[['month','day','hour','To_cy','Ho_cy','Co_cy','So_cy','To_1','Ho_1','So_1','To_2','Ho_2','So_2','To_3','Ho_3','So_3','Ho']]
        self._train_xgb_model(dataset_To, label='To')
        print('Tem xgb model trained..\n')
        self._train_xgb_model(dataset_Ho, label='Ho')
        print('Hum xgb model trained..\n')


if __name__ == '__main__':
    Reg = Regression()
    # print(Pre._getTrainData())
    Reg.get_res()