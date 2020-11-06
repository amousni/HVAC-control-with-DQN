from Data_interaction import DataInteraction as DI 
import time, os
import warnings
warnings.filterwarnings("ignore")

on = int(input('Please input 1 to turn on env outdoor data collection:'))

while (on==1):
    time_str = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
    year_str = time_str.split('-')[0][-2:]
    month_str = time_str.split('-')[1]
    day_str = time_str.split('-')[2]
    hour_str = time_str.split('-')[3]
    min_str = time_str.split('-')[4]
    if int(day_str) in [1,8,15,22]:
        df =  DI().getDfAll()
        data = df.iloc[:,[0,1,3,4,6,8,9,24]]
        data.columns = ['id', 'date','To','Ho','So','Ti','Hi','CTL_1']
        data.loc[:,'timestamp'] = data.loc[:,'date'].apply(lambda x: x.timestamp())
        data.loc[:,'year'] = data.loc[:,'date'].apply(lambda x: x.strftime('%Y')).astype('int')
        data.loc[:,'month'] = data.loc[:,'date'].apply(lambda x: x.strftime('%m')).astype('int')
        data.loc[:,'day'] = data.loc[:,'date'].apply(lambda x: x.strftime('%d')).astype('int')
        data.loc[:,'hour'] = data.loc[:,'date'].apply(lambda x: x.strftime('%H')).astype('int')
        data.loc[:,'minute'] = data.loc[:,'date'].apply(lambda x: x.strftime('%M')).astype('int')
        # df_hour = data[data['minute']==0].reset_index(drop=True)
        df_15min = data[data['minute']%15 == 0].reset_index(drop=True)
        df_10min = data[data['minute']%10 == 0].reset_index(drop=True)
        df_5min = data[data['minute']%5 == 0].reset_index(drop=True)
        df_1min = data[data['minute']%1 == 0].reset_index(drop=True)
        print('df generated..\n')
        rows = [x for x in range(len(df_15min)-1) if df_15min.iloc[x]['minute'] == df_15min.iloc[x+1]['minute'] ]
        df_15min = df_15min[~df_15min.index.isin(rows)]
        rows = [x for x in range(len(df_10min)-1) if df_10min.iloc[x]['minute'] == df_10min.iloc[x+1]['minute'] ]
        df_10min = df_10min[~df_10min.index.isin(rows)]
        rows = [x for x in range(len(df_5min)-1) if df_5min.iloc[x]['minute'] == df_5min.iloc[x+1]['minute'] ]
        df_5min = df_5min[~df_5min.index.isin(rows)]
        rows = [x for x in range(len(df_1min)-1) if df_1min.iloc[x]['minute'] == df_1min.iloc[x+1]['minute'] ]
        df_1min = df_1min[~df_1min.index.isin(rows)]
        time_label_str = year_str + month_str+day_str
        df_5min.to_csv('../../data/env/5min_{}.csv'.format(time_label_str), encoding='utf-8',index=0)
        df_10min.to_csv('../../data/env/10min_{}.csv'.format(time_label_str), encoding='utf-8',index=0)
        df_15min.to_csv('../../data/env/15min_{}.csv'.format(time_label_str), encoding='utf-8',index=0)
        df_1min.to_csv('../../data/env/1min_{}.csv'.format(time_label_str), encoding='utf-8',index=0)
        print('env outdoor 1/5/10/15 min data saved..\n')
    time.sleep(86400)