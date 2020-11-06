from Data_interaction import DataInteraction as DI 
import time, os
import warnings
warnings.filterwarnings("ignore")

def _get_time_string():
    return time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

on = int(input('Please input 1 to turn on weather data collection:'))
timeStep_forcast = 0
timeStamp = int(time.time())
while (on==1):
    time_min = int(_get_time_string().split('-')[4])
    
    if (time_min>0 and time_min<2 and int(time.time())>=timeStep_forcast*3600+timeStamp):
        df_forecast = DI().getWeatherForecast()
        df_forecast.to_csv('../../data/caiyun/6h_forcast_cy.csv', mode='a', header=False, index=False)
        timeStep_forcast += 1 
        print(_get_time_string() + ' :df_forecast data collected' )
    
    df_realtime = DI().getWeatherRealtime()
    df_realtime.to_csv('../../data/caiyun/30sec_realtime_cy.csv', mode='a', header=False, index=False)
    print(_get_time_string() + ' :df_realtime data collected')
    time.sleep(30)
    
    
