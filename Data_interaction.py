import numpy as np
import pymssql
import os,time
import pandas as pd
import json
import csv
import codecs
from io import StringIO
from urllib.request import Request, urlopen
import warnings
warnings.filterwarnings("ignore")

# * @dev data interaction between sql and program
class DataInteraction(object):
	def __init__(
		self,
		server = "localhost",
		user = "**",
		password = "**",
		):
		# define variable to store IP/account/password
		self.server = server 
		self.user = user
		self.password = password
		self.token = '**'
		self.lon = 113.9777700000
		self.lat = 22.5924260000

	def getWeatherForecast(self):
		time_str = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
		df = pd.DataFrame(np.zeros((1,1)))
		df['year'] = time_str.split('-')[0]
		df['month'] = time_str.split('-')[1]
		df['day'] = time_str.split('-')[2]
		df['hour'] = int(time_str.split('-')[3])+1
		df['timestamp'] = time.mktime(time.localtime(time.time())) + 3540
				
		url_forcast =  "https://api.caiyunapp.com/v2/{}/{},{}/forecast.json".format(self.token, self.lon, self.lat)
		maxTryNum = 10
		for tries in range(maxTryNum):
			try:
				json_response = urlopen(url_forcast).read()
				json_data = json.loads(json_response.decode('utf-8'))
				if json_data.get('status') == 'ok':
					for i in range(6):
						df['To_forecast{}'.format(i+1)] = json_data.get('result').get('hourly').get('temperature')[i].get('value')
						df['Ho_forecast{}'.format(i+1)] = json_data.get('result').get('hourly').get('humidity')[i].get('value')*100 
						df['So_forecast{}'.format(i+1)] = json_data.get('result').get('hourly').get('dswrf')[i].get('value')
						df['Co_forecast{}'.format(i+1)] = json_data.get('result').get('hourly').get('cloudrate')[i].get('value')
						# print(tem ,hum, rad, cloud)
				else:
					print(json_data.get('error'))
				df.drop(columns=[0],inplace=True)
				return df
			except:
				if tries < (maxTryNum-1):
					time.sleep(1.5)
					continue
				else:
					print("Has tried %d times to access url %s, all failed!",maxTryNum,url_forcast)
    
	def getWeatherRealtime(self):
		time_str = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
		df = pd.DataFrame(np.zeros((1,1)))
		df['year'] = time_str.split('-')[0]
		df['month'] = time_str.split('-')[1]
		df['day'] = time_str.split('-')[2]
		df['hour'] = int(time_str.split('-')[3])
		df['minute'] = int(time_str.split('-')[4])
		df['timestamp'] = time.mktime(time.localtime(time.time()))

		url_realtime = "https://api.caiyunapp.com/v2/{}/{},{}/realtime.json".format(self.token, self.lon, self.lat)
		maxTryNum = 10
		for tries in range(maxTryNum):
			try:		
				json_response = urlopen(url_realtime).read()
				json_data = json.loads(json_response.decode('utf-8'))
				if json_data.get('status') == 'ok':
					df['To_cy'] = json_data.get('result').get('temperature')     
					df['Ho_cy'] = json_data.get('result').get('humidity')*100
					df['So_cy'] = json_data.get('result').get('dswrf')
					df['Co_cy'] = json_data.get('result').get('cloudrate')
				else:
					print(json_data.get('error'))
				df.drop(columns=[0],inplace=True)
				return df     
			except:
				if tries < (maxTryNum-1):
					time.sleep(1.5)
					continue
				else:
					print("Has tried %d times to access url %s, all failed!",maxTryNum,url_realtime)
    
	def getDf(self, rows=1000):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute('select top %d * from dbo.BaseData_Y2019 order by datew desc'%(rows))
		data = self.cursor.fetchall()[0:rows][:]
		cols = self.cursor.description
		col = []
		for i in cols:
			col.append(i[0])
		df = pd.DataFrame(data,columns=col)
        
		return df  
	# **
	# * @dev get dataframe from sql
	# * @param timeStep The step main algorithm runs
	# * @param col The step main algorithm runs
	# * @param startId The timestamp of data in step 0
	# * @param num The number of rows of table in sql
	# * @return data The data read from sql
	# * 
	def getData(self,timeStep,step0,col,startId,num = 2):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master',charset='utf8')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute("select top %d * from dbo.BaseData_Y2019 order by datew desc"%(num))
		data = self.cursor.fetchall()
		# transfer date to timestamp
		timeStamp = data[0][1].timestamp()
		# loop break condition:timestamp >= expected timestamp of this step ; value of column col is not None；
		while (not(timeStamp >= (timeStep - step0)* 900 + startId and data[0][col] != None)):
			self.cursor.execute("select top %d * from dbo.BaseData_Y2019 order by datew desc"%(num))
			time.sleep(15)
			data = self.cursor.fetchall()
			timeStamp = data[0][1].timestamp()
		self.conn.close()  
		return data

	# **
	# * @dev write data into sql
	# * @param col The column of table in sql
	# * @param col The step main algorithm runs
	# * @param startId The timestamp of data in step 0
	# * @param num The number of rows of data in sql
	# * @return True Bool represent whether data is successfully pushed
	# * 
	def pushData(self,id,value,col_name):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute("update dbo.BaseData_Y2019 set %s = %f where id = %d"%(col_name, value, id))
		self.conn.commit()
		self.conn.close()  
		return True

	# **
	# * @dev get the timestamp of row 1 in the table in step 0 
	# * @return The timestamp of row 1 in the table in step 0
	# *
	def getStartId(self):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute("select top 1 * from dbo.BaseData_Y2019 order by datew desc")
		step = self.cursor.fetchone()
		startId = int(step[1].timestamp())
		self.conn.close()
		return startId

	def getStartStep(self):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute("select top 3000 * from dbo.BaseData_Y2019 order by datew desc")
		data = self.cursor.fetchall()
		for i in range(3000):
			if (data[i][23] != None):
				return data[i][23]
		return 0

	def getDfAll(self):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute('select * from dbo.BaseData_Y2019 order by datew desc')
		data = self.cursor.fetchall()
		cols = self.cursor.description
		col = []
		for i in cols:
			col.append(i[0])
		df = pd.DataFrame(data,columns=col)       
		return df

	# def getDf(self, n=1000):
	# 	self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
	# 	self.cursor = self.conn.cursor()
	# 	self.table = self.cursor.execute("use maGroup")
	# 	self.cursor.execute('select top {} * from dbo.BaseData_Y2019 order by datew desc'.format(n))
	# 	data = self.cursor.fetchall()
	# 	cols = self.cursor.description
	# 	col = []
	# 	for i in cols:
	# 		col.append(i[0])
	# 	df = pd.DataFrame(data,columns=col)       
	# 	return df

	# **
	# * @dev get a certain row of data in table if id equals to input value
	# * @param id The id of inquired row
	# * @return The row of table inquired
	# * 
	def getDataById(self, id):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute('select top 1 * from dbo.BaseData_Y2019 where id = %d '%(id))
		data = self.cursor.fetchall()
		return data  	

	# **
	# * @dev get amounts of action in table
	# * @return the amounts of action
	# * 
	def getActionCount(self):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute("select max(act)-min(act) as rows from actQuery")
		row = self.cursor.fetchone()
		count = int(row[0] + 1)
		self.conn.close()
		return count
	
	def cal_power(self,id):
		self.conn = pymssql.connect(self.server,self.user,self.password,database='master')
		self.cursor = self.conn.cursor()
		self.table = self.cursor.execute("use maGroup")
		self.cursor.execute('select top 100 * from dbo.BaseData_Y2019 where id <= %d order by datew desc '%(id))
		data = self.cursor.fetchall()
		data = pd.DataFrame(data=data)
		data = data.iloc[:,[0,1,31]]
		data.columns = ['id', 'date','C01_Watt']
		data['minute'] = data['date'].apply(lambda x: x.strftime('%M')).astype('int')
		rows = [x for x in range(len(data)-1) if data.iloc[x]['minute'] == data.iloc[x+1]['minute'] ]
		data.drop(rows, inplace=True)
		elec_kwh = sum(data.iloc[:15]['C01_Watt'].values) / 4 /1000
		self.conn.close()
		return elec_kwh


if __name__ == '__main__':
	DI = DataInteraction()
