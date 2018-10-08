import pandas as pd
import numpy as np
import time


coin_list = ['bitcoin','ethereum','ripple','bitcoin-cash',
'eos','stellar','litecoin','tether','monero']

for coin in coin_list:
    # read  data form coinmarketcap
    market_info = pd.read_html("https://coinmarketcap.com/currencies/"+coin+"/historical-data/?start=20170701&end="+time.strftime("%Y%m%d"))[0]
    market_info = market_info.assign(Date=pd.to_datetime(market_info['Date']))
    # when Volume is equal to '-' convert it to 0
    # market_info.loc[market_info['Volume']=="-",'Volume']=0
    # convert to int
    market_info['Volume'] = market_info['Volume'].astype('int64')
    #save data to csv file
    market_info.to_csv("data/"+coin+".csv")

market_dic = {}

for coin in coin_list:
    # read  data form coinmarketcap
    df = pd.read_csv("data/"+coin+".csv")
    market_dic[coin+"_volume"] = df['Volume'].values[-441:]
    market_dic[coin+"_close"] = df['Close**'].values[-441:]
for col in market_dic:
	print(len(market_dic[col]))
final_df = pd.DataFrame(data=market_dic)
final_df.to_csv("data/final.csv",index=False)


