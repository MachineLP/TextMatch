import requests
import pandas as pd
import json
import time 


for i in range(100):
    # 构造需要进行推断的数据
    newJson = '{"text":"我在九寨沟,很喜欢"}'
    # 指定ip, 端口
    url = "http://127.0.0.1:5000/invocations"
    # 传递的参数需要从dataframe转化为json格式
    json_data = json.loads( newJson )
    model_input = pd.DataFrame([json_data])
    req_data = model_input.to_json(orient='split')
    headers = {'content-type': 'application/json; format=pandas-split'}
    # 使用POST方式调用REST api
    start_time = time.time()
    respond = requests.request("POST", url, data=req_data, headers=headers) 
    print ("time>>>>>>>", time.time() - start_time)
    print ( "respond>>", respond )
    # 获取返回值
    print (respond.json()) 

'''
{'bow': [['0', 0.27735009448572867], ['1', 0.5303300779349595], ['2', 0.8660253835771797], ['3', 0.0]], 'tfidf': [['0', 0.9999996100001527], ['1', 0.9999996100001527], ['2', 0.9999996100001527], ['3', 0.0]], 'ngram_tfidf': [['0', 0.9999996100001527], ['1', 0.9999996100001527], ['2', 0.9999996100001527], ['3', 0.0]]}
'''