#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
path='Data/CHEXI_NER_Data/negtive_predict.txt'
datas=pd.read_csv(path,header=None,sep='\t')
datas.columns=['MODEL_STANDARD_NAME','CATE_NAME','PREDICT']
# print(datas.head())
datas.drop_duplicates(subset=['CATE_NAME','PREDICT'],keep='first',inplace=True)
num=[i for i in range(1,len(datas)+1)]
datas['ID']=num
datas=pd.DataFrame(datas,columns=['ID','MODEL_STANDARD_NAME','CATE_NAME','PREDICT'])
datas.to_excel('Data/CHEXI_NER_Data/quchong_negtive_predict.xlsx',index=None,encoding='utf-8')
