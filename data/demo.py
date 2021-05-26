# coding=utf-8

import pandas as pd

file_path = r'D:\kaikeba_resources\02_NLP项目实战\01_摘要自动生成\02_项目导论中与中文词向量实战\summary\data'

file_name_train = file_path + r'\train.csv'
file_name_test  = file_path + r'\test.csv'

dataframe1 = pd.read_csv(file_name_train, encoding='utf-8')
dataframe2 = pd.read_csv(file_name_test, encoding='utf-8')

df_1 = dataframe1[:300]
df_2 = dataframe2[:300]
df_1.to_csv(r'train.csv', encoding='utf-8-sig', index=False)
df_2.to_csv(r'test.csv', encoding='utf-8-sig', index=False)
print('...')
