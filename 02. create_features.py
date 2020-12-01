# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:42:23 2020
@author: ukjo
"""
import json
import pymysql
import pandas as pd

df = pd.read_csv('news_df.csv')
df.rename(columns={' title': 'title'}, inplace=True)  # 황당..
vocab = []

# todo : 이 부분에서 필요없는 문자열, 정보성 없는 토큰들 제거하는 로직이 추가되어야 함
df['title'] = df.title.str.replace('\\', ' ')
df['title'] = df.title.str.lower()
df['title'] = df.title.str.split()

for t in df.title:
    for word in t:
        vocab.append(word)
vocab = list(set(vocab))

wc = {unique_word: [0] * len(df.title) for unique_word in vocab}

for idx, t in enumerate(df.title):
    for word in t:
        wc[word][idx] += 1

wc_df = pd.DataFrame(wc)
wc_df.head()

train_df = pd.concat([wc_df, df], axis=1)
train_df.to_csv('features.csv')

