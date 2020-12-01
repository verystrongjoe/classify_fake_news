# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:42:23 2020
@author: ukjo
"""
import json
import pandas as pd
import re

train_df = pd.read_csv('features.csv')

vocab = []
# remind vocab
for t in train_df.title:
    for word in t:
        vocab.append(word)
vocab = list(set(vocab))

# Isolating fake and real news first
real_df = train_df[train_df['is_valid'] == '1']
fake_df = train_df[train_df['is_valid'] == '0']

# P(real) and P(fake)
p_real = len(real_df) / len(train_df)
p_fake = len(fake_df) / len(train_df)

# count of real news
n_words_per_real_news = p_real['title'].apply(len)
n_real = n_words_per_real_news.sum()

# count of fake news
n_words_per_fake_news = p_fake['title'].apply(len)
n_fake = n_words_per_fake_news.sum()

# N_Vocabulary
n_vocabulary = len(vocab)

# Laplace smoothing
alpha = 1

# Initiate parameters
parameters_real = {unique_word:0 for unique_word in vocab}
parameters_fake = {unique_word:0 for unique_word in vocab}

# Calculate parameters
for word in vocab:
   n_word_given_real = real_df[word].sum()
   p_word_given_real = (n_word_given_real + alpha) / (n_real + alpha*n_vocabulary)
   parameters_real[word] = p_word_given_real

   n_word_given_fake = fake_df[word].sum()
   p_word_given_fake = (n_word_given_fake + alpha) / (n_fake + alpha * n_vocabulary)
   parameters_fake[word] = p_word_given_fake


def classifier(message):
   message = re.sub('\W', ' ', message)
   message = message.lower().split()

   p_real_given_message = p_real
   p_fake_given_message = p_fake

   for word in message:
      if word in parameters_real:
         p_real_given_message *= parameters_real[word]

      if word in parameters_fake:
         p_fake_given_message *= parameters_fake[word]

   print('P(real|message):', p_real_given_message)
   print('P(fake|message):', p_fake_given_message)

   if p_real_given_message > p_fake_given_message:
      print('Label: real')
   elif p_real_given_message < p_fake_given_message:
      print('Label: fake')
   else:
      print('Equal proabilities, have a human classify this!')


# Example
classifier('SK건설, 친환경 부유식 해상풍력 발전사업 추진')
classifier("경기·인천·울산·세종 등 전국 대부분서 전세시장 공급부족 시달려")
