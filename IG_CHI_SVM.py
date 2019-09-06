# -*- coding: UTF-8 -*-
import json
import jieba
import sys
import pandas as pd
import numpy as np
import random
import csv
import re
import operator
import sys
from argparse import ArgumentParser
from collections import Counter
import string
import math
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

#read id_article
file = "id_article_" + 'hatepoli' + ".json"
f = open(file)
dealed = json.load(f)
f.close()
#print(dealed)

def chi_square(A,B,C,D):
  return float(A + B + C + D) * (A*D-C*B)*(A*D-C*B) / ((A+C)*(B+D)*(A+B)*(C+D))

#read df_list
file = "df_" + "hatepoli" + ".json"
f = open(file)
df_dict = json.load(f)
f.close()

#read wordCountDocs
file = "wordCountArticle_" + 'hatepoli' + ".json"
f = open(file)
wordCountDocs = json.load(f)
f.close()

docs = {}
for i in dealed:
  if dealed[i]['推'] - 5 * dealed[i]['噓'] > 0 :
    docs[i] = 1
  elif dealed[i]['推'] - dealed[i]['噓'] < 0:
    docs[i] = 2
  
  

stopWords=[]
with open('stopWord.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)
docs_num = len(docs)
print(docs_num)
count_classes = np.array([len([i for i in docs if docs[i] == 1]),len([i for i in docs if docs[i] == 2])])
P_c = np.array([float(len([i for i in docs if docs[i] == 1]))/ docs_num, float(len([i for i in docs if docs[i] == 2]))/docs_num])
print(P_c)
word2id = {}
count = 0

for word in df_dict:
  if word == '..' or word == '...' or word == '....' or word == '.....' or word == 'www' or word == '◆'\
  or word == 'XD' or word == '\n' or word == 'https' or word == 'From' or word == 'from' or word == 'com' \
  or word == '........' or df_dict[word] < 600 or df_dict[word] > 30000 or word in stopWords:
    continue
  word2id[word] = count
  count += 1
count_terms_classes = np.zeros((2,len(word2id)))
P_not_terms_classes = np.zeros((2,len(word2id)))
P_terms_classes = np.zeros((2,len(word2id)))
count_terms = np.zeros(len(word2id))
for word in word2id:
  count_terms[word2id[word]] = df_dict[word]
for i in dealed:
  is_used = {}
  for word in dealed[i]['分詞後內文']:
    if word not in is_used:
      if word not in word2id:
        continue
      if i in docs:
        count_terms_classes[docs[i]-1][word2id[word]] += 1
        is_used[word] = 1
P_terms = count_terms / docs_num
print(P_terms[:60])
P_terms_not = 1 - P_terms
for i in range(0,2):
  P_terms_classes[i] = count_terms_classes[i]/count_terms
  P_not_terms_classes[i] = (P_c[i]*docs_num - count_terms_classes[i]) / (docs_num - count_terms)

#做Information Gain
'''ig_dict = {}
count = 0
#df_dict[word] >= 20000 or df_dict[word] < 1000 or
for word in df_dict:
  if word != '韓國瑜' and word != '陳水扁':
    continue
  ig_word = IG(word)
  if count % 50 == 0:
    print(ig_word)
  ig_dict[word] = ig_word
  count += 1
count = 0
for key,value in ig_dict.items():
  print([key,value])
  if count > 20:
    break
  count += 1
items= ig_dict.items() 
backitems=[[v[1],v[0]] for v in items] 
backitems.sort(reverse=True) 
ig_key_list = [ backitems[i][1] for i in range(0,len(backitems))]
print(len(ig_key_list))
print(ig_key_list[:50])'''

#做IG
word_ig_information = []
e_0 = 0.0
for c_index in range(0, 2):
    e_0+=P_c[c_index]*np.log2(P_c[c_index])
e_0 = -e_0
for w in word2id:
    e_1 = 0.0
    for c_index in range(0, 2):
        tmp1 = P_terms_classes[c_index][word2id[w]]
        if tmp1 !=0:
            e_1 += P_terms[word2id[w]]*tmp1*np.log2(tmp1)
        tmp2 = P_not_terms_classes[c_index][word2id[w]]
        if tmp2 !=0:
            e_1 += (P_terms_not[word2id[w]])*(tmp2*np.log2(tmp2))
    e_1 = -e_1
    
    information_gain = e_0 - e_1
    
   
    
    word_ig_information.append([information_gain, w])
    
word_ig_information = sorted(word_ig_information, key=lambda x: x[0], reverse=True)
print(word_ig_information[:100])

'''preview = pd.DataFrame(word_ig_information)
preview.columns=['information_gain', 'word']
preview.head(10)'''
'''word_mi_information= []
for w in word2id:
    mi_list = []
    for c_index in range(0, 2):
        N = docs_num
        N_1_1=count_terms_classes[c_index][word2id[w]]
        N_1_0=count_terms[word2id[w]]-N_1_1
        N_0_1= (count_classes[c_index]) - N_1_1
        N_0_0= (N - count_classes[c_index]) - N_1_0
        mi = 0
        if (N*N_1_1)!=0:
            mi += (N_1_1/N)*np.log2((N*N_1_1)/((N_1_1+N_1_0)*(N_0_1+N_1_1)))
        if (N*N_0_1)!=0:
            mi += (N_0_1/N)*np.log2((N*N_0_1)/((N_0_1+N_0_0)*(N_0_1+N_1_1)))
        if (N*N_1_0)!=0:
            mi += (N_1_0/N)*np.log2((N*N_1_0)/((N_1_1+N_1_0)*(N_0_0+N_1_0)))
        if (N*N_0_0)!=0:
            mi += (N_0_0/N)*np.log2((N*N_0_0)/((N_0_1+N_0_0)*(N_0_0+N_1_0)))
        mi_list.append(mi)
    
    mi_list = np.asarray(mi_list)
    average = np.sum(mi_list * P_c)
    max_mi = np.max(mi_list)
    max_index = np.argmax(mi_list)
    word_mi_information.append([average, max_mi, max_index, w])
    

word_mi_information = sorted(word_mi_information, key=lambda x: x[0], reverse=True)
#print(word_ig_information[:100])

preview = pd.DataFrame(word_mi_information)
preview.columns=['mutual information(MI)', 'main class MI', 'main_class', 'word']
print(preview.head(120))'''
'''def H(p):
  return p * math.log(p+1e-10)

def IG(term):
  P_t = float(df_dict[term]) / docs_num
  P_nt = 1 - P_t
  print(P_c,P_nc,P_t,P_nt)
  P_t_c,P_nt_c,P_t_nc,P_nt_nc = 0.0,0.0,0.0,0.0
  for i in dealed:
    if term in wordCountDocs[i] and docs[i] == 1:
      P_t_c += 1
    elif term not in wordCountDocs[i] and docs[i] == 1:
      P_nt_c += 1
    elif term in wordCountDocs[i] and docs[i] == -1:
      P_t_nc += 1
    elif term not in wordCountDocs[i] and docs[i] == -1:
      P_nt_nc += 1
  P_t_c /= len(docs)
  P_nt_c /= len(docs)
  P_t_nc /= len(docs)
  P_nt_nc /= len(docs)
  print(P_t_c,P_nt_c,P_t_nc,P_nt_nc)
  P_t_c /= P_t
  P_nt_c /= P_nt
  P_t_nc /= P_t
  P_nt_nc /= P_nt
  print('---------------------------------------------------------------')
  return -H(P_c)-H(P_nc)+P_t*(H(P_t_c)+H(P_t_nc))+P_nt*(H(P_nt_c)+H(P_nt_nc))'''
#做chi_square
word_chi_information= []
for w in word2id:
    chi_list = []
    for c_index in range(0, 2):
        N = docs_num
        N_1_1=count_terms_classes[c_index][word2id[w]]
        N_1_0=count_terms[word2id[w]]-N_1_1
        N_0_1= count_classes[c_index] - N_1_1
        N_0_0= (N - count_classes[c_index]) - N_1_0
        chi = 0.0
        chi += N
        
        chi /= (N_1_1+N_0_1)
        
        tmp1 =(N_1_1*N_0_0)-(N_0_1*N_1_0)
        chi *= tmp1
        chi /=(N_1_0+N_0_0)
        chi /=(N_1_1+N_1_0)
        
        chi *= tmp1
        chi /= (N_0_0+N_1_0)

        chi_list.append(chi)

    chi_list = np.asarray(chi_list)
    average = np.sum(chi_list * P_c)
    max_chi = np.max(chi_list)
    max_index = np.argmax(chi_list)
    word_chi_information.append([average, max_chi, max_index, w])


word_chi_information = sorted(word_chi_information, key=lambda x: x[0], reverse=True)
print(word_chi_information[:100])
'''chi_dict = {}
for word in df_dict:
  T_C,T_NC,NT_C,NT_NC = 0,0,0,0
  if df_dict[word] >= 20000 or df_dict[word] < 1000:
    continue
  for query in ["支持","理想","當選","舔共","廢物","討厭"]:
    for i in dealed:
      if word in wordCountDocs[i] and query in wordCountDocs[i] and docs[i] == 1:
        T_C += 1
      elif word in wordCountDocs[i] and query in wordCountDocs[i] and docs[i] == -1:
        T_NC += 1
      elif word not in wordCountDocs[i] and query in wordCountDocs[i] and docs[i] == 1:
        NT_C += 1
      elif word not in wordCountDocs[i] and query in wordCountDocs[i] and docs[i] == -1:
        NT_NC += 1
      elif word in wordCountDocs[i] and query not in wordCountDocs[i] and docs[i] == 1:
        NT_C += 1
      elif word in wordCountDocs[i] and query not in wordCountDocs[i] and docs[i] == -1:
        NT_NC += 1
      elif word not in wordCountDocs[i] and query not in wordCountDocs[i] and docs[i] == 1:
        NT_C += 1
      elif word not in wordCountDocs[i] and query not in wordCountDocs[i] and docs[i] == -1:
        NT_NC += 1  '''
      
    
'''chi_T_C = chi_square(T_C,T_NC,NT_C,NT_NC)
  chi_dict[word+','+query] = chi_T_C
count = 0
for key,value in chi_dict.items():
  print([key,value])
  if count > 20:
    break
  count += 1
with open('chi_dict2.json','w') as fw:
  json.dump(chi_dict,fw)'''
'''f = open('chi_dict.json')
chi_dict = json.load(f)
f.close()
items= chi_dict.items() 
backitems=[[v[1],v[0]] for v in items] 
backitems.sort(reverse=True) 
chi_key_list = [ backitems[i][1] for i in range(0,len(backitems))]
print(len(chi_key_list))
print(chi_key_list[:50])'''



'''for index in dealed:
  #if word1 in dealed[index]['分詞後內文']:
  reply_num = 0
  for replier in dealed[index]['回覆']:
    reply_num += 1
    if list(replier.keys())[0] not in id_reply_num:
      id_reply_num[list(replier.keys())[0]] = 1
    else:
      id_reply_num[list(replier.keys())[0]] += 1
  if reply_num > avg_reply / 3:
    if dealed[index]['推'] > 3 * dealed[index]['噓']:
      pos_docs.append(dealed[index])
    else:
      neg_docs.append(dealed[index])'''
##USE IG's top 100 as features
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
kf = KFold(n_splits=2, shuffle=True)

features = []
for key in word_ig_information[:300]:
  features.append(key[1])
print(features)
feature_size = len(features)
X_count = np.zeros((docs_num, feature_size), dtype=float)
docs_list = list(docs.keys())
Y = np.zeros(docs_num)
for i in range(0,docs_num):
    for j in range(0, feature_size):
        tmp_word = features[j]
        if tmp_word in wordCountDocs[docs_list[i]]:
            X_count[i][j]= wordCountDocs[docs_list[i]][tmp_word]
        else:
            X_count[i][j]= 0
    Y[int(i)] = docs[docs_list[i]] - 1
k,b,doc_len = 2.0,0.9,236.83484837831594
X = X_count * (k + 1) / (X_count + k * (1 - b + b * doc_len))
#Do SVM
from sklearn import svm
#clf = svm.LinearSVC(tol = 1e-6)
clf= svm.SVC(tol = 1e-6,shrinking=False,max_iter=5000,verbose = 1)
shuffled_index = np.arange(0, docs_num)
np.random.shuffle(shuffled_index)
confusion_matrix_sum = np.zeros((2, 2), dtype=float)
for train_index, test_index in kf.split(X):
    
    train_index_shuffled = np.take(shuffled_index, train_index)
    test_index_shuffled = np.take(shuffled_index, test_index)
    X_train, X_test = X[train_index_shuffled], X[test_index_shuffled]
    y_train, y_test = Y[train_index_shuffled], Y[test_index_shuffled]
    
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(pred.tolist())
    confusion_matrix_sum += confusion_matrix(y_test, pred)

confusion_matrix_avg = confusion_matrix_sum / k

tmp = 0
for i in range(0, 2):
    tmp += confusion_matrix_avg[i][i]
accuracy = (tmp*k)/ docs_num
print("accuracy: ",accuracy)
print("confusion matrix:")
print(confusion_matrix_avg)


