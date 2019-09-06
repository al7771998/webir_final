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
'''from gensim.models import Word2Vec,Doc2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors'''


sentences = []
file =  "id_article_" + 'hatepoli' + ".json"
f = open(file)
dealed = json.load(f)
f.close()
docs = {}
for i in dealed:
  if dealed[i]['推'] - 3 * dealed[i]['噓'] > 0 and dealed[i]['推'] + dealed[i]['噓'] >= 15:
    docs[i] = 1
  if dealed[i]['推'] - 5 * dealed[i]['噓'] > 0 and dealed[i]['推'] + dealed[i]['噓'] < 15:
    docs[i] = 1
  elif dealed[i]['推'] - 2 * dealed[i]['噓'] < 0 and dealed[i]['推'] + dealed[i]['噓'] >= 15:
    docs[i] = 2
  elif dealed[i]['推'] - dealed[i]['噓'] < 0 and dealed[i]['推'] + dealed[i]['噓'] < 15:
    docs[i] = 2
docs_num = len(docs)
count_classes = np.array([len([i for i in docs if docs[i] == 1]),len([i for i in docs if docs[i] == 2])])
P_c = np.array([float(len([i for i in docs if docs[i] == 1]))/ docs_num, float(len([i for i in docs if docs[i] == 2]))/docs_num])
print(P_c)
count = 0
for i in dealed:
  tmp = ''
  for word in dealed[i]['分詞後內文']:
    tmp += word
    tmp += ' '
  sentences.append(gensim.models.doc2vec.TaggedDocument(tmp, [str(count)]))
  count += 1
model = gensim.models.Doc2Vec(sentences,size=250, window=5)
model.train(sentences)
model.save('doc2vec_ir.model')

model_load = Doc2Vec.load('doc2vec_ir.model')
print(model_load == model)

print(model.docvecs.most_similar(0))

X = np.zeros((len(docs),250))
Y = np.zeros(len(docs))
count = 0
for i in docs:
  if docs[i] == 1:
    X[count] = model.docvecs[int(i)-1]
    Y[count] = 1
    count += 1
  else:
    X[count] = model.docvecs[int(i)-1]
    Y[count] = -1
    count += 1

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
k=2
kf = KFold(n_splits=k, shuffle=True)
from sklearn import svm
clf = svm.LinearSVC(tol = 1e-6)
#clf= svm.SVC(tol = 1e-6,shrinking=False,max_iter=5000,verbose = 1)
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