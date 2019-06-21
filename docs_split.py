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
from argparse import ArgumentParser
from collections import Counter
from zhon.hanzi import punctuation
import string
jieba.load_userdict('dict.txt.big')
file = "ptt_" + sys.argv[1] + "_dealed.json"
f = open(file)
dealed = json.load(f)
f.close()
#此處設每個文章的推噓數，以及文章及推文斷詞後list跟詞彙個數
id_article = {}
#EX : id : {內容: {推:17,噓:3,分詞後的內文:[...,...,...],回覆:{回覆內容們}}}
#此處算每個id推及噓的文章內容
id_reply = {}
#EX : id:{推的文章:[],噓的文章:[]}
wordCount_all = {}
wordCount_article = {}
i=0
for web in dealed:
	i+=1
	if i%20 == 0:
		print(i)
	#	break
	#print('\n\n\n\n')
	#print('======================================================')
	#print(web['title'])
	#print (web['time'])
	#print(web['author'])
	#web['content'] = ''.join(ch for ch in web['content'] if ch not in punctuation and ch not in string.punctuation)#.decode("utf-8")
	tmp = jieba.cut(web['content'],cut_all=False)
	tmp = '$'.join(tmp)
	tmp = tmp.split('$')#分詞後的文章內容
	article_split = []
	for ch in tmp:
		if ch not in punctuation and ch not in string.punctuation and ch != ' ':
			article_split.append(ch)
	#print(article_split)
	#print('====================')
	sig_pos,sig_neg = 0,0
	replies = []
	wordCount = {}
	for word in article_split:
		if word not in wordCount_all:
			wordCount_all.update({word:1})
		else:
			wordCount_all[word] += 1
		if word not in wordCount:
			wordCount.update({word:1})
		else:
			wordCount[word] += 1
	wordCount_article[i] = wordCount
	for ack in web['ack']:
		#判斷推噓and建立每個id的推噓文章
		if ack['signal'][0] == '推':
			if ack['author'] not in id_reply:
				id_reply.update({ack['author']:{'推':[i], '噓':[]}})
			else:
				if i not in id_reply[ack['author']]['噓']:
					id_reply[ack['author']]['推'].append(i)
			sig_pos += 1
		elif ack['signal'][0] == '噓':
			if ack['author'] not in id_reply:
				id_reply.update({ack['author']:{'推':[], '噓':[i]}})
			else:
				if i not in id_reply[ack['author']]['推']:
					id_reply[ack['author']]['噓'].append(i)
			sig_neg += 1

		#ack['content'] = ''.join(ch for ch in ack['content'] if ch not in punctuation)
		tmp = jieba.cut(ack['content'],cut_all=False)
		tmp = '$'.join(tmp)
		tmp = tmp.split('$')
		ack_split = []
		for ch in tmp:
			if ch not in punctuation and ch not in string.punctuation and ch != ' ':
				ack_split.append(ch)
		replies.append({ack['author']:ack_split})
		#print(  ack['signal'] +'  ,  ' + ack['author'] +'  ,  ' ) 
		#print('  ,  ' + ack['date'] +'  ,  ' + ack['time']  )
	if web['author'] not in id_article:
		id_article.update({web['author']:{i:{'推':sig_pos, '噓':sig_neg, \
		'分詞後內文':article_split, '回覆':replies}}})
	else:
		id_article[web['author']].update({i :{'推':sig_pos, '噓':sig_neg, \
		'分詞後內文':article_split, '回覆':replies}})
#print(id_article)
#print('------------------------------------------------')
#print(id_reply)
#print('================================================')
#print(wordCount_article)
#print('++++++++++++++++++++++++++++++++++++++++++++++++')
#print(wordCount_all)
with open('id_article_'+sys.argv[1]+'.json','w') as fw:
	json.dump(id_article,fw)
with open('id_reply_'+sys.argv[1]+'.json','w') as fw:
	json.dump(id_reply,fw)
with open('wordCountArticle_'+sys.argv[1]+'.json','w') as fw:
	json.dump(wordCount_article,fw)
with open('wordCountAll_'+sys.argv[1]+'.json','w') as fw:
	json.dump(wordCount_all,fw)
