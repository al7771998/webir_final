# -*- coding: UTF-8 -*-
import json
import jieba
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
f = open('ptt_BG_dealed.json')
BG_dealed = json.load(f)
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
for web in BG_dealed:
	i+=1
	if i>500:
		break
	#print('\n\n\n\n')
	#print('======================================================')
	#print(web['title'])
	#print (web['time'])
	#print(web['author'])
	web['content'] = ''.join(ch for ch in web['content'] if ch not in punctuation and ch not in string.punctuation)#.decode("utf-8")
	tmp = jieba.cut(web['content'],cut_all=False)
	tmp = '$'.join(tmp)
	article_split = tmp.split('$')#分詞後的文章內容
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
		if word not in wordCount_article:
			wordCount.update({word:1})
		else:
			wordCount[word] += 1
	wordCount_article[web['context']] = wordCount
	for ack in web['ack']:
		#判斷推噓and建立每個id的推噓文章
		if ack['signal'][0] == '推':
			if ack['author'] not in id_reply:
				id_reply.update({ack['author']:{'推':[article_split], '噓':[]}})
			else:
				if article_split not in id_reply[ack['author']]['噓']:
					id_reply[ack['author']]['推'].append(article_split)
			sig_pos += 1
		elif ack['signal'][0] == '噓':
			if ack['author'] not in id_reply:
				id_reply.update({ack['author']:{'推':[], '噓':[article_split]}})
			else:
				if article_split not in id_reply[ack['author']]['推']:
					id_reply[ack['author']]['噓'].append(article_split)
			sig_neg += 1

		ack['content'] = ''.join(ch for ch in ack['content'] if ch not in punctuation)
		tmp = jieba.cut(ack['content'],cut_all=False)
		tmp = '$'.join(tmp)
		ack_split = tmp.split('$')
		replies.append(ack_split)
		#print(  ack['signal'] +'  ,  ' + ack['author'] +'  ,  ' ) 
		#print('  ,  ' + ack['date'] +'  ,  ' + ack['time']  )
	if web['author'] not in id_article:
		id_article.update({web['author']:{web['content']:{'推':sig_pos, '噓':sig_neg, \
		'分詞後內文':article_split, '回覆':replies}}})
	else:
		id_article[web['author']].update({web['content']:{'推':sig_pos, '噓':sig_neg, \
		'分詞後內文':article_split, '回覆':replies}})
print(id_article)
print('------------------------------------------------')
print(id_reply)
print('================================================')
print(wordCount_article)
print('++++++++++++++++++++++++++++++++++++++++++++++++')
print(wordCount_all)







