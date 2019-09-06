# -*- coding: UTF-8 -*-
import json
import jieba
import pandas as pd
import numpy as np
import random
import csv
import operator
from argparse import ArgumentParser
from collections import Counter
import sys

file = "id_article_" + sys.argv[1] + ".json"
f = open(file)
id_article = json.load(f)
f.close()
list_key = list(id_article.keys())
print(len(list_key))
for i in list_key[40:60]:
	for j in id_article[i].keys():
		print(id_article[i][j]['分詞後內文'])
