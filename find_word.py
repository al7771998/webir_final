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
from zhon.hanzi import punctuation
import string

### get arguments
file = "id_article_" + sys.argv[1] + ".json"
f = open(file)
dealed = json.load(f)
f.close()

word = sys.argv[2]


i = 0
for author in dealed:
	for index in dealed[author]:
		if word in dealed[author][index]['分詞後內文']:
			print(dealed[author][index]['分詞後內文'])