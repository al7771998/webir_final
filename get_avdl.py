import json
import sys
file =  'id_article_' + sys.argv[1] + ".json"
f = open(file)
id_article = json.load(f)
f.close()
avdl = 0.0
count = 0
for i in id_article:
	for j in id_article[i]:
		avdl += len(id_article[i]['分詞後內文'])
		count += 1
print(avdl/count)