import sys
import json
file = "ptt_" + sys.argv[1] + "_dealed.json"
f = open(file)
dealed = json.load(f)
f.close()
file =  'id2docsid' + sys.argv[1] + ".json"
f = open(file)
id2doc = json.load(f)
f.close()
file =  'id_article_' + sys.argv[1] + ".json"
f = open(file)
id_article = json.load(f)
f.close()
max_count = int(sys.argv[2])
file =  'id_reply_' + sys.argv[1] + ".json"
f = open(file)
id_reply = json.load(f)
f.close()
print(dealed[0])

while True:
	id_ = input('Press id: ')
	if id_ in id_reply:
		count = 0
		for doc_id in id_reply[id_]['推']:
			print(dealed[doc_id-1]['content'])
			count += 1
			print("Total 推, 噓： %d, %d"%(id_article[str(doc_id)]['推'],id_article[str(doc_id)]['噓']))
			print("======================================")
			if count >= max_count:
				break
