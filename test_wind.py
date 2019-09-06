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
max_count = int(sys.argv[2])
file =  'id_reply_' + sys.argv[1] + ".json"
f = open(file)
id_reply = json.load(f)
f.close()
print(dealed[:10])
while True:
	id_ = input('Press id: ')
	count = 0
	for web in dealed:
		if id_ in web['author']:
			print(web['content'])
			print('======================================================')
			count+=1
			if count >= max_count:
				break
