# web_scrap.py
 
import concurrent.futures
import requests                # This is not standard library
import json
import sys
from bs4 import BeautifulSoup
HTML_PARSER = "html.parser"

board = sys.argv[1]
pages = sys.argv[2]
url = 'https://www.ptt.cc/bbs/' + board + '/index.html'
#url = 'https://www.ptt.cc/bbs/Gossiping/index.html' ##什麼版可以換
URLS = []
for _ in range(int(pages)):
  if 'Gossiping' in url:
    r = requests.Session()
    payload = {
      "from": "/bbs/Gossiping/index.html",
      "yes": "yes"
    }
    r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FGossiping%2Findex.html", payload)
    req = r.get(url)

  elif 'sex' in url:
    r = requests.Session()
    payload = {
      "from": "/bbs/sex/index.html",
      "yes": "yes"
    }
    r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FGossiping%2Findex.html", payload)
    req = r.get(url)

  else:
    req = requests.get(url)
  soup = BeautifulSoup(req.content, HTML_PARSER)
  #print(soup)
  titles = soup.find_all('div', class_='title')
#print(titles)
  for t in titles:
    tag = t.find('a')
    if tag is None:
      continue
    url_str = 'http://www.ptt.cc' + tag.get('href')
    print(url_str)
    URLS.append(url_str)
  page_button = soup.find('div', class_='btn-group btn-group-paging')
  last_page = page_button.find_all('a', class_='btn wide')[1]
#print(last_page)
  url = 'http://www.ptt.cc' + last_page['href']
#print(URLS)

'''with open ('URLSPttGossip3.json','w') as f:
  json.dump(URLS,f)'''

'''f = open('URLSPttGossip3.json', 'r')
URLS = json.load(f)
f.close()'''
 

def get_content(url):
  if 'Gossiping' in url:
    r = requests.Session()
    payload = {
        "from": "/bbs/Gossiping/index.html",
        "yes": "yes"
    }
    r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FGossiping%2Findex.html", payload)
    return r.get(url).text
  elif 'sex' in url:
    r = requests.Session()
    payload = {
        "from": "/bbs/sex/index.html",
        "yes": "yes"
    }
    r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FGossiping%2Findex.html", payload)
    return r.get(url).text
  else:
    return requests.get(url).text
 
def scrap():
    docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_url = {executor.submit(get_content, url): url for url in URLS}
        count = 0
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Execption as exc:
                print('%r generated an exception: %s' % (url, exc))
            else:
                #print('%r data is %s' % (url, data))
                soup = BeautifulSoup(data, "html.parser")
                links = soup.find_all("div", class_="bbs-screen bbs-content")
                for link in links:
                    doc = link.text
                    docs.append(doc)
                    count += 1
                    
        output = 'jsons/ptt_' + board + '_docs.json'
        with open(output, 'w') as outfile:
            json.dump(docs, outfile)
def main():
    for url in URLS:
        try:
            data = get_content(url)
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page length is %d' % (url, len(data)))
 
 
if __name__ == '__main__':
    scrap()
    print('finish!')