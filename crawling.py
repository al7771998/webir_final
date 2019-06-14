#coding=utf-8
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import json 

HTML_PARSER = "html.parser"

if __name__ == '__main__':
  url = 'http://www.ptt.cc/bbs/Gossiping/index.html'
  # over 18 request
  r = requests.Session()
  payload = {
    "from": "/bbs/Gossiping/index.html",
    "yes": "yes"
  }
  r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FGossiping%2Findex.html", payload)
  urls = []
  for i in range(500):
    req = r.get(url)
    soup = BeautifulSoup(req.content, HTML_PARSER)
    #print(soup)
    titles = soup.find_all('div', class_='title')
    #print(titles)
    for t in titles:
      tag = t.find('a')
      if tag is None:
        continue
      url_str = 'http://www.ptt.cc' + tag.get('href')
      urls.append(url_str)
    page_button = soup.find('div', class_='btn-group btn-group-paging')
    last_page = page_button.find_all('a', class_='btn wide')[1]
    #print(last_page)
    url = 'http://www.ptt.cc' + last_page['href']
  print(urls)

  #存連結用的
  """with open ('urlsPttGossip2.json','w') as f:
           json.dump(urls,f)"""

  '''f = open('urlsPttGossip2.json', 'r')
        urls = json.load(f)
  f.close()'''
  #count = 0
  docs = []
  r = requests.Session()
  payload = {
    "from": "/bbs/Gossiping/index.html",
    "yes": "yes"
  }
  r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FGossiping%2Findex.html", payload)
  for url_str in urls: 
      page = 1 
      print(count)
      res = r.get(url_str)
      soup = BeautifulSoup(res.text, "html.parser")
      #print(soup)
      links = soup.find_all("div", class_="bbs-screen bbs-content")
      #print(links)
      '''if count <= 20:
          print(links)
          print(url_str)
          for link in links:
              print(link)
              print('--------------------')
              link.text是網站內文
              print(link.text)

              print('======================================================')'''

      for link in links:
          doc = link.text
          if '--' in doc:
            index = doc.index('--')
            #print(index)
            docs.append(doc[:index])
          else:
            docs.append(doc)
      #print('===============================================')
      #count += 1
  #print(docs[107:127])
  
  #存爬到的文章用的
  """with open('ptt_docs8.json', 'w') as outfile:
            json.dump(docs, outfile)"""