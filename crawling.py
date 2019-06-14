# web_scrap.py
 
import concurrent.futures
import requests                # This is not standard library
import json
from bs4 import BeautifulSoup
f = open('urlsPttGossip2.json', 'r')
URLS = json.load(f)
f.close()
 

def get_content(url):
    r = requests.Session()
    payload = {
        "from": "/bbs/Gossiping/index.html",
        "yes": "yes"
    }
    r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FGossiping%2Findex.html", payload)
    return r.get(url).text
 
 
def scrap():
    docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_url = {executor.submit(get_content, url): url for url in URLS}
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
        with open('ptt_gossip_docs.json', 'w') as outfile:
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