#%% LIB
import pandas as pd
import requests
from bs4 import BeautifulSoup

from nltk import ngrams
from collections import Counter


#%% CRAWLING
class URLCrawl:
    def __init__(self):
        pass

    def get_title_urls(self, url):
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(
            url=url, 
            headers=headers, 
            # verify=False
            )
        soup = BeautifulSoup(response.text, "html.parser")
        
        titles = []
        urls = []

        for a in soup.find_all("a", class_="dt-text-black-mine"):
            titles.append(a.get_text())
            urls.append(a['href'])
            
        df = pd.DataFrame({
            "titles": titles,
            "urls": urls
            })
        
        return df
    
    def get_url_collection(self, url_list):
        df = pd.DataFrame()
        
        for url in url_list:
            df = pd.concat([df, self.get_title_urls(url)], axis='rows')
            
        return df
    

#%% DATA ANALYSIS
class DataRepository:
    def __init__(self):
        pass
    
    def load_documents(self):
        df = pd.read_csv("database/content.csv")
        df['content'] = df['content'].map(
            lambda x: x.replace('Thông tin doanh nghiệp - sản phẩm', '')
                )

        df = df.drop_duplicates()
        
        return df
    
    def export_ngram_list(self, num_gram=2, num_top=20):
        text=" ".join(self.load_documents()['content'])
       

        result = {"word": [],
                  "frequency": []}

        tokens = text.lower().split()
        n_grams = list(ngrams(tokens, num_gram))
        counter_ngrams = Counter(n_grams)
        counter_ngrams_top = counter_ngrams.most_common(num_top)

        for i in range(num_top):
            word = " ".join(doc for doc in counter_ngrams_top[i][0])
            count = counter_ngrams_top[i][1]
            
            # Save in result
            result["word"].append(word)
            result["frequency"].append(count)
        
        df = pd.DataFrame(result)
        # df.to_excel(f'{num_top} most common {num_gram}-gram words.xlsx')
        return df
    
    
    
    
