#%% LIB
import pandas as pd
import requests
from bs4 import BeautifulSoup


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
    


