#%% LIB
from data import URLCrawl, DataRepository
import pandas as pd

uc = URLCrawl()
repo = DataRepository()

#%% CRAWL URLS
# url_list = [f"https://dantri.com.vn/tim-kiem/gi%C3%A1+v%C3%A0ng.htm?pi={index}"
#             for index in range(1, 31)]

# df = uc.get_url_collection(url_list)
# df.to_csv('database/url_collection.csv', encoding='utf-8-sig')

#%% CRAWL CONTENT
df = repo.load_documents()
df.head()

repo.export_ngram_list(num_gram=2, num_top=20)
