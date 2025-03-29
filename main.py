#%% LIB
from data import URLCrawl
import pandas as pd

uc = URLCrawl()

#%% CRAWL URLS
# url_list = [f"https://dantri.com.vn/tim-kiem/gi%C3%A1+v%C3%A0ng.htm?pi={index}"
#             for index in range(1, 31)]

# df = uc.get_url_collection(url_list)
# df.to_csv('database/url_collection.csv', encoding='utf-8-sig')

#%% CRAWL CONTENT

df = pd.read_csv("database/content.csv")
df['content'] = df['content'].map(
    lambda x: x.replace('Thông tin doanh nghiệp - sản phẩm', '')
        )

df = df.drop_duplicates()
df.info()
df.head()
