#%% LIB
import pandas as pd
import requests
from bs4 import BeautifulSoup

from nltk import ngrams
from collections import Counter
import tiktoken
encoding = tiktoken.encoding_for_model("text-embedding-3-small")

from dotenv import load_dotenv
import os
load_dotenv()

from openai import OpenAI

from pyvi import ViTokenizer
from gensim import corpora
from gensim.models import LdaModel
import re
import string

import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def load_setiment_docs(self):
        df = pd.read_csv("database/content_wt_sentiment.csv")
        df.drop(columns='Unnamed: 0', inplace=True)
        
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
    
    def count_tokens(self, doc):
        return len(encoding.encode(doc))
    
    def remove_long_docs(self):
        max_tokens = 8192
        
        df = self.load_documents()
        
        df['token_count'] = df['content'].apply(lambda x: self.count_tokens(x))
        
        df_filtered = df[df['token_count'] < max_tokens]
              
        print(f"Remove {len(df)-len(df_filtered)} overly long docs!")
        
        return df_filtered
        
    
    def classify_gpt(self, csv_file_name):
        api_key = os.getenv("SECRETE_KEY")
        
        def classify_text(text):
            labels = [
                "Yếu tố kinh tế vĩ mô (bao gồm: lạm phát, lãi xuất, tăng trưởng, chính sách tiền tệ/tài khoá)",
                "Yếu tố tiền tệ và tài chính (bao gồm: USD index, tỷ giá ngoại tệ, cung tiền, nợ công, lợi suất trái phiếu)",
                "Yếu tố địa chính trị và xã hội (bao gồm: khủng hoảng địa chính trị, chiến tranh, thiên tai, đại dịch)",
                "Yếu tố thị trường tài chính (bao gồm: biến động thị trường chứng khoán, đầu cơ, khủng hoảng tài chính)",
                "Yếu tố khác"
                ]
            
            messages = [
                {"role": "system", 
                 "content": "Bạn là một mô hình phân loại văn bản viết về vàng."},
                {"role": "user", 
                 "content": f"""Cho văn bản sau: \" {text} \"
                Danh sách nhãn: {', '.join(labels)}
                Chỉ trả về một nhãn duy nhất nêu lên tác động chính tới giá vàng. 
                Không thêm kí tự nào.
                """}
                ]
                 
            client = OpenAI(api_key = api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
                )
            
            return response.choices[0].message.content
        
        def classify_documents(documents):
            values = []
            
            for text in documents:
                topic = classify_text(text)
                values.append(topic)
            
            return values
        
        df = self.remove_long_docs()

        df['sentiment'] = classify_documents(df['content'])
        
        df.to_csv(csv_file_name)
        
        return print('Successful classification!')
    
    def grouping_lda(self):

        ### Stop word
        with open("database/vietnamese-stopwords.txt", "r") as file:
            vn_stopwords = file.read().splitlines()
        
        vn_stopwords = [word.replace(' ', '_') for word in vn_stopwords]
                
        ### Tokenize   
        tokens = [  
            [word.lower() for word in ViTokenizer.tokenize(doc).split()
             if word.lower() not in vn_stopwords
             ] for doc in self.load_documents()['content']
            ]
        
        def keep_meaningful_word(word):
            for num_str in '1234567890':
                if num_str in word:
                    return False
            
            if re.fullmatch(r'\W+', word):
                return False
            
            if all(char in string.punctuation for char in word):
                return False
            
            return True
        
        tokens_updated = [
            [w for w in token if keep_meaningful_word(w)]
            for token in tokens
            ]
        
        all_words = []
        for token in tokens_updated:
            all_words.extend(token)
        
        counter_word = Counter(all_words)
        top_common_word = [
            word[0] for word in counter_word.most_common(int(0.05*len(counter_word)))
            ]
        top_10k_word = [
            word[0] for word in counter_word.most_common(
                int(0.05*len(counter_word)) + 10000
                )
            ]
        
        (pd.Series(top_common_word[:100])
         .rename('top_100_most_common')
         .to_excel('top_common_words.xlsx')
         )
        
        tokens_updated = [
            [word for word in token 
             if word not in top_common_word and word in top_10k_word] 
            for token in tokens_updated
            ]
        
        tokens_updated = [token for token in tokens_updated if len(token)>=1]
        
        
        dictionary = corpora.Dictionary(tokens_updated)
        
        corpus = [dictionary.doc2bow(text) for text in tokens_updated]
        
        num_topics = 15
        
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=30,
            random_state=42
            )
        
        topics = lda_model.print_topics(num_topics=num_topics, num_words=20)
        
        topics_dict = {"top_weighted_words": []}
        
        for topic in topics:
            topics_dict["top_weighted_words"].append(
                ", ".join(topic[1].split('"')[1::2])
                )
            
        df_topics = pd.DataFrame(topics_dict)
        
        doc_topics = [lda_model.get_document_topics(bow) for bow in corpus]
        
        topic_interest = dict(zip(range(num_topics), [0]*num_topics))
        
        for doct in doc_topics:
            for i, prob in doct:
                topic_interest[i] += prob
        
        topic_interest = pd.DataFrame(
            {"prob": topic_interest.values()},
            index=topic_interest.keys()
            )
        
        topic_interest = topic_interest/ len(self.load_documents())
        
        result = topic_interest.join(df_topics)
        
        result.to_csv('database/topic_allocation.csv', index=False)
        
        return print("Succesfully allocating words!")
    
    
    
        
    
    
    
    
    
    
