#%% LIB
from data import URLCrawl, DataRepository
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
pd.set_option('display.max_columns', None)

uc = URLCrawl()
repo = DataRepository()

# df = repo.load_documents()
# df.head()

#%% CRAWL
# url_list = [f"https://dantri.com.vn/tim-kiem/gi%C3%A1+v%C3%A0ng.htm?pi={index}"
#             for index in range(1, 31)]

# df = uc.get_url_collection(url_list)
# df.to_csv('database/url_collection.csv', encoding='utf-8-sig')

#%% SENTIMENT TAGGING
# repo.classify_gpt('database/content_wt_sentiment.csv')

df = repo.load_setiment_docs()
df.head()

#%% EDA
    #%% 1. What are the most common words in the content?
repo.export_ngram_list(num_gram=10, num_top=20)

    #%% 2. How frequent do they talk about the topic? Guess peak at new year
df['month'] = pd.to_datetime(df['time']).dt.to_period("M")
df['month'] = df['month'].dt.to_timestamp()
df_frequency = df.groupby('month').size().rename('count').reset_index()
df_frequency['month_str'] = df_frequency['month'].dt.strftime('%Y-%m')

plt.figure(figsize=(9, 6))
sns.barplot(
    data=df_frequency,
    # data=df_frequency[df_frequency['month']>pd.Timestamp(2024,1,1)], 
    x='month_str', 
    y='count'
    )
x_ticks = df_frequency['month_str'].to_list()
plt.xticks(ticks=range(0, len(x_ticks), 6), 
           label=x_ticks[::6],
           rotation=45
           )
plt.xlabel(xlabel='Thời điểm đăng bài [Năm-tháng]')
plt.ylabel('Số lượng bài viết')
plt.title('Thống kê số lượng bài viết theo thời gian')
plt.show()

# Which month has the most article?
# df_frequency['month_num'] = pd.to_datetime(df_frequency['month']).dt.month
# sns.boxplot(
#     data=df_frequency[df_frequency['month']< pd.Timestamp(2024,1,1)], 
#     y='month_num', x='count', orient='y')
# plt.ylabel('Tháng')
# plt.xlabel('Phân bố số bài viết')
# plt.show()

    #%% 3. Sentiment analysis
df[df['sentiment']=='Yếu tố kinh tế vĩ mô']['content'].iloc[0]
df[df['sentiment']=='Yếu tố tiền tệ và tài chính']['content'].iloc[1]
df[df['sentiment']=='Yếu tố địa chính trị và xã hội']['content'].iloc[0]
df[df['sentiment']=='Yếu tố thị trường tài chính']['content'].iloc[1]

ax = sns.barplot(
    df['sentiment'].value_counts(), 
    orient='y'
    )
ax.bar_label(ax.containers[0])
plt.ylabel('Nguyên nhân chính ảnh hưởng tới giá vàng')
plt.xlabel('Số bài viết')
plt.xlim(0, 250)
plt.show()



    #%% 4. Latent Dirichlet Allocation

repo.grouping_lda()

df_lda = pd.read_csv('database/topic_allocation.csv')

df_lda['topic_words'] = df_lda['top_weighted_words'].apply(
    lambda x: ", ".join(x.split(',')[:4])
    )

df_lda = df_lda.sort_values('prob', ascending=False)

df_lda = df_lda.reset_index()

df_lda['topic_words'] = df_lda.index.astype(str) + '. ' + df_lda['topic_words']
 
plt.figure(figsize=(6, 8), dpi=200)
sns.barplot(
    x=df_lda['prob']*100,
    y=df_lda['topic_words'],
    orient='y'
    )
plt.xlabel('Tỉ lệ xuất hiện của mỗi chủ đề [%]')
plt.ylabel('Bốn từ có tỉ trọng lớn nhất mỗi chủ đề')
plt.show()



