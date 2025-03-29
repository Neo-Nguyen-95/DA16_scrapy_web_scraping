import scrapy
import pandas as pd

class GoldtextSpider(scrapy.Spider):
    name = "goldtext"
    # allowed_domains = ["example.com"]
    df = pd.read_csv('/Users/dungnguyen/Desktop/Data Science off/Python Programming/3. Publication/DA16_scrapy_web_scraping/database/url_collection.csv')
    start_urls = df['urls'].to_list()

    def parse(self, response):
        content = " ".join(response.css("p").xpath("string(.)").getall())
        time = response.css("time::attr(datetime)").get()
        
        yield {
            "time": time,
            "content": content
            }
