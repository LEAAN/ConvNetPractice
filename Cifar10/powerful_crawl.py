from icrawler.examples import GoogleImageCrawler
import os


path = '/home/workstation/Documents/humandataset/beforethermal/'

keywords_0 = ['empty street','empty office', 'empty city road', 'trees street empty', 'street light at night']
keywords_1 = ['people on the street', 'pedestrians', 'pedestrian',  'traffic warden', 'people on road']
keywords = keywords_1 + keywords_0

for word in keywords:
    word = 'empty city road'
    dirpath = path + word
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    google_crawler = GoogleImageCrawler(dirpath)
    google_crawler.crawl(keyword=word, offset=0, max_num=1000,
                         date_min=None, date_max=None, feeder_thr_num=1,
                         parser_thr_num=1, downloader_thr_num=4,
                         min_size=None, max_size= None)

