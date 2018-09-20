---
title: "[CRM] TripAdvisor Comments Analysis"
layout: post
date: 2017-07-26 15:10
tag:
- tripadvisor
- text mining
- emotion detection
- comment analysis

image: /assets/images/2017-07-26-TripAdvisor-comment-analysis/background.jpg
headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "댓글 crawling부터 감정 분석까지, 일련의 과정을 공유합니다."
category: project
author: angrypark
externalLink: false
---

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">

## 요약

2017년 1학기 연세대학교 고객관계경영 수업(손소영 교수님)을 들으며 진행하였던 프로젝트를 소개한다. 해외 사이트에서 댓글을 긁어오는 것부터 이를 통해 댓글의 감정을 분석하는 과정까지 정리해보았다.

**목차**

- [프로젝트 목표](#프로젝트-목표)
- [Comments Crawling](#comments-crawling)
- [Preprocessing](#preprocessing)
- [발표 자료](#발표-자료)

---
## 프로젝트 목표

#### TripAdvisor
TripAdvisor는 전세계에서 가장 많은 사람들이 사용하는 여행 정보 사이트 중 하나이다. 숙박 예약에서부터 각 주요 관광지에 대한 수많은 관광객들의 평가 정보까지 아주 자세히 나와있다.

![what-is-tripadvisor](/assets/images/2017-07-26-TripAdvisor-comment-analysis/what-is-tripadvisor.gif)

수업에서 진행한 프로젝트는 한국 관광지에 방문한 외국인 관광객들의 댓글들을 불러온 후, 그 중 무슬림 관광객들에 대한 한국 관광지의 선호도와 댓글을 분석하여 무슬림이 좋아하는 관광지를 추천했지만 이번 포스트에서는 그중에서도 내가 맡았던 **댓글의 감정 분석** 을 소개한다.

---
## Comments Crawling

#### 필요한 라이브러리 불러오기
~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
from bs4 import BeautifulSoup
import time
from tqdm import trange
import math
~~~

#### Crawling할 장소들 불러오기
~~~python
info_list = [
    # 1. 고궁
    ['Gyeongbokgung_Palace','https://www.tripadvisor.com/Attraction_Review-g294197-d324888-Reviews-Gyeongbokgung_Palace-Seoul.html',378],
    ['Changdeokgung_Palace','https://www.tripadvisor.com/Attraction_Review-g294197-d320359-Reviews-Changdeokgung_Palace-Seoul.html',183],
    ['Changgyeonggung Palace','https://www.tripadvisor.com/Attraction_Review-g294197-d1554603-Reviews-Changgyeonggung_Palace-Seoul.html',21],
    ['Deoksugung','https://www.tripadvisor.com/Attraction_Review-g294197-d324887-Reviews-Deoksugung-Seoul.html',33],
    # 2. 박물관(기념관)
    ['The War Memorial of Korea','https://www.tripadvisor.com/Attraction_Review-g294197-d554537-Reviews-The_War_Memorial_of_Korea-Seoul.html',225],
    ['National Museum of Korea','https://www.tripadvisor.com/Attraction_Review-g294197-d325043-Reviews-National_Museum_of_Korea-Seoul.html',94],
    ['Seoul Arts Center','https://www.tripadvisor.com/Attraction_Review-g294197-d1862210-Reviews-Seoul_Arts_Center-Seoul.html',9],
    # 3. 인사동
    ['Insadong','https://www.tripadvisor.com/Attraction_Review-g294197-d592506-Reviews-Insadong-Seoul.html',267],
    # 4. 남산타워
    ['N_Seoul_Tower','https://www.tripadvisor.com/Attraction_Review-g294197-d1169465-Reviews-N_Seoul_Tower-Seoul.html',362],
    # 5. 명동
    ['Myeongdong_Shopping_Street','https://www.tripadvisor.com/Attraction_Review-g294197-d553546-Reviews-Myeongdong_Shopping_Street-Seoul.html',360],
    # 6. 남대문시장
    ['Namdaemun Market','https://www.tripadvisor.com/Attraction_Review-g294197-d324907-Reviews-Namdaemun_Market-Seoul.html',81],
    # 7. 코엑스
    ['Starfield COEX Mall','https://www.tripadvisor.com/Attraction_Review-g294197-d554303-Reviews-Starfield_COEX_Mall-Seoul.html',23],
    # 8. 동대문시장
    ['Dongdaemun_Design_Plaza','https://www.tripadvisor.com/Attraction_Review-g294197-d6671988-Reviews-Dongdaemun_Design_Plaza-Seoul.html',57],
    # 9. 이태원
    ['Itaewon','https://www.tripadvisor.com/Attraction_Review-g294197-d2571660-Reviews-Itaewon-Seoul.html',22],
    # 10. 잠실(롯데월드)
    ['Lotte World','https://www.tripadvisor.com/Attraction_Review-g294197-d324891-Reviews-Lotte_World-Seoul.html',95],
    # 11. 여의도(63빌딩)
    ['63 City','https://www.tripadvisor.com/Attraction_Review-g294197-d554551-Reviews-63_City-Seoul.html',9],
    ['Yeouido Hangang Park','https://www.tripadvisor.com/Attraction_Review-g294197-d4798715-Reviews-Yeouido_Hangang_Park-Seoul.html',15],
    # 12. 한강/유람선
    ['Hangang Park','https://www.tripadvisor.com/Attraction_Review-g294197-d1519813-Reviews-Hangang_Park-Seoul.html',41],
    # 13. 청계천/광화문광장
    ['Gwanghwamun Gate','https://www.tripadvisor.com/Attraction_Review-g294197-d590748-Reviews-Gwanghwamun_Gate-Seoul.html',19],
    ['Gwanghwamun Square','https://www.tripadvisor.com/Attraction_Review-g294197-d6847900-Reviews-Gwanghwamun_Square-Seoul.html',23],
    # 14. 신촌/홍대주변
    ['Hongik_University_Street','https://www.tripadvisor.com/Attraction_Review-g294197-d1958940-Reviews-Hongik_University_Street-Seoul.html',76],
    ['Ewha Womans University','https://www.tripadvisor.com/Attraction_Review-g294197-d1862191-Reviews-Ewha_Womans_University-Seoul.html',34],
    # 15. DMC/월드컵 경기장
    ['Seoul World Cup Stadium','https://www.tripadvisor.com/Attraction_Review-g294197-d561808-Reviews-Seoul_World_Cup_Stadium-Seoul.html',6],
    # 16. 한옥마을(남산)
    ['Namsangol Hanok Village','https://www.tripadvisor.com/Attraction_Review-g294197-d1551271-Reviews-Namsangol_Hanok_Village-Seoul.html',33],
    # 17. 북촌/삼청동
    ['Bukchon Hanok Village','https://www.tripadvisor.com/Attraction_Review-g294197-d1379963-Reviews-Bukchon_Hanok_Village-Seoul.html',164],
    # 18. 청담동 / 압구정동
    ['Apgujeong Rodeo Street','https://www.tripadvisor.com/Attraction_Review-g294197-d1847008-Reviews-Apgujeong_Rodeo_Street-Seoul.html',5],
    # 19. 가로수길
    ['Garosu-gil','https://www.tripadvisor.com/Attraction_Review-g294197-d1604009-Reviews-Garosu_gil-Seoul.html',17],
    # 20. 강남역
    # ?
    ['Bongeunsa_Temple','https://www.tripadvisor.com/Attraction_Review-g294197-d592486-Reviews-Bongeunsa_Temple-Seoul.html',47],
    ['Gwangjang Market','https://www.tripadvisor.com/Attraction_Review-g294197-d1552278-Reviews-Gwangjang_Market-Seoul.html',57],    
]
~~~

#### 자동으로 Crawling하게 함수 정의!
~~~python
def get_data():
    global total
    for [name, url, pagenum] in info_list:
        print("%s Start..."%name)
        userid = [] # 평가자 아이디
        home = [] # 평가자 출신지
        rating = [] # 평점
        title = [] #
        comment = []
        for i in trange(pagenum):
            if i == 0 : tmp = requests.get(url)
            else : tmp = requests.get(url.replace('Reviews-','Reviews-{}-'.format('or%d'%(i*10))))
            soup = BeautifulSoup(tmp.text, 'html.parser')
            soup = BeautifulSoup(str(soup.findAll('div','review_collection')),'html.parser')
            for x in soup.findAll('div','location'):
                 home.append(x.contents[0][1:-1])
            for x in soup.findAll('div','rating reviewItemInline'):
                if len(x)==0 : print(name)
                rating.append(str([y for y in x.contents if len(str(y))>1][0]).split('"')[1].split('of')[0][:-1])
            for x in soup.findAll('span','noQuotes'):
                title += x.contents
            for x in soup.findAll('p','partial_entry'):
                comment.append(x.contents[0][1:-1])
            for x in soup.findAll('div','username mo'):
                userid.append(str(x.contents[1]).split('''user_name_name_click')">''')[1][:-7])
        df = pd.DataFrame([[name,a,b] for (a,b) in zip(home,rating)],columns = ['region','home','rating'])
        df['title'] = pd.DataFrame(title)
        df['comment'] = pd.DataFrame(comment)
        df['userid'] = pd.DataFrame(userid)
        total = pd.concat([total,df])
    total = total[['userid','region','home','rating','title','comment']]
~~~

#### 이를 csv로 저장
~~~python
get_data()
total.to_csv('review.csv',encoding = 'utf-8')
~~~

---
## Preprocessing
이 과정에서는 영어 단어를 불러올 때 과거형 / 과거분사형 / 복수형 등의 단어를 기본형으로 바꾸고, 쉼표와 마침표, 대문자를 제거하는 과정이다.

#### 필요한 라이브러리 불러오기
~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("Display.max_columns",200)
import cufflinks as cf
import plotly.plotly
plotly.tools.set_credentials_file(username='your_username', api_key='your_api_key')
import os
~~~


#### 대문자, 품사 처리
~~~python
import nltk
doc_en = pd.read_csv("TripAdvisor_reviews.csv")['comment'].sum()
for fname in os.listdir(path= 'Review_Texts'):
    for line in open('Review_Texts/%s'%fname):
        if line[:9] == '<Content>':
            doc_en += line[9:]
print("Text Downloaded")
from nltk import regexp_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
pattern = r"""(?x)                   # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |(?:[+/\-@&*])         # special characters with meanings
            """
from nltk.tokenize.regexp import RegexpTokenizer
tokeniser=RegexpTokenizer(pattern)
tokens_en = []
for line in doc_en.split('\n'):
    tmp = tokeniser.tokenize(line)
    tmp2 = []
    for word in tmp:
        word = word.lower()
        word = lmtzr.lemmatize(word,pos = 'n')
        word = lmtzr.lemmatize(word, pos = 'v')
        tmp2.append(word)
    tokens_en.append(tmp2)
~~~

최종 긁어온 데이터는 다음과 같은 형태를 띄고 있다.

![df](/assets/images/2017-07-26-TripAdvisor-comment-analysis/df.png)

---
## Emotion detection
자 이제 각각의 리뷰에 대해 기본적인 긍/부정 분석을 해보자. 영어 자연어 처리에 쓰이는 nltk에는 바로 사용할 수 있는 긍/부정 분석기가 있다.
#### 긍/부정 분석을 위한 라이브러리 불러오기
~~~python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
~~~

#### 전체 관광지 별로 해당 단어가 들어간 문장들의 긍/부정 분석
~~~python
def detect_positive(df, place_name, word):
    tmp = df[lambda x: x['region']==place_name]
    return np.mean([sid.polarity_scores(x)['compound'] for x in tmp['comment'].values if word in x.lower()])
~~~

#### 관광지별 긍/부정 정도 시각화

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~sungnampark/291.embed"></iframe>

---
## 발표 자료

<iframe src="//www.slideshare.net/slideshow/embed_code/key/siqmADvCQ6tOad" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/SungnamPark2/ss-78297303" title="데이터를 활용한 무슬림 맞춤형 서울 관광지도 제작" target="_blank">데이터를 활용한 무슬림 맞춤형 서울 관광지도 제작</a> </strong> from <strong><a target="_blank" href="https://www.slideshare.net/SungnamPark2">Sungnam Park</a></strong> </div>

---
