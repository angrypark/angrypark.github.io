---
title: "[Visualization] Markdown에 plotly 그래프 넣기"
layout: post
date: 2017-07-27 14:01
image: /assets/images/170720/background.jpeg
headerImage: false
tag:
- plotly
- blog
- interactive graph
category: blog
author: angrypark
description: Markdown에 plotly 그래프 넣는 방법

---

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">
## 요약

이번 포스트에서는 블로그나 기타 markdown을 작성할 때 쉽게 interactive한 graph를 넣는 방법을 소개한다.

- [Plotly?](#plotly)
- [예시](#예시)
- [결과](#결과)

---
## Plotly?
plotly는 간단한 막대그래프에서부터 3차원의 복잡한 그래프까지 누구나 쉽게 예쁜 차트를 만들 수 있는 시각화 오픈 소스이다. 데이터 분석에서 가장 많이 쓰이는 언어인 R과 Python 뿐만 아니라 Javascript까지 완벽하게 호환되며 프로그래밍 언어를 모르더라도 데이터만 가져와서 그래프를 만들 수 있다. 특히 기본적으로 그래프를 만든 뒤에, 쉽게 색깔이나 점 모양, 크기 등을 세부적으로 조절할 수 있어서 매력적이고, interactive한 그래프이기 때문에 그래프를 돌려본다거나, 확대한다거나 같은 일들도 쉽게 할 수 있다. 기본적으로는 ipython notebook을 사용할 때 이용하지만, 이를 markdown에 embed할 수 있다면 결과물을 만들 때 훨씬 유용할 것이라 생각한다.

---
## 예시
#### 데이터 전처리를 위한 모듈 불러오기
~~~python
import plotly.plotly as py
import pandas as pd
~~~

#### plotly 로그인
~~~python
py.sign_in('your_username','your_api_key')
~~~

#### 데이터 만들고 시각화
~~~python
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

for col in df.columns:
    df[col] = df[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df['text'] = df['state'] + '<br>' +\
    'Beef '+df['beef']+' Dairy '+df['dairy']+'<br>'+\
    'Fruits '+df['total fruits']+' Veggies ' + df['total veggies']+'<br>'+\
    'Wheat '+df['wheat']+' Corn '+df['corn']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df['code'],
        z = df['total exports'].astype(float),
        locationmode = 'USA-states',
        text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Millions USD")
        ) ]

layout = dict(
        title = '2011 US Agriculture Exports by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )

fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
~~~
---
## 결과
기본적으로 png나 jpg로 export 가능하기 때문에 markdown에 그 결과를 사진으로 보여주는 것은 어렵지 않다.
~~~markdown
![결과](/your_directory/your_graph.jpg)
~~~
![결과](/assets/images/2017-07-27-Embed-Plotly/graph.png)

그러나 이는 단순 스크린샷일 뿐, 세부적으로 각 항목이 수치가 얼마나 되는지도 알 수 없고 그래프가 복잡해지거나 3차원이 될 때 보기가 힘들다.

따라서 이를 해결하기 위해 다음과 같은 과정을 거치면 interactive한 그래프를 넣을 수 있다.

1. [위의 코드](#데이터-만들고-시각화)를 실행하였을 때 그래프 우측 하단에 있는 **edit chart** 클릭
2. 여기서 세부적인 색깔 조정, 크기 조정 등을 할 수 있음. 좌측 하단에 있는 **share** 클릭
![share](/assets/images/2017-07-27-Embed-Plotly/share.png)
3. 거기서 **embed** 를 클릭하면 본인 plotly 사이트를 통해 연동할 수 있는 iframe 링크 생성 가능!
![embed](/assets/images/2017-07-27-Embed-Plotly/embed.png)

#### interactive map visualization

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~sungnampark/473.embed"></iframe>
