---
title: "15분만에 깃허브 블로그 만들기"
layout: post
date: 2017-07-20 02:23
image: /assets/images/170720/background.jpeg
headerImage: false
tag:
- github
- blog
- jekyll
category: blog
author: angrypark
description: 깃허브 블로그를 시작하며
# jemoji: '<img class="emoji" title=":ramen:" alt=":ramen:" src="https://assets.github.com/images/icons/emoji/unicode/1f35c.png" height="20" width="20" align="absmiddle">'
---

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">
## 요약

첫 포스트에서는 내가 시도했던 방법들 중 **가장 쉽게** 깃허브 블로그를 시작할 수 있는 방법을 소개하고자 한다.

- [왜 깃허브 블로그인가?](#왜-깃허브-블로그인가)
- [설치](#설치)
- [참고](#참고)

---
## 왜 깃허브 블로그인가?

개발자를 꿈꾸는 사람이나, 코딩을 통해 전문성을 키우고 싶은 사람이라면 누구나 한번쯤 본인의 이름으로 된 간지나는 블로그를 만들어서 거기에 본인의 프로젝트 결과물을 저장해보는 상상을 해볼 것이다. 나 또한 컴퓨터공학 전공은 아니지만, 데이터 분석을 하는 과정에서 코딩을 하는 편이기 때문에, 이를 꿈꾸어 본적은 있지만 실력상 시도해볼 엄두가 안났다. 언어라고는 java, python밖에 모르는데 내 주제에..

그러나 최근에 정말 친절하게 이를 실현시켜주는 다양한 툴들을 알게 되었는데 이 중 대표적인 것이 바로 이 Github blog이고, 이를 연습해보면서 천천히 시작하게 되었다.

물론 워드프레스나 티스토리 등의 툴들이 강력하지 않다는 것은 아니지만, Github Blog는 누구나 15분만 투자하면 본인의 이름으로 된 블로그를 무료로 만들 수 있고, html 언어를 몰라도 본인 프로젝트 결과물을 쉽게 포스팅할 수 있으며, 무엇보다 상당히 예쁜 디자인의 블로그들을 쉽게 복붙할 수 있다는 장점들이 있다. Git을 쓴다는 간지도 있음ㅋ

---
## 설치
먼저 **Git**, **Jekyll**이 설치되어 있어야 한다.

**1. Git 설치** : [Windows(링크)](http://msysgit.github.com/
), [Mac(링크)](http://sourceforge.net/projects/git-osx-installer/
)

**2. Jekyll 설치**
~~~
sudo gem install jekyll
~~~

**3. Github에서 new repository 생성**

이름은 반드시 [본인 Github 아이디].github.io 이어야 한다.

![Markdown Image](../assets/images/170720/settings.png)


**4. 본인 컴퓨터에 앞으로 블로그 글들을 관리할 폴더 생성**

나는 Desktop에 angrypark이라는 폴더를 만들어 관리하기로 했다.

**5. Github의 빈 저장소 불러오기**
~~~
$cd Desktop/angrypark
$git remote add origin [내 저장소 링크]
$git add .
$git commit -m "빈 저장소 불러오기"
$git push origin master
~~~
그렇게 하면 내 폴더 안에 [본인 아이디].github.io 폴더가 생성된 것을 확인할 수 있다. 안에 내용물이 있다면 필요없으니 바로 지우자.

**6. 예쁜 블로그 템플릿 받아오기**

[Jekyll 테마 페이지](http://jekyllthemes.org/)에서 원하는 테마를 선택 한 후 download한다. 이 때, 안의 내용물들만 복사하여 5번에서 생성된 폴더에 붙여넣는다.

**7. 템플릿 적용**

5번과 같은 방식으로 하면 된다.
~~~
$git status
$git add .
$git commit -m "템플릿 적용"
$git push origin master
~~~

이제 바로 본인의 홈페이지로 들어가보면, 따단! 생성완료~

---
## 참고
Jekyll을 이용하여 github에 블로그 만들기 : [https://brunch.co.kr/@hee072794/39](https://brunch.co.kr/@hee072794/39)

github로 간단한 사이트 만들기&Jekyll Theme 적용하기:
[https://youtu.be/eVc3S5wk18o](https://youtu.be/eVc3S5wk18o)
[https://youtu.be/H5h4s7b6XcU](https://youtu.be/H5h4s7b6XcU)

Markdown-syntax : [https://simhyejin.github.io/2016/06/30/Markdown-syntax](https://simhyejin.github.io/2016/06/30/Markdown-syntax)
