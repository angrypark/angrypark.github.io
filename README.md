# AngryPark Blog

Hugo와 PaperMod 테마를 사용하여 만든 개인 블로그입니다.

## 기술 스택

- **정적 사이트 생성기**: Hugo
- **테마**: PaperMod
- **배포**: GitHub Pages

## 로컬 개발 환경

### 사전 요구사항

- Hugo 설치 (https://gohugo.io/installation/)

### 실행 방법

1. 저장소 클론
```bash
git clone https://github.com/angrypark/angrypark.github.io.git
cd angrypark.github.io
```

2. 서브모듈 업데이트
```bash
git submodule update --init --recursive
```

3. 로컬 서버 실행
```bash
hugo server --buildDrafts --buildFuture
```

4. 브라우저에서 `http://localhost:1313` 접속

## 새 포스트 작성

```bash
hugo new posts/포스트-제목.md
```

## 빌드

```bash
hugo
```

빌드된 파일은 `public/` 디렉토리에 생성됩니다.

## 배포

GitHub Pages를 통해 자동 배포됩니다. `main` 브랜치에 푸시하면 자동으로 사이트가 업데이트됩니다.

## 라이선스

MIT License
