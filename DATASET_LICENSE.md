# Twenty Newsgroups 출처와 라이선스

## 출처와 귀속

- 데이터셋: [Twenty Newsgroups — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/113/twenty%2Bnewsgroups)
- 생성자: Tom Mitchell
- UCI 표기 인용: Mitchell, T. (1997). *Twenty Newsgroups* [Dataset]. UCI Machine Learning Repository.
- DOI: `10.24432/C5C323`
- 프로젝트 loader: [scikit-learn `fetch_20newsgroups`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)

UCI는 이 데이터셋을 [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0)로 표시한다. 이 조건은 적절한 출처를 밝히고 변경 여부를 표시하면 공유와 개작을 허용한다. 이 프로젝트는 세 카테고리만 선택하고 아래의 privacy pipeline을 적용한 파생 데이터를 사용한다.

## 이 저장소의 처리 정책

- 원본 newsgroup 메시지는 저장소에 재배포하지 않는다.
- scikit-learn loader는 실행 환경의 로컬 데이터 캐시에 원본을 내려받을 수 있다.
- loader 경계에서 headers, footers, quotes를 제거한다.
- 명확한 email, phone, URL과 유효한 IPv4·IPv6를 redaction한다.
- sanitization 후 빈 문서를 폐기한다.
- 정제된 전체 문서는 실행 중 TF-IDF, 분류와 검색 행렬 생성에만 사용하고 저장하지 않는다.
- runtime artifact와 report에는 detector를 다시 통과한 최대 240자의 snippet만 저장한다.
- build가 실제 redaction, 빈 문서 제외와 카테고리별 유지 통계를 `privacy_report.json`에 기록한다.

이 처리는 원본에 대한 변경이다. CC BY 4.0 귀속 의무와 별개로, 원문에 적용될 수 있는 개인정보·퍼블리시티·인격권 등 다른 권리가 모두 해결되었다고 가정하지 않는다. structured redaction은 자유형 엔터티까지 탐지하는 PII-free 보증이 아니므로, 이 저장소는 원문과 정제된 전체 문서를 배포하지 않고 제한된 snippet과 수치 파생물만 저장한다.
