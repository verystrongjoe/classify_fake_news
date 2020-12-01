# 부동산 뉴스 제목을 기준으로 한 (관련없는) 가짜 부동산 뉴스 필터링 알고리즘 

## 1. 데이터 
아래 테이블에서 뉴스 제목(title) 가짜 여부(is_valid), 일자(yyyymmdd) 정도 조회
select yyyymmdd,title,is_valid from richgo.realestate_news

## 2. 사용한 모델
간단한 머신러닝 기법인 naive bayes classifier을 사용하였음

## 3. 모델 설명
 - Naive Bayes Theorem을 더 단순환하여 사용한 토큰들의 조합의 확률로 가짜/진짜를 판별
   theorem example to predict fake news
 - P(Y|X) = (P(X|Y) * P(Y)) / P(X)  -> P(X|Y) * P(Y)
 - Naive라는 것은 Prior확률이 각각 독립적으로 발생했다라고 가정함으로써 단순 Joint probabilities 로
   조합의 확률을 계산( 현실에서 추적하기도 어렵고 복잡한 상황을 고려안해도 됨)

## 4. 추후 업그레이드 제안

부동산 뉴스가 계속 새로운 토큰들이 유입된다고 가정했을 때 Semi supervised learning NB classfier로 접근
가능하며, 여기에서 사용자가 클릭한 정보를 활용할수 있으며, 이 클릭 안한 것의 의미는 가짜 부동산 뉴스일수도 있지만
별로 관심이 없다고 봐도 가정한다. 이 또한 무의미하여 필터링 한다라는 생각으로 접근

이런 것을 고려하기 위해서는 Semi supervised learning NB classfier이 필요한데
초반에 수동으로 라벨링하여 레이블있는 샘플 M개를 모으고 되지 않는 N개의 샘플이 있다 가정
데이터가 만약 D = M + N이라고 한다면 먼저 M개의 샘플을 가지고 나이브 베이즈 적용함.
수렴할때까지
  - D에 속하는 모든 샘플 X에 대해서 P(C|x)를 예측
  - 이전 단계에서 예측된 확률 기반하여 다시 훈련
  - 참조 코드 :   https://github.com/aboyker/semi-supervised-bayesian-classifier

## 5. references
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html
