# kaggle2023-multi-objective-recommender
- TASK: Multi-category classifiers(8classes in news dataset) 전자 상거래 로그 기반(클릭, 장바구니 추가) 주문 예측
- Dataset: 독일 전자상거래 OTTO data set <br>
- Evaluation:  Recall@20 for each action type, and the three recall values are weight-averaged <br>
`Score` = 0.10 * `Recall of clicks` + 0.30 * `Recall of carts` + 0.60 * `Recall of Order`
- TASK Duration: Jan 25,2023 - Feb 01,2023 (6days) <br>
- Author: Daniel Park in South Korea https://github.com/DSDanielPark <br>
<br>
Code will be open after data de-identification and refactoring.
<br>

## Remark
- Quick Competition challenges.
<br>

## Outline
- **Problem Definition:** 
<br> competition is to predict e-commerce clicks, cart additions, and orders

- **Data Description:** <br>
  - 구체적인 태스크는 aid 세션이 잘린 뒤에 나올 다음 클릭과 및 장바구니에 추가될 나머지 항목을 예측하는 것이며, 각 세션별로 최대 20개의 값을 예측할 수 있음.

```
- train.jsonl - train용 전체 세션 데이터
  - session- 고유한 세션 ID
  - events- 세션에서 발생한 이벤트의 시계열 데이터
    - aid- 관련 이벤트 품목 ID(제품 코드)
    - ts-  이벤트 타임스탬프
    - type- 이벤트 유형, 즉 제품이 클릭되었는지, 사용자의 장바구니에 추가되었는지 또는 세션동안 주문되었는지 여부
- test.jsonl - 세션의 잘린부분을 포함하는 테스트 데이터
```

- **EDA:**
<br> you can see simple data description in official [github.](https://github.com/otto-de/recsys-dataset)
<br>

<br>


<br><br>

## References <Br>

#### Daily Commit Summary <br>
|Date|Description|
|:---:|:---|
|23.01.26|- 데이터 셋업 및 태스크 확인 <br> - Multi Object에 대한 레퍼런스 확인|
|23.01.27|- 간단한 ML 모델 학습 진행 <br> - 대용량 데이터 전처리 시작|
|23.01.29|- 간단한 ML 모델 결과 확인|

<br><br>
