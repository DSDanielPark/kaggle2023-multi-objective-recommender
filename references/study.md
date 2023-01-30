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


## Study

### About Co-visitation Metric

[1] Inference of Suspicious Co-Visitation and Co-Rating Behaviors and Abnormality Forensics for Recommender Systems <br>
**Z. Yang, Q. Sun, Y. Zhang, L. Zhu and W. Ji, "Inference of Suspicious Co-Visitation and Co-Rating Behaviors and Abnormality Forensics for Recommender Systems," in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 2766-2781, 2020, doi: 10.1109/TIFS.2020.2977023.**
<br>
![](https://github.com/DSDanielPark/kaggle2023-multi-objective-recommender/imgs/img1.jpg)

