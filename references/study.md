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
<br>

## References Study
<Br>

## Study

### About Co-visitation Metric

#### [1] Inference of Suspicious Co-Visitation and Co-Rating Behaviors and Abnormality Forensics for Recommender Systems <br>
<!--![alt text](https://github.com/DSDanielPark/kaggle2023-multi-objective-recommender/blob/main/imgs/img1.jpg?raw=true)-->

<img src="../imgs/img1.jpg" width="600">
<br>


<br>
<br>

## Useful Library or Sources
|No|Description|URL|
|:---:|:---|:---|
|1| Python scikit for recommender systems | [Surprise](https://surprise.readthedocs.io/en/stable/index.html)|
|2| Python tensorflow for recommender systems | [TensorFlow Recommenders](https://github.com/tensorflow/recommenders)|
|3| A python library of evalulation metrics and diagnostic tools for recommender systems. | [Ricmetric](https://github.com/statisticianinstilettos/recmetrics) |
|4| cuDF - GPU DataFrames | [cuDF](https://github.com/rapidsai/cudf) |

<br>
<br>


## Tips [Optional]
<br>

### Use conda in GoogleColab
```
    !nvidia-smi                          # check runtype
    !conda --version                     # check if you can use conda in kernel
/bin/bash: conda: command not found
    !pip install -q condacolab           # install conda colab
    import condacolab
    condacolab.install()
```
Install cudf
```
!conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=22.10 python=3.9 cudatoolkit=11.5
```
