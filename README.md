# kaggle2023-multi-objective-recommender
- TASK: Multi-category classifiers(8classes in news dataset) <br>
- 전자 상거래 로그 기반(클릭, 장바구니 추가) 주문 예측
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

- **EDA:**
<br> EDA will be conducted in official dataset [github](https://github.com/otto-de/recsys-dataset)
<br>

<br>

## Experiments
- All model architecture and learning conditions in whole pipeline were fixed except initial layer in MODE5. <br>
- Pre-trained Models are KO-BERT [3], SBERT [5].

|No.<br>(INFERENCE MODE)|Model|DataSet|Result|
|:---:|:---|:---|:---
|MODE`1`|KO-BERT fine tunning| article body ||
|MODE`2`|KO-BERT fine tunning| title ||
|MODE`3`|KO-BERT fine tunning| article body + title ||
|MODE`4`|KO-BERT fine tunning| article body + title + clustering info from SBERT model [5] ||
|MODE`5`|KO-BERT (with modified initial hidden layer) fine tunning | article body + title + clustering info from SBERT model [5] ||
<br>

- You can see whole experiments result in [`experments/exp.md`](https://github.com/DSDanielPark/news-article-classification-using-KO-BERT/blob/main/experiments/exp.md) 

## Result
- Best accuracy

| |MODE`1`|MODE`2`|MODE`3`|MODE`4`|MODE`5`|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Data|article body only|articel title only|article with title|article with title which have clustered infomation from SBERT model|article with title|
|Model|KO-BERT|KO-BERT|KO-BERT|KO-BERT, SBERT|KO-BERT, SBERT with clustered infomation in initial hidden layer|
|TestSet Accuracy|0.|0.|0.|||
|Remark|Model architecture and all conditions in models were fixed.|-|-|-|Model architecture and initial layer are changed only in `MODE5`.|
- As I mentioned above 'Experiments' section, All conditions in whole pipeline was fixed except initial layer in MODE5 and Pre-trained Models were KO-BERT [3], SBERT [5].
<br>

## Discussion
- **If there is anything that needs to be corrected, productive criticism is always welcome.** <br>
### `MODE1 & MODE3` 
  - If the sentence length is sufficiently secured, the title of the article is just information that is a repetition of important keywords in the article. Thus it showed a similar accuracy weather title added or not. The accuracy after first epoch can be ignored as the result of random initialization.
  - The title of the article implies the connotation of very well-refined information. Therefore, it is possible to consider giving a little more weight to the title. however, considering the complexity, it is not worth that much.


### `MODE2` 
  - The BERT model has strengths in semantic inference through a slightly wider context and self-emphasis in the context than existing natural language processing models. Therefore, it shows that it is very difficult to classify the topic with only the use of articel title(limited length).
  - In fact, looking at the learning aspect of MODE2, it seems to converge stably in the train set, but it can be seen that it is not a global minimum at all.
  - it seems difficult for the BERT model to achieve the goal of the task due to the characteristics of Korean like word order and ambiguous expressions in some cases. If I could inference using GPT-3, the result will be much better in same condition.


### `MODE4 & MODE5` 
  - Basic Architecture of BERT models is Next token prediction, hence it is performed to compare the difference between giving information about BERT-based clustering infomation to the initial hidden layer and just putting information about clustering in the sentence.
  - K-means clustering is performed through the euclidean distance of the latent vector generated from the mutilangual-BERT model [5].
  - Since two pretrained BERT model weights were used in the entire pipeline, only information was distilled and embedded when it was determined to be significant to prevent model confusion.
<br>

## Install Issues [Optional]
- Although my time was dead, I want to save your time.
- In `Google Colab`, some resource can handle version conflict, but the others failed to handle this issue. So you need to try 3-4 times while changing resource type(GPU or CPU).<br>
- All issues are caused by legacy `numpy` version in each packages. 
- Do not try to use mxnet acceleration cause mxnet is very incomplete. The issues about mxnet acceleration in some cases have not been closed.
- `**CLONE REPOSITORY AND USE THE MODULES AS IT IS - HIGHTLY RECOMMENDED**`
<br>

#### About `mxnet` install (numpy version conflict)
- If you have problem in process of installing `mxnet`, you need to use `python=<3.7.0`.
- Any other options can not solve the numpy version conflict problem while `pip install mxnet`. 
- Except using python=3.7.xx, *EVERYTHING IS USELESS.*
```
error: subprocess-exited-with-error × 

python setup.py bdist_wheel did not run successfully. │ exit code: 1 
╰─> [890 lines of output] Running from numpy source directory. 

C:\Users\user\AppData\Local\Temp\pip-install-8vepm8z2\numpy_34577a7b4e0f4f25959ef5aa089c5687\numpy\distutils\misc_util.py:476: SyntaxWarning: "is" with a literal. Did you mean "=="? return is_string(s) and ('*' in s or '?' is s) blas_opt_info: blas_mkl_info: No module named 'numpy.distutils._msvccompiler' in numpy.distutils; trying from distutils
```
- You may try to install `mxnet` before installing `gluonnlp`.
<br>
<br>



#### About KO-BERT version conflict
- As a result of conflict from above note(About `mxnet` install), KO-BERT still has version conflict.
```
INFO: pip is looking at multiple versions of boto3 to determine which version is compatible with other requirements. This could take a while.
...
To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip attempt to solve the dependency conflict
```
**Solution of this version conflict**
- If you DO NOT use mxnet with KO-BERT, then remove mxnet in kO-bert `requirements.txt` or adjust(loosen) version as your environments.
- If you DO use mxnet with KO-BERT, then just clone the repository and import moudule as it is.
<br>




#### About Korean NLP Models
- Almost of Korean NLP Models have not been updated often. Thus you should try to install lower version of packages.
- recommendation: `python<=3.7.0` <br>
**Some Tips**
- IndexError: Target 2 is out of bounds. => num_classes error (please check)
- broken pipe error => num_workers error, if you use CPU => check your num of threds, else remove args numworkers.
- If you can not download torch KO-BERT weight with urlib3 or boto3 library error message include 'PROTOCOL_TLS' issue, This is an error related to Amazon aws server download. Thus, => use huggingface interface https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf
- If you have other questions, please send me a e-mail. *All is Well!! Happy Coding!!*
- 이슈는 issue를 생성해주시거나, 메일로 문의주시기 바랍니다.

<br>

#### About random initilization in mxnet
```
RuntimeError: Parameter 'bertclassifier2_dense0_weight' has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks.
```
- Even when random initialization is not normally performed due to variable management problems, you can observe indicators that seem to be successfully learned in the trainset. 
- However, it was observed that the performance of the trainset continued to drop because it could not be directed to the candidate space of other local minima in the loss space. 
- Therefore, unlike other models, in the case of the Bert model, it is recommended to perform indicator checks in the trainset every iter.
- When using the Apache mxnet framework, carefully organize which layers to lock and which layers to update and in what order. Even when refactoring, initialize and update layers carefully. => for save your time.

<br>

#### About random initilization in mxnet

```
The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'XLNetTokenizer'. 
The class this function is called from is 'KoBERTTokenizer'.
```

- First of all, check that num_classes are the same. And make sure classes of torch and mxnet framework are not used interchangeably. Finally, check if there is a conflict or mixed use in the wrapped class as written below. (Ignore the following if it has been resolved.)
- As mentioned above, If it is impossible to download the pretrained KOBERT weights from the aws server, you can download the wrapped weights to the hugging face interface through XLNetTokenizer. [ [3](https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf) ]



<br><br>

## References <Br>
[1] GPT fine-tune https://www.philschmid.de/getting-started-setfit <br>
[2] KO-GPT https://github.com/kakaobrain/kogpt <br>
[3] KO-BERT https://sktelecom.github.io/project/kobert <br>
`download` of kobert-base-v1 https://huggingface.co/gogamza/kobart-base-v1 <br>
[4] Sentence-embedding https://github.com/UKPLab/sentence-transformers <br>
[5] Multilingual SBERT https://www.sbert.net/examples/training/multilingual/README.html <br>
[6] NLP gluon BERT documentation https://nlp.gluon.ai/model_zoo/bert/index.html

- 참고하진 않았지만, 후에 BERT model few shot learning 관련된 을 찾아보다가, mode4와 비슷한 컨셉의 논문을 발견하였습니다.
  [Evaluation of Pretrained BERT Model by Using Sentence Clustering](https://aclanthology.org/2020.paclic-1.32) (Shibayama et al., PACLIC 2020)


#### Daily Commit Summary <br>
|Date|Description|
|:---:|:---|
|23.01.17|- 베이스 코드 작성 <br> - 기간 내 수행 가능한 실험 리스트 작성|
|23.01.18|- 실험결과 정리 <br>- 파이프라인 확정|

<br><br>
