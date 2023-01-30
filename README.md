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
<br> Competition is to predict e-commerce clicks, cart additions, and orders

- **Data Description:** <br>
  - 구체적인 태스크는 aid 세션이 잘린 뒤에 나올 다음 클릭과 및 장바구니에 추가될 나머지 항목을 예측하는 것이며, 각 세션별로 최대 20개의 값을 예측할 수 있음.

```
- train.jsonl - 전체 시퀀스의 train set (10.5GB)
  - session- 고유한 세션 ID
  - events- 세션에서 발생한 이벤트의 시계열 데이터
    - aid- 관련 이벤트 품목 ID(제품 코드)
    - ts-  이벤트 타임스탬프
    - type- 3가지 이벤트 유형(click, add to carts, order)
- test.jsonl - 시퀀스 일부가 잘린 test set (383MB)
```

- **EDA:**
<br> You can see simple data description in official [OTTO data set github.](https://github.com/otto-de/recsys-dataset)
<br>

- **Data Sample**:
  - aid: product node
  - ts: time stamping
  - type: action type

`Raw data for one session`
```
[{'aid': 424964, 'ts': 1659304800025, 'type': 'carts'},
 {'aid': 1492293, 'ts': 1659304852871, 'type': 'clicks'},
 {'aid': 1492293, 'ts': 1659304863627, 'type': 'carts'},
 {'aid': 910862, 'ts': 1659304891923, 'type': 'clicks'},
 {'aid': 910862, 'ts': 1659304900209, 'type': 'carts'},
...
 {'aid': 1464360, 'ts': 1661447432716, 'type': 'clicks'},
 {'aid': 207905, 'ts': 1661686200666, 'type': 'clicks'},
 {'aid': 1628317, 'ts': 1661686382213, 'type': 'clicks'},
 {'aid': 376932, 'ts': 1661705869789, 'type': 'clicks'},
 {'aid': 497868, 'ts': 1661714854992, 'type': 'clicks'}]
```

`Other view`
```
	session	  aid	      ts	     type
0 	  0	    1517085	1659304800025	clicks
1 	  0	    1563459	1659304904511	clicks
2	  0	    1309446	1659367439426	clicks
3	  0	    16246	1659367719997	clicks
4	  0	    1781822	1659367871344	clicks
...	  ...	  ...	    ...	          ...
10676	99	369914	1661390037098	clicks
10677	99	759787	1661713071231	clicks
10678	99	759787	1661713144577	clicks
10679	99	1400630	1661713177535	clicks
10680	99	1358447	1661713244707	clicks
```




<br>


<br><br>



## References
### About Co-visitation Metric
#### [1] Inference of Suspicious Co-Visitation and Co-Rating Behaviors and Abnormality Forensics for Recommender Systems <br>
<!--![alt text](https://github.com/DSDanielPark/kaggle2023-multi-objective-recommender/blob/main/imgs/img1.jpg?raw=true)-->
*Z. Yang, Q. Sun, Y. Zhang, L. Zhu and W. Ji, "Inference of Suspicious Co-Visitation and Co-Rating Behaviors and Abnormality Forensics for Recommender Systems," in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 2766-2781, 2020, doi: 10.1109/TIFS.2020.2977023.*

<img src="../imgs/img1.jpg" width="600">
<br>
<br>

#### [2] Co-visitation network in tourism-driven peri-urban area based on social media analytics: A case study in Shenzhen, China
*Yao Sun, Yiwen Shao, Edwin H.W. Chan,
Co-visitation network in tourism-driven peri-urban area based on social media analytics: A case study in Shenzhen, China,
Landscape and Urban Planning,
Volume 204,
2020,
103934,
ISSN 0169-2046,
https://doi.org/10.1016/j.landurbplan.2020.103934.* <br>

<img src="../imgs/img2.png" width="600">





<br>

## Useful Sources

|No|Description|URL|
|:---:|:---|:---|
|1| Python scikit for recommender systems | [Surprise](https://surprise.readthedocs.io/en/stable/index.html)|
|2| Python tensorflow for recommender systems | [TensorFlow Recommenders](https://github.com/tensorflow/recommenders)|
|3| A python library of evalulation metrics and diagnostic tools for recommender systems. | [Ricmetric](https://github.com/statisticianinstilettos/recmetrics) |
|4| cuDF - GPU DataFrames | [cuDF](https://github.com/rapidsai/cudf) |
|5| Stellar Graph Demos| [StellarGraph demos](https://stellargraph.readthedocs.io/en/stable/demos/index.html)|
|6| Conda Colab | [Conda Colab](https://github.com/conda-incubator/condacolab)
|7| NetworkX | [Documentation](https://networkx.org/documentation/stable/index.html)


<br>

## Tips [Optional]

### 1 About using CUDF
- Competition data size is so huge that I need to use gpu acceleration on preprocessing. This is 30x faster than using Pandas CPU
- cudf library supports only Linux OS. Cause I could't use linux server, I install 
cudf in Google Colab.
- Check rapidsai-community example on [here.](https://github.com/rapidsai-community/showcase/blob/main/getting_started_tutorials/10min_to_cudf_colab.ipynb?nvid=nv-int-tblg-386840-vt27#cid=av02_nv-int-tblg_en-us)

```
!pip install cudf-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
```

#### 1.1 Use conda in GoogleColab
```python
>>> !nvidia-smi                          # check type of runtime
>>> !conda --version                     # check if you can use conda in kernel
/bin/bash: conda: command not found

>>> !pip install -q condacolab           # install conda colab
>>> import condacolab
>>> condacolab.install()
>>> condacolab.check()
✨🍰✨ Everything looks OK!

```

#### 1.2 Install cudf

```
!conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=22.10 python=3.9 cudatoolkit=11.5
```

#### 1.3 Inspite of sucessful installation, you may can see error message in importing cudf library. Change type of runtime until you can find some server that can have RAPIDS compatible GPU. 

```
ModuleNotFoundError: No module named 'cudf'
```
=> There wasn't a RAPIDS compatible GPU connected to the Colab instance.

Just remember that the RAPIDS-Colab install script will check if you have a RAPIDS compatible GPU and let you know within the first 15 seconds. Instead of erroring out, it will print out the issue and resolution steps, while NOT installing RAPIDS, as to not waste your time on something that won't work.

```python
!nvidia-smi
import torch
torch.cuda.is_available()
```
- Final check
```python
>>> import cudf, itertools
>>> print('We will use RAPIDS version',cudf.__version__)
We will use RAPIDS version 22.12.0
```


<br><br>

### 2. jsonl to parquet function
- 본 프로젝트의 train 데이터 셋의 경우, 10GB가 넘으므로 chunk_size를 통해서 분할 저장하여, concat 하는 등의 별도 전처리가 필요하며, cudf 사용을 위해 parquet 포맷으로 변경하는 것을 추천합니다.
```python
#pip install pyarrow
#pip install fastparquet
import pandas as pd
def jsonl_to_parquet(input_jsonl_fpath:str, save_parquet_dpath: str, chunk_size: int) -> pd.core.frame.DataFrame:
    '''
    jsonl_to_parquet('../data/otto-recommender-system\\test.jsonl', '.')                # for small size jsonl file
    jsonl_to_parquet('../data/otto-recommender-system\\train.jsonl', '.', 100000)       # for large size jsonl file
    '''
    if chunk_size == None:
        chunks = pd.read_json(path_or_buf=input_jsonl_fpath, lines=True)
        df = pd.DataFrame(chunks)
        try:
            df.to_parquet(f'{save_parquet_dpath}/result.parquet')
        except Exception as e:
            print(f'Error occurs: {e}')
    elif chunk_size != None:
        assert type(chunk_size) == int, "type of chunk_size should be integer"
        chunks = pd.read_json(path_or_buf=input_jsonl_fpath, lines=True, chunksize=chunk_size)
        for i, c in enumerate(chunks):
            temp_df = pd.DataFrame(c)
            try:
                temp_df.to_parquet(f'{save_parquet_dpath}/result{i}.parquet') 
            except Exception as e:
                print(f'Error occurs: {e}')   
```

- Data loader reference

```python
%%time
# CACHE FUNCTIONS
def read_file(f):
    return cudf.DataFrame(data_cache[f])
def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts/1000).astype('int32')
    df['type'] = df['type'].map(type_labels).astype('int8')
    return df

# CACHE THE DATA ON CPU BEFORE PROCESSING ON GPU
data_cache = {}
type_labels = {'clicks':0, 'carts':1, 'orders':2}
files = glob.glob('../input/otto-chunk-data-inparquet-format/*_parquet/*') #parquet format path
for f in files: data_cache[f] = read_file_to_cache(f)

# CHUNK PARAMETERS
READ_CT = 5
CHUNK = int( np.ceil( len(files)/6 ))
print(f'We will process {len(files)} files, in groups of {READ_CT} and chunks of {CHUNK}.')  
```





<br>
<br>

### Daily Commit Summary <br>
|Date|Description|
|:---:|:---|
|23.01.26|- 데이터 셋업 및 태스크 확인 <br> - Multi Object에 대한 레퍼런스 확인|
|23.01.27|- 간단한 ML 모델 학습 진행 <br> - 대용량 데이터 전처리 시작|
|23.01.29|- 휴식, 간단한 ML 모델 결과 확인|
|23.01.30|- 대용량 데이터 처리 <br> - Co-Visitation, Co-Ranking Study |

<br><br>
