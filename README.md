# 🏆수식 인식: To be Modelers and Beyond!

![logo2](C:\Users\iloveslowfood\Documents\workspace\p4-fr-sorry-math-but-love-you\images\logo2.png)

# Task Description

### Subject

본 대회의 주제는 수식 이미지를 [LaTex](https://ko.wikipedia.org/wiki/LaTeX) 포맷의 텍스트로 변환하는 문제였습니다. LaTex은 논문 및 기술 문서 작성 포맷으로, 자연 과학 분야에서 널리 사용됩니다. 일반적인 광학 문자 인식(optical character recognition)과 달리 수식인식은 multi-line recognition을 필요로 합니다.



![task_intro](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/master/images/task_intro.png?raw=true)



일반적 문장과 달리 수식은 분수의 분자·분모, 극한의 구간 표현 등 다차원적 관계 파악이 필요합니다. 따라서 수식인식 문제는 일반적인 single line recognition 기반의 OCR이 아닌 multi line recognition을 이용하는 OCR 문제로 바라볼 수 있습니다. Multi line recognition의 관점에서 수식 인식은 기존 OCR과 차별화되는 task라고 할 수 있습니다.

### Data

- 학습 데이터: 출력물 수식 이미지 5만 장, 손글씨 수식 이미지 5만 장, 총 10만 장의 수식 이미지
- 테스트 데이터: 출력물 수식 이미지 6천 장, 손글씨 수식 이미지 6천 장



![task_intro2](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/master/images/task_intro2.png?raw=true)



### Metric

- 평가 척도: 0.9 × 문장 단위 정확도 + 0.1 × (1 - 단어 오류율)

- 문장 단위 정확도(Sentence Accuracy): 전체 추론 결과 중 몇 개의 수식이 정답과 정확히 일치하는 지를 나타낸 척도입니다.

  

  ![sa](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/master/images/sa.png?raw=true)

  

- 단어 오류율(Word Error Rate, WER): 추론 결과를 정답에 일치하도록 수정하는 데 단어의 삽입, 삭제, 대체가 총 몇 회 발생하는 지를 측정하는 척도입니다.



![wer](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/master/images/wer.png?raw=true)



# Project Result

* 12팀 중 1위

* Public LB Score: 0.8574 / Private LB Score: 0.6288

  

# Installation

```shell
# clone repository
git clone https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you.git

# install necessary tools
pip install -r requirments.txt
```

## Structure

#### Dataset

```shell
[dataset]/
├── gt.txt
├── tokens.txt
└── images/
    ├── *.jpg
    ├── ...     
    └── *.jpg
```

#### Code

```shell
[code]
├── configs/ # configuration files
├── data_tools/ # modules for dataset
├── networks/ # modules for model architecture
├── postprocessing/ # modules for postprocessing during inference
├── schedulers/ # scheduler for learning rate, teacher forcing ratio
├── utils/ # useful utilities
├── inference_modules/ # modules for inference
├── train_modules/ # modules for train
├── README.md
├── requirements.txt
├── train.py
└── inference.py
```



# Command Line Interface

## Train

#### 단일 옵티마이저 활용 학습

```shell
$ python train.py --train_type single_opt --config_file './configs/EfficientSATRN.yaml'
```

#### 인코더와 디코더에 옵티마이저를 개별 부여한 학습

```shell
$ python train.py --train_type dual_opt --config_file './configs/EfficientSATRN.yaml'
```

#### Weight & Bias 로깅 툴을 활용한 학습

```shell
$ python train.py --train_type single_opt --project_name <PROJECTNAME> --exp_name <EXPNAME> --config_file './configs/EfficientSATRN.yaml'
```

#### Arguments

##### `train_type (str)`: 학습 방식

* `'single_opt'`: 단일 optimizer를 활용한 학습을 진행합니다.
* `'dual_opt'`: 인코더, 디코더에 optimizer가 개별 부여된 학습을 진행합니다.

##### `config_file (str)`: 학습 모델의 configuration 파일 경로

- 모델 configuration은 아키텍처별로 상이하며, [이곳](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/master/configs/EfficientASTER.yaml)에서 해당 예시를 보실 수 있습니다.
- 학습 가능한 모델은 ***[EfficientSATRN](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/EfficientSATRN.py#L664)***, ***[EfficientASTER](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/EfficientASTER.py#L333)***, ***[SwinTRN](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/SWIN.py#L1023)***입니다.

##### `project_name (str)`: (optional) 학습 중 [Weight & Bias](https://wandb.ai/site) 로깅 툴을 활용할 경우 사용할 프로젝트명

##### `exp_name (str)`: (optional) 학습 중 [Weight & Bias](https://wandb.ai/site) 로깅 툴을 활용할 경우 사용할 실험명

---

## Inference

#### 단일 모델 추론

```shell
$ python inference.py --inference_type singular --checkpoint <MODELPATH.pth>
```

#### 앙상블 모델 추론

```shell
$ python inference.py --inference_type ensemble --checkpoint <MODEL1PATH.pth> <MODEL2PATH.pth> ...
```

#### Arguments

##### `inference_type (str)`: 추론 방식

- `singular`: 단일 모델을 불러와 추론을 진행합니다.
- `ensemble`: 여러 모델을 불러와 앙상블 추론을 진행합니다.

##### `checkpoint (str)`: 불러올 모델 경로

- 앙상블 추론시 다음과 같이 모델의 경로를 나열합니다.

  ```shell
  --checkpoint <MODELPATH_1.pth> <MODELPATH_2.pth> <MODELPATH_3.pth> ...
  ```

##### `max_sequence (int)`: 수식 문장 생성 시 최대 생성 길이 (default. 230)

##### `batch_size (int)` : 배치 사이즈 (default. 32)

##### `decode_type (str)`: 디코딩 방식

- ``'greedy'``: 그리디 디코딩 방법으로 디코딩을 진행합니다.
- `'beam'`: 빔서치 방법으로 디코딩을 진행합니다.

##### `decoding_manager (bool)`: DecodingManager 사용 여부

##### `tokens_path (str)`: 토큰 파일 경로

- ***NOTE.*** DecodingManager를 사용할 경우에만 활용됩니다.

##### `max_cache (int)`: 앙상블(`'ensemble'`) 추론 시 인코더 추론 결과를 임시 저장할 배치 수

- ***NOTE.*** 높은 값을 지정할 수록 추론 속도가 빨라지만, 일시적으로 많은 저장 공간을 차지합니다.

##### `file_path (str)`: 추론할 데이터 경로

##### `output_dir (str)`: 추론 결과를 저장할 디렉토리 경로 (default: `'./result/'`)



