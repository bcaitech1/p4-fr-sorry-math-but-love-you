# 🏆수식 인식: To be Modeler and Beyond!

<div style="text-align:center"><img src=https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/master/images/logo2.png?raw=true /></div>

# Contents

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🧐Task Description](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you#task-description-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🏆Project Result](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you#project-result-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[⚙Installation](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you#installation-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🕹Command Line Interface](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you#command-line-interface-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[🤝Collaboration Tools](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you#collaboration-tools-1)**

#### &nbsp;&nbsp;&nbsp;&nbsp;**[👩‍👦‍👦Who Are We?](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you#who-are-we-1)**

# Task Description

### Subject

본 대회의 주제는 수식 이미지를 [LaTex](https://ko.wikipedia.org/wiki/LaTeX) 포맷의 텍스트로 변환하는 문제였습니다. LaTex은 논문 및 기술 문서 작성 포맷으로, 자연 과학 분야에서 널리 사용됩니다. 일반적인 광학 문자 인식(optical character recognition)과 달리 수식인식은 multi-line recognition을 필요로 합니다.

![](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/master/images/task_intro1_2.png?raw=true)



일반적 문장과 달리 수식은 분수의 분자·분모, 극한의 구간 표현 등 다차원적 관계 파악이 필요합니다. 따라서 수식인식 문제는 일반적인 single line recognition 기반의 OCR이 아닌 multi line recognition을 이용하는 OCR 문제로 바라볼 수 있습니다. Multi line recognition의 관점에서 수식 인식은 기존 OCR과 차별화되는 task라고 할 수 있습니다.

### Data

- 학습 데이터: 출력물 수식 이미지 5만 장, 손글씨 수식 이미지 5만 장, 총 10만 장의 수식 이미지

- 테스트 데이터: 출력물 수식 이미지 6천 장, 손글씨 수식 이미지 6천 장

  

### Metric

- 평가 척도: 0.9 × 문장 단위 정확도 + 0.1 × (1 - 단어 오류율)

- 문장 단위 정확도(Sentence Accuracy): 전체 추론 결과 중 몇 개의 수식이 정답과 정확히 일치하는 지를 나타낸 척도입니다.


- 단어 오류율(Word Error Rate, WER): 추론 결과를 정답에 일치하도록 수정하는 데 단어의 삽입, 삭제, 대체가 총 몇 회 발생하는 지를 측정하는 척도입니다.

  ![](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/master/images/metric.png?raw=true)



# Project Result

* 12팀 중 1위

* Public LB Score: 0.8574 / Private LB Score: 0.6288

* 1등 솔루션 발표 자료는 [이곳](https://drive.google.com/file/d/1aXhJ7-cEXDKa1Y_9vOBdydOdIfACZrVG/view)에서 확인하실 수 있습니다.

* 수식 인식 결과 예시

![](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/master/images/example1_2.png?raw=true)

  

# Installation

```shell
# clone repository
git clone https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you.git

# install necessary tools
pip install -r requirements.txt
```

### Dataset Structure

```shell
[dataset]/
├── gt.txt
├── tokens.txt
└── images/
    ├── *.jpg
    ├── ...     
    └── *.jpg
```

### Code Structure

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

#### Train with single optimizer

```shell
$ python train.py --train_type single_opt --config_file './configs/EfficientSATRN.yaml'
```

#### Train with two optimizers for encoder and decoder

```shell
$ python train.py --train_type dual_opt --config_file './configs/EfficientSATRN.yaml'
```

#### Knowledge distillation training

```shell
$ python train.py --train_type distillation --config_file './configs/LiteSATRN.yaml' --teacher_ckpt 'TEACHER-MODEL_CKPT_PATH'
```

#### Train with Weight & Bias logging tool

```shell
$ python train.py --train_type single_opt --project_name <PROJECTNAME> --exp_name <EXPNAME> --config_file './configs/EfficientSATRN.yaml'
```

#### Arguments

##### `train_type (str)`: 학습 방식

* `'single_opt'`: 단일 optimizer를 활용한 학습을 진행합니다.
* `'dual_opt'`: 인코더, 디코더에 optimizer가 개별 부여된 학습을 진행합니다.
* `'distillation'`: Knowledge Distillation 학습을 진행합니다.

##### `config_file (str)`: 학습 모델의 configuration 파일 경로

- 모델 configuration은 아키텍처별로 상이하며, [이곳](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/master/configs/EfficientASTER.yaml)에서 해당 예시를 보실 수 있습니다.
- 학습 가능한 모델은 ***[EfficientSATRN](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/EfficientSATRN.py#L664)***, ***[EfficientASTER](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/EfficientASTER.py#L333)***, ***[SwinTRN](https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/7502ec98b49999eaf19eed3bc05a57e0d712dfde/networks/SWIN.py#L1023)***,    ***[LiteSATRN](https://github.com/iloveslowfood/p4-fr-sorry-math-but-love-you/blob/3ffa06229659505fc2b4ef2ec652168b4ff7857b/networks/LiteSATRN.py#L548)*** 입니다.

##### `teacher_ckpt (str)`: Knowledge Distillation 학습 시 불러올 Teacher 모델 checkpoint 경로

##### `project_name (str)`: (optional) 학습 중 [Weight & Bias](https://wandb.ai/site) 로깅 툴을 활용할 경우 사용할 프로젝트명

##### `exp_name (str)`: (optional) 학습 중 [Weight & Bias](https://wandb.ai/site) 로깅 툴을 활용할 경우 사용할 실험명

---

## Inference

#### Inference with single model

```shell
$ python inference.py --inference_type single --checkpoint <MODELPATH.pth>
```

#### Ensemble inference

```shell
$ python inference.py --inference_type ensemble --checkpoint <MODEL1PATH.pth> <MODEL2PATH.pth> ...
```

#### Arguments

##### `inference_type (str)`: 추론 방식

- `single`: 단일 모델을 불러와 추론을 진행합니다.
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


# Collaboration Tools
<table>
    <tr height="200px">
        <td align="center" width="350px">	
            <a href="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/issues?q=is%3Aissue+is%3Aclosed"><img height="180px" width="320px" src="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/master/images/issue.gif?raw=true"/></a>
            <br />
            <a href="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/issues?q=is%3Aissue+is%3Aclosed">Github Issues</a>
        </td>
        <td align="center" width="350px">	
            <a href="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/discussions"><img height="180px" width="320px" src="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/master/images/discussion.gif?raw=true"/></a>
            <br />
            <a href="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/discussions">Github Discussions</a>
        </td>
    </tr>
    <tr height="200px">
        <td align="center" width="350px">	
            <a href="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/pulls?q=is%3Apr+is%3Aclosed"><img height="180px" width="320px" src="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/master/images/pr.gif?raw=true"/></a>
            <br />
            <a href="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/pulls?q=is%3Apr+is%3Aclosed">Github Pull Requests</a>
        </td>
        <td align="center" width="350px">	
            <a href="https://wandb.ai/smbly"><img height="180px" width="320px" src="https://github.com/bcaitech1/p4-fr-sorry-math-but-love-you/blob/master/images/wandb.gif?raw=true"/></a>
            <br />
            <a href="https://wandb.ai/smbly">Experiments Logging(W&B)</a>
        </td>
    </tr>
</table>

# Who Are We?

<table>
    <tr height="140px">
        <td align="center" width="130px">	
            <a href="https://github.com/iloveslowfood"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/48649606?v=4"/></a>
            <br />
            <a href="https://github.com/iloveslowfood">고지형<br />silkstaff@naver.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/ahaampo5"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/60084351?v=4"/></a>
            <br />
            <a href="https://github.com/ahaampo5">김준철<br />ahaampo5@gmail.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/doritos0812"><img height="100px" width="100px" src="https://raw.githubusercontent.com/doritos0812/p4-fr-sorry-math-but-love-you/master/KakaoTalk_Image_2021-06-20-17-25-08.jpeg"/></a>
            <br />
            <a href="https://github.com/doritos0812">김형민<br />doritos2498@gmail.com</a>
        </td>
    </tr>
    <tr height="140px">
        <td align="center" width="130px">
            <a href="https://github.com/nureesong"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/76163168?v=4"/></a>
            <br />
            <a href="https://github.com/nureesong">송누리<br />nuri3136@naver.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/Lala-chick"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/76460750?v=4"/></a>
            <br />
            <a href="https://github.com/Lala-chick">이주영<br />vvvic313@gmail.com</a>
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/soupbab"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/67000572?v=4"/></a>
            <br />
            <a href="https://github.com/soupbab">최준구<br />jungu1106@naver.com</a>
        </td>
    </tr>
</table>
