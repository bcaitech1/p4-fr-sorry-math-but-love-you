# 🏆To be Modellers and Beyond!

![logo2](C:\Users\iloveslowfood\Documents\workspace\p4-fr-sorry-math-but-love-you\images\logo2.png)

## Summary

- 본 대회의 주제는 수식인식이었습니다. 어쩌고 저쩌고 해가지고 이랬다

![example3](C:\Users\iloveslowfood\Documents\workspace\p4-fr-sorry-math-but-love-you\images\example3.png)



#### 대회 결과

* 12팀 중 1위

* Public LB Score: 0.8574 / Private LB Score: 0.6288



## Usage

### Requirements

```shell
pip install -r requirments.txt
```



### Train

```shell
# Train with integrated optimizer for Encoder and Decoder
$ python train.py --config_name './configs/EfficientSATRN.yaml'

# Train with allocating individual lr for Encoder and Decoder
$ python train_dual_opt.py --config_name './configs/EfficientSATRN.yaml'
```



### Inference







```shell
[folder]
│
├── configs/
├── data_tools/
├── networks/
├── postprocessing/
├── schedulers/
├── utils/
├── README.md
├── requirements.txt
├── train.py
├── train_dual_opt.py
├── ensemble.py
└── inference.py
```

## Usage

### Training

```sh
python train.py
```


### Evaluation

```sh
python evaluate.py
```

[arxiv-zhang18]: https://arxiv.org/pdf/1801.03530.pdf
[CROHME]: https://www.isical.ac.in/~crohme/
[Aida]: https://www.kaggle.com/aidapearson/ocr-data
[Upstage]: https://www.upstage.ai/
[IM2LATEX]: http://lstm.seas.harvard.edu/latex/
[pytorch]: https://pytorch.org/
