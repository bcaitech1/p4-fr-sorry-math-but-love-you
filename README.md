# 업스테이지 수학 수식 OCR 모델

## Requirements

- Python 3
- [PyTorch][pytorch]

All dependencies can be installed with PIP.

```sh
pip install tensorboardX tqdm pyyaml psutil
```

현재 검증된 GPU 개발환경으로는
- `Pytorch 1.0.0 (CUDA 10.1)`
- `Pytorch 1.4.0 (CUDA 10.0)`
- `Pytorch 1.7.1 (CUDA 11.0)`


## Supported Models

- [CRNN][arxiv-zhang18]
- [SATRN](https://github.com/clovaai/SATRN)
- [ASTER](https://github.com/bgshih/aster)
- [EfficientNetV2](https://github.com/google/automl/tree/master/efficientnetv2)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)


## Supported Data
- [Aida][Aida] (synthetic handwritten)
- [CROHME][CROHME] (online handwritten)
- [IM2LATEX][IM2LATEX] (pdf, synthetic handwritten)
- [Upstage][Upstage] (print, handwritten)


모든 데이터는 팀 저장소에서 train-ready 포맷으로 다운 가능하다.
```
[dataset]/
├── gt.txt
├── tokens.txt
└── images/
    ├── *.jpg
    ├── ...     
    └── *.jpg
```

폴더 구조
```
[folder]
│
├── configs/
│	├── EfficientASTER.yaml
│	├── EfficientSATRN.yaml
│	├── SATRN.yaml
│	├── SWIN.yaml
│	├── data_info.txt
│	└── tokens.txt
│
├── data_tools/
│	├── augmentations.py
│	├── dataset.py
│	└── loader.py
│
├── networks/
│	├── EfficientASTER.py
│	├── EfficientSATRN.py
│	└── SWIN.py
│
├── postprocessing/
│	├── decoding.py
│	└── postprocessing.py
│
├── schedulers/
│	├── circular_lr.py
│	├── cosineannealing.py
│	└── tf_scheduler.py
│
├── utils/
│	├── checkpoint.py
│	├── ensemble_utils.py
│	├── data_utils.py
│	├── flags.py
│	├── metrics.py
│	└── utils.py
│
├── README.md
├── requirements.txt
├── train.py
├── train_dual_opt.py
├── ensemble_v2.py
└── inference.py

```

## Usage

### Training

```sh
python train.py
```


### Evaluation

```sh
python inference.py
```

[arxiv-zhang18]: https://arxiv.org/pdf/1801.03530.pdf
[CROHME]: https://www.isical.ac.in/~crohme/
[Aida]: https://www.kaggle.com/aidapearson/ocr-data
[Upstage]: https://www.upstage.ai/
[IM2LATEX]: http://lstm.seas.harvard.edu/latex/
[pytorch]: https://pytorch.org/
