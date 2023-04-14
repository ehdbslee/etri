# Etri (Sleep Dataset)

> "무작위 모달 결손에 강건한 멀티모달 수면 질 예측 인공지능 기법" 논문의 실험 코드


## Development Environment

* Window 10
* GPU: NVIDIA GeForce RTX 3090


## Requirements
* **[requirements.txt](https://github.com/ehdbslee/etri/blob/main/requirements.txt)** 에서 필요한 패키지를 확인하여 설치하여 주십시오. 
* *torchsummary*의 경우 모델의 구조 및 total parameter를 확인하기 위해 사용하였으므로 필수 패키지는 아닙니다.
* cmd창(혹은 anaconda)에서 아래의 명령어로 설치할 수 있습니다. 
 
```pip install package_name=version```

## Dataset
ETRI 라이프로그 데이터셋: [Assessing Sleep Quality Using Mobile EMAs: Opportunities, Practical Consideration, and Challenges](https://ieeexplore.ieee.org/document/9667514).

2020년 수면 측정 데이터셋 중, 19개의 멀티모달 특성과, 수면의 질을 표현하는 1개의 레이블(sleep score)로 구성되어 있다.

아래의 표와 같이 특성의 modality에 따라 4개의 군집을 형성하였다.

```표 1:``` Modality에 따른 특성 분류
 |       Modality     |Features                                                                                                |
 |-----               |-----                                                                                                          |
 |Modal 1(시간 특성)   | startDt, endDt, lastUpdate                                                                                    |
 |Modal 2(가속도 특성) | wakeupduration, wakeupcount, durationtosleep, durationtowakeup                                                |
 |Modal 3(뇌파 특성)   | lightsleepduration, deepsleepduration, remsleepduration                                                       |
 |Modal 4(신체 특성)   | hr_average, hr_min, hr_max, rr_average, rr_min, rr_max, breathing_disturbances_intensity, snoring,snoringepisodecount|
 
 
-----




## Running the experiments

* ```Single_modal.ipynb:``` 센서의 multi-modal 전처리 없이, 각 modal 특성을 단순 접합하여 single-modal 데이터로 취급하여 훈련한 모델 구현
* ```Multi_modal.ipynb:``` 제안하는 multi-modal 예측 구조 (그림 참조)
