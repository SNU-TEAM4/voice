# 7단계 Log-Mel CNN 결과

## 모델과 입력

- 입력: 공통 MP3 64kbps 조건의 10초 Log-Mel
- 입력 크기: 1 × 128 mel bands × 249 time bins
- CNN 파라미터: 97,521개
- 장치: mps
- Train sampling: REAL 50%, FAKE 50%; FAKE는 12개 생성기 균형
- 선택 기준: Validation Track EER
- 최적 epoch: 5

128-band Log-Mel은 원래 약 996개 시간 frame을 가지지만, CPU·메모리 사용을
줄이기 위해 네 frame씩 평균하여 249개로 만들었다. 주파수축 128개는 유지했다.

## 통제 조건 Test 곡 단위 모델 비교

| 모델 | EER | ROC-AUC | Macro-F1 | REAL 오탐률 | FAKE 놓침률 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 9.74% | 97.67% | 84.52% | 13.64% | 5.19% |
| RBF-SVM | 9.29% | 95.84% | 80.62% | 22.73% | 5.87% |
| Log-Mel CNN | 10.99% | 96.9% | 80.89% | 18.18% | 6.55% |

- CNN Track EER: **10.99%**
- CNN Track ROC-AUC: **96.90%**
- CNN Macro-F1: **80.89%**

## 생성기별 관찰

- EER이 가장 낮은 생성기: **songgen (0.00%)**
- EER이 가장 높은 생성기: **mubert (20.23%)**

생성기별 결과는 `generator_metrics.csv`에 저장했다. 이 모델은 모든 12개
생성기를 학습에 포함한 표준 모델이므로, 생성기별 수치는 seen 성능이다.

## 해석 시 주의

이번 결과는 seed 42 한 번의 초기 CNN 기준선이다. 최종 보고 전에는 여러 seed로
반복해 평균과 변동을 제시해야 한다. 또한 MusicGen·Suno를 완전히 제외한 CNN을
따로 학습해야 미지 생성기 일반화를 ML 모델과 공정하게 비교할 수 있다.
