# 4단계 기계학습 기준선 결과

## 이 단계에서 한 일

각 10초 구간을 266개의 음향 특징으로 요약한 뒤 Logistic Regression과
RBF-SVM을 학습했다. 266개에는 MFCC, MFCC 변화량, 스펙트럴 중심·대역폭·
rolloff·flatness·contrast, RMS, zero-crossing rate의 평균과 표준편차가 포함된다.

Train만 모델 학습에 사용했고, 모델 설정과 판정 threshold는 Validation으로
선택했다. Test는 선택이 끝난 모델의 최종 성능을 확인할 때만 사용했다.

## 선택된 설정

| 모델 | 선택된 설정 | Validation Segment EER | Validation Track EER |
| --- | --- | --- | --- |
| Logistic Regression | {"C": 1.0} | 18.41% | 8.49% |
| RBF-SVM | {"C": 10.0, "gamma": "scale"} | 17.08% | 8.8% |

## Test 결과

| 모델 | 평가 단위 | 표본 수 | EER | ROC-AUC | Macro-F1 | Balanced Accuracy | REAL 오탐률 | FAKE 놓침률 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | segment | 1374 | 15.76% | 91.79% | 71.02% | 82.95% | 19.7% | 14.41% |
| Logistic Regression | track | 487 | 9.74% | 97.67% | 84.52% | 90.59% | 13.64% | 5.19% |
| RBF-SVM | segment | 1374 | 17.49% | 90.68% | 70.63% | 83.04% | 18.94% | 14.98% |
| RBF-SVM | track | 487 | 9.29% | 95.84% | 80.62% | 85.7% | 22.73% | 5.87% |

- Track EER이 가장 낮은 기계학습 모델: **RBF-SVM**
- 해당 Track EER: **9.29%**
- 해당 Track ROC-AUC: **95.84%**

EER은 낮을수록 좋고 ROC-AUC, Macro-F1, Balanced Accuracy는 높을수록 좋다.
Segment는 10초 한 구간의 판정이고 Track은 한 곡의 최대 3개 점수를 평균한
판정이다.

## 클래스 불균형 처리

Train에는 FAKE가 REAL보다 훨씬 많다. 학습 가중치의 절반을 REAL에, 나머지
절반을 FAKE에 배정했고, FAKE 가중치는 다시 12개 생성기에 균등 배분했다.
Validation과 Test의 표본은 삭제하거나 복제하지 않았다.

## 결과를 해석할 때 주의할 점

이 성능은 현재 데이터와 학습에서 본 생성기에 대한 기준선이다. 모델이 실제
AI 생성 흔적뿐 아니라 코덱이나 생성기별 음질을 이용했을 가능성이 있으므로,
아직 실서비스 성능이라고 해석하면 안 된다. 다음 실험에서 MusicGen과 Suno를
각각 학습에서 완전히 제외하여 미지 생성기 일반화를 확인해야 한다.

## 생성 파일

- `model_selection.csv`: Validation 모델 설정 비교
- `metrics.csv`: Validation/Test 상세 지표
- `segment_predictions.csv`: 10초 구간별 예측
- `track_predictions.csv`: 곡별 평균 예측
- `test_roc_curves.png`: Test ROC 곡선
- `test_confusion_matrices.png`: Test 혼동행렬
- `models/*.joblib`: 저장된 최종 모델
