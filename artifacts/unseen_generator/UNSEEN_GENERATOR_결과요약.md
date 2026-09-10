# 6단계 미지 생성기 일반화 실험 결과

## 실험 질문

MusicGen 또는 Suno 음악을 학습에서 한 번도 보여주지 않아도 AI 생성 음악으로
탐지할 수 있는지 확인했다.

## 데이터 구성

- Train: 기존 train의 REAL + holdout 생성기를 제외한 FAKE
- Validation: 기존 validation의 REAL + holdout 생성기를 제외한 FAKE
- Test: 기존 test의 REAL + holdout 생성기 FAKE만 사용
- 동일 원곡 그룹 분할과 동일 10초 구간 유지
- Original encoding과 Common MP3 64kbps 조건 모두 평가

`seen`은 해당 생성기를 학습에 포함한 기존 모델이고, `unseen`은 해당 생성기를
Train과 Validation에서 완전히 제외한 새 모델이다. 양쪽을 같은 Test 하위집합에
평가했다.

## Test 곡 단위 결과

| 조건 | 모델 | 제외 생성기 | Seen EER | Unseen EER | EER 격차 | Seen ROC-AUC | Unseen ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Common MP3 64kbps | Logistic Regression | musicgen | 9.2% | 20.69% | 11.5% | 96.93% | 86.15% |
| Common MP3 64kbps | Logistic Regression | suno | 8.3% | 12.95% | 4.66% | 98.92% | 97.9% |
| Common MP3 64kbps | RBF-SVM | musicgen | 18.39% | 25.29% | 6.9% | 91.75% | 81.87% |
| Common MP3 64kbps | RBF-SVM | suno | 4.77% | 8.41% | 3.64% | 98.01% | 97.16% |
| Original encoding | Logistic Regression | musicgen | 9.2% | 32.19% | 22.99% | 96.04% | 68.6% |
| Original encoding | Logistic Regression | suno | 9.55% | 8.41% | -1.14% | 97.1% | 95.8% |
| Original encoding | RBF-SVM | musicgen | 8.06% | 44.85% | 36.79% | 95.82% | 55.81% |
| Original encoding | RBF-SVM | suno | 7.16% | 5.91% | -1.25% | 98.52% | 97.5% |

EER 격차는 `Unseen EER - Seen EER`이다. 양수이면 처음 보는 생성기에서
성능이 나빠졌다는 뜻이고, 0에 가까우면 학습에 없던 생성기에도 비교적 잘
일반화했다는 뜻이다.

## 관찰

- 가장 큰 성능 저하: **Original encoding / RBF-SVM / musicgen**, +36.79%p
- 가장 작은 변화: **Original encoding / RBF-SVM / suno**, -1.25%p

생성기를 제외했는데 EER이 낮아지는 경우도 있을 수 있다. 이는 탐지기가 더
좋아졌다는 확정 증거가 아니라, Test REAL 44곡과 생성기별 FAKE 40여 곡으로
표본이 작고 학습 구성과 threshold가 함께 달라졌기 때문에 생기는 변동으로
해석해야 한다.

## 한계와 다음 단계

두 생성기만으로 모든 미래 생성기를 대표할 수 없고, 원본 코덱 흔적도 완전히
제거되지는 않는다. 다음으로 같은 데이터 분할에 Log-Mel CNN을 학습해 수작업
특징 모델과 딥러닝 모델의 표준·미지 생성기 성능을 비교한다.
