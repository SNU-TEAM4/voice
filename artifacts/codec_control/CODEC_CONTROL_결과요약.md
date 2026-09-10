# 5단계 코덱 통제 실험 결과

## 실험 목적

기존 모델이 AI 생성 특징 대신 REAL과 FAKE의 MP3/WAV, 샘플링레이트,
비트레이트 차이를 이용했는지 확인했다. 기존 데이터와 분할은 변경하지 않았다.

## 비교 조건

- Original encoding: 원본을 mono·24kHz로 디코딩한 기존 조건
- Common MP3 64kbps: 모든 10초 REAL·FAKE 구간을 동일한 MP3 64kbps로
  인코딩한 뒤 다시 디코딩한 통제 조건
- Metadata-only: 음악 내용 없이 원본 형식, 코덱, 샘플링레이트, 채널,
  추정 비트레이트만 사용한 Logistic Regression

## 오디오 모델의 Test 곡 단위 결과

| 모델 | 조건 | EER | ROC-AUC | Macro-F1 | REAL 오탐률 | FAKE 놓침률 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Original encoding | 9.51% | 97.37% | 83.11% | 9.09% | 6.77% |
| RBF-SVM | Original encoding | 7.02% | 97.21% | 86.63% | 13.64% | 4.06% |
| Logistic Regression | Common MP3 64kbps | 9.74% | 97.67% | 84.52% | 13.64% | 5.19% |
| RBF-SVM | Common MP3 64kbps | 9.29% | 95.84% | 80.62% | 22.73% | 5.87% |

- Logistic Regression EER 변화: **+0.23%p**
- RBF-SVM EER 변화: **+2.27%p**

## Metadata-only 진단

- 선택된 C: 0.1
- Validation EER: 9.01%
- Test EER: 11.33%
- Test ROC-AUC: 98.77%

음악을 듣지 않고 파일 정보만 사용해도 일정 수준의 구분이 가능하므로 원본
데이터에 인코딩 편향이 존재한다.

## 해석

공통 MP3 처리 후 RBF-SVM의 EER은 상승하여 기존 성능 일부가 코덱 차이에
영향받았음을 보여준다. 반면 Logistic Regression의 EER 변화는 작고, 두
오디오 모델 모두 통제 조건에서 약 9%대 EER을 유지했다. 따라서 기존 성능을
전부 파일 형식 편향만으로 설명할 수는 없으며, 음향 내용에도 분류 가능한
신호가 남아 있다고 해석할 수 있다.

## 한계

공통 MP3 재압축은 최종 형식을 같게 하지만 원본 파일에 이미 남은 과거 압축
흔적을 완전히 제거하지는 못한다. 따라서 이 실험은 편향의 완전 제거가 아니라
코덱 차이를 약화했을 때 성능이 유지되는지를 보는 민감도 분석이다.

## 다음 단계

MusicGen과 Suno를 각각 Train에서 완전히 제외하고, Original과 Common MP3
조건에서 미지 생성기 성능을 비교한다. 이후 같은 분할로 CNN을 학습한다.
