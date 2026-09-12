# 전처리 파이프라인 단계별 결과

이 문서는 Echoes TTA(완전 AI 생성 음악)와 FMA REAL 음원을 학습 가능한 형태로 정리한 01–08 전처리 노트북의 실행 과정과 실제 산출물을 요약한다. 노트북은 [`notebooks/preprocessing`](../../notebooks/preprocessing), 공유 가능한 CSV 결과는 이 디렉터리의 `metadata`와 `eda`에 있다.

## 전체 결과 요약

| 항목 | 결과 |
|---|---:|
| 원곡 그룹 | 296 |
| REAL 트랙 | 296 |
| FAKE 트랙 | 3,162 |
| 전체 트랙 | 3,458 |
| AI 생성기 | 12 |
| 10초 세그먼트 | 10,077 |
| 수작업 음향 특징 | 세그먼트당 266개 |
| 원곡 그룹 split 누수 | 0건 |

## 01. Echoes/FMA 데이터 품질 확인

노트북: [`01_data_quality_check.ipynb`](../../notebooks/preprocessing/01_data_quality_check.ipynb)

- Echoes manifest의 구조, TTA 필터링 결과, 실제 파일 존재 여부를 확인했다.
- 같은 파일 경로가 여러 원곡에 연결되는 충돌을 찾아 학습 데이터에서 제외할 수 있게 정리했다.
- FMA metadata에서 이후 매칭에 필요한 제목, 아티스트, 장르, 라이선스, 트랙 ID를 확인했다.

이 단계의 목적은 모델 성능보다 먼저 잘못된 경로, 중복, 누락으로 인한 데이터 오염 가능성을 제거하는 것이다.

## 02. Echoes 원곡과 FMA REAL 매칭

노트북: [`02_fma_matching_check.ipynb`](../../notebooks/preprocessing/02_fma_matching_check.ipynb)

- `original_audio`의 `곡 제목 - 아티스트` 문자열을 정규화해 FMA metadata와 매칭했다.
- 최종 원곡 296개를 FMA 트랙 296개와 연결했다.
- 장르 분포는 Electronic 118개, Rock 111개, Pop 67개다.
- 결과: [`fma_real_mapping.csv`](metadata/fma_real_mapping.csv)

단일 후보는 자동 확정하고, 복수 후보는 장르 및 라이선스 정보를 함께 확인하도록 구성했다.

## 03. FMA REAL 음원 확보 및 검증

노트북: [`03_fma_audio_validation.ipynb`](../../notebooks/preprocessing/03_fma_audio_validation.ipynb)

보조 스크립트: [`src/02_download_fma_real.py`](../../src/02_download_fma_real.py), [`src/03_extract_fma_real_remote.py`](../../src/03_extract_fma_real_remote.py)

- 개별 FMA URL 다운로드 시험 5건은 모두 404로 실패했다. 이 실패 기록도 [`fma_real_download_report.csv`](metadata/fma_real_download_report.csv)에 남겼다.
- 대안으로 원격 `fma_large.zip`에서 필요한 멤버만 선택적으로 추출했다.
- 총 296개 중 291개를 새로 받았고, 기존 5개는 재사용했다. 최종 성공률은 296/296이다.
- 296개 모두 파일 존재, 최소 크기, MP3 디코딩 검사를 통과했다.
- 실제 길이는 약 29.989–30.015초 범위였다.
- 결과: [`fma_remote_extract_report.csv`](metadata/fma_remote_extract_report.csv), [`fma_real_audio_validation.csv`](metadata/fma_real_audio_validation.csv)

## 04. REAL/FAKE master manifest 구축

노트북: [`04_build_master_manifest.ipynb`](../../notebooks/preprocessing/04_build_master_manifest.ipynb)

- FMA REAL 296개와 정제된 Echoes FAKE 3,162개를 하나의 manifest로 합쳤다.
- 전체 3,458개 파일이 존재하는 것을 확인했다.
- 장르 분포는 Electronic 1,249개, Rock 1,238개, Pop 971개다.
- 296개 원곡 그룹과 12개 AI 생성기를 포함한다.
- 결과: [`master_manifest.csv`](metadata/master_manifest.csv)

## 05. 원곡 그룹 기반 train/validation/test 분할

노트북: [`05_group_split.ipynb`](../../notebooks/preprocessing/05_group_split.ipynb)

동일한 `original_audio`에 속한 REAL과 FAKE가 서로 다른 split에 들어가지 않도록 원곡 단위로 먼저 나눴다.

| split | 원곡 그룹 | 트랙 | REAL | FAKE |
|---|---:|---:|---:|---:|
| train | 207 | 2,392 | 207 | 2,185 |
| validation | 44 | 527 | 44 | 483 |
| test | 45 | 539 | 45 | 494 |

- 그룹 비율은 약 69.9% / 14.9% / 15.2%다.
- 원곡 그룹이 둘 이상의 split에 등장한 경우는 0건이다.
- 결과: [`original_audio_split.csv`](metadata/original_audio_split.csv), [`master_manifest_with_split.csv`](metadata/master_manifest_with_split.csv)

## 06. 탐색적 데이터 분석

노트북: [`06_eda.ipynb`](../../notebooks/preprocessing/06_eda.ipynb)

- split, label, 장르, 생성기별 분포를 그룹 수준과 트랙 수준에서 확인했다.
- FAKE 3,162개와 REAL 296개로 클래스 불균형이 존재한다.
- 생성기별 샘플은 149–300개 범위이며, 총 12개 생성기가 모든 split에 분포한다.
- 세부 집계표: [`eda`](eda)

따라서 모델 평가에서는 단순 accuracy뿐 아니라 클래스별 recall, F1, ROC-AUC 등 불균형에 강한 지표가 필요하다.

## 07. 10초 세그먼트 manifest 생성

노트북: [`07_build_segment_manifest.ipynb`](../../notebooks/preprocessing/07_build_segment_manifest.ipynb)

- 오디오 파일을 복제하지 않고 시작·종료 위치만 기록하는 방식으로 세그먼트를 정의했다.
- 총 10,077개 세그먼트를 만들었다: train 6,967개, validation 1,538개, test 1,572개.
- label별로 REAL 888개, FAKE 9,189개다.
- 30초보다 수십 ms 짧은 파일 때문에 463개 세그먼트에 padding 표시가 있으며 최대 padding은 0.06초다.
- 경계 오류와 원곡 그룹 split 누수는 발견되지 않았고 제외된 세그먼트도 없다.
- 결과: [`segment_manifest_10s.csv`](metadata/segment_manifest_10s.csv), [`segment_exclusions.csv`](metadata/segment_exclusions.csv)

## 08. 수작업 음향 특징 추출

노트북: [`08_extract_handcrafted_features.ipynb`](../../notebooks/preprocessing/08_extract_handcrafted_features.ipynb)

- 모든 오디오를 mono, 24 kHz, 10초로 통일해 읽는다.
- MFCC 40차원과 delta/delta-delta, RMS, ZCR, spectral centroid/bandwidth/rolloff/flatness/contrast, chroma, tonnetz의 통계량을 계산한다.
- 10,077개 세그먼트 각각에 대해 266개 특징을 추출했으며 특징값 결측은 0건이다.
- 결과 파일 `data/processed/features/handcrafted_features_10s.csv`는 약 52 MiB로, 원본 오디오 및 모델 바이너리와 함께 Git에서 제외했다. 노트북을 실행하면 동일 경로에 다시 생성된다.

## 재현 순서

프로젝트 루트에서 아래 순서대로 노트북을 실행한다.

```text
01_data_quality_check
→ 02_fma_matching_check
→ 03_fma_audio_validation
→ 04_build_master_manifest
→ 05_group_split
→ 06_eda
→ 07_build_segment_manifest
→ 08_extract_handcrafted_features
```

노트북은 프로젝트 루트의 `data/` 디렉터리를 기준으로 경로를 찾는다. 저장소에는 라이선스와 용량 문제로 원본 Echoes/FMA 오디오를 포함하지 않는다. `artifacts/preprocessing/metadata`는 실행 당시 결과를 검토할 수 있도록 보존한 스냅샷이며, 새 환경에서 재실행할 때는 노트북이 생성하는 `data/metadata`를 사용한다.

## 다음 단계

생성된 특징을 이용해 Logistic Regression과 RBF-SVM 기준선을 학습하고, validation에서 하이퍼파라미터와 threshold를 정한 뒤 test는 최종 평가에만 사용한다. 대상 저장소의 기존 `scripts/05_train_ml_baselines.py`와 후속 CNN 실험 흐름으로 연결할 수 있다.
