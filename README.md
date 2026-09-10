# Echoes 완전 AI 생성 음악 탐지

현재까지의 연구설계, 수업 개념, 분석 방법과 실험결과를 한 문서로 합친
보고서는 `artifacts/통합_연구설계_및_실험결과.md`에서 확인할 수 있다.

원본 데이터셋과 음원은 Git에 포함하지 않는다. 기본적으로 프로젝트 아래의
`data/Echoes`, `data/fma_metadata`를 읽으며, 데이터가 다른 위치에 있다면
환경변수로 경로를 지정한다.

```bash
export ECHOES_ROOT="/path/to/Echoes"
export FMA_METADATA_ROOT="/path/to/fma_metadata"
```

연구에서 사용할 클래스는 다음과 같다.

- REAL: FMA의 인간 제작 30초 음원
- FAKE: Echoes의 `TTA`(Text-to-Audio) 음원
- 제외: Echoes의 `ATA`(Audio-to-Audio) 음원

## 1. REAL 매칭표 생성

```bash
python3 scripts/build_fma_mapping.py
```

Echoes manifest에는 FMA track ID가 없어서 `원곡 제목 - 아티스트`를 기준으로
FMA metadata와 정확 매칭한다. 후보가 여러 개인 경우 Echoes 장르 일치,
논문에 명시된 라이선스(CC0/CC-BY/Public Domain), 낮은 track ID 순으로
결정한다. 모든 후보는 `artifacts/ambiguous_matches.csv`에 보존한다.
FMA에서 디코딩되지 않는 것으로 검증된 `148786`, `148788`은 같은
제목·아티스트의 정상 후보 `148802`, `148804`로 대체한다.

## 2. 필요한 REAL 음원만 다운로드

먼저 한 곡으로 동작을 확인한다.

```bash
python3 scripts/download_fma_real.py --limit 1 --workers 1
```

정상 작동하면 전체 296곡을 받는다.

```bash
python3 scripts/download_fma_real.py
```

이 스크립트는 약 93 GiB인 `fma_large.zip` 전체를 받지 않는다. HTTP byte
range를 이용하여 ZIP 중앙 디렉터리와 선택된 296개 MP3 멤버만 가져온다.
완료된 음원은 `data/fma_real`에 저장된다.

## 3. REAL 음원 검증

```bash
python3 scripts/verify_fma_real.py
```

모든 MP3를 `ffprobe`로 열어 코덱, 샘플레이트, 길이를 검사한다. 결과는
`artifacts/fma_real_validation.csv`에 기록하며, 10초 미만 파일은 학습에
사용할 수 없는 항목으로 표시한다.

## 4. REAL/TTA 통합 및 그룹 분할

```bash
python3 scripts/build_track_manifest.py
```

FMA REAL과 Echoes TTA를 합치고, `original_audio`가 같은 REAL·FAKE를 하나의
그룹으로 묶는다. 장르별 비율을 유지하면서 그룹을 train 70%, validation
15%, test 15%로 나눈다. 오디오를 10초로 자르기 전에 이 분할을 먼저
적용하여 같은 원곡 계열이 여러 split에 섞이는 데이터 누수를 방지한다.

주요 결과 파일:

- `artifacts/track_manifest.csv`: REAL/TTA 전체 트랙 메타데이터
- `artifacts/split_groups.csv`: 원곡 그룹별 split
- `artifacts/split_summary.csv`: split 및 클래스 분포
- `artifacts/generator_split_summary.csv`: 생성기별 분포
- `artifacts/excluded_tta_duplicates.csv`: 충돌 경로로 제외한 TTA 행

Echoes manifest의 MusicGen 행 3개가 서로 다른 원곡을 가리키면서 실제로는
같은 `_musicgen_TTA_001.wav` 경로를 사용한다. 어느 원곡의 파일인지 확인할
수 없으므로 세 행과 해당 파일을 전부 분석에서 제외한다.

## 5. 통합 오디오 검증

```bash
python3 scripts/verify_track_audio.py
```

통합 manifest의 모든 REAL/TTA 파일을 `ffprobe`로 검사하고 결과를
`artifacts/track_audio_validation.csv`에 저장한다.

## 6. 첫 모델링 단계: 메타데이터 EDA

모델 학습 전에 곡 수, 생성기, 장르, 길이, 파일 형식, 샘플링레이트,
채널과 데이터 누수를 확인한다. 현재 노트북에 설치된 Anaconda 환경에서는
다음 명령으로 바로 실행할 수 있다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/01_metadata_eda.py
```

결과는 `artifacts/eda`에 저장된다.

- `EDA_결과요약.md`: 비전공자를 위한 핵심 해석
- `01_label_overview.csv` 등: 발표와 검증에 사용할 표
- `01_class_counts.png` 등: 발표에 바로 사용할 그래프

다른 컴퓨터에서 패키지가 없다면 프로젝트 전용 환경에서
`requirements-eda.txt`를 먼저 설치한다.

## 7. 두 번째 모델링 단계: 10초 구간 위치 만들기

원본 오디오 파일을 직접 자르거나 복사하지 않고, 각 트랙에서 학습에 사용할
최대 3개의 10초 구간 위치를 CSV로 만든다.

```bash
python3 scripts/02_build_segment_manifest.py
```

주요 결과 파일:

- `artifacts/segment_manifest.csv`: 모든 10초 구간의 파일 경로와 시작 위치
- `artifacts/segment_summary.csv`: split·클래스별 구간 수
- `artifacts/segment_generator_summary.csv`: 생성기별 구간 수
- `artifacts/segment_quality_checks.csv`: 경계·중복·누수 검사
- `artifacts/SEGMENT_결과요약.md`: 비전공자를 위한 결과 해석

## 8. 세 번째 모델링 단계: 실제 오디오 입력과 Log-Mel 확인

각 split에서 인간과 12개 생성기 대표 구간을 실제로 읽어, 서로 다른 MP3/WAV
파일이 모두 10초·mono·24kHz 입력으로 변환되는지 확인한다. 같은 원곡 그룹의
REAL·MusicGen·Suno 파형과 Log-Mel 스펙트로그램도 만든다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/03_audio_sanity_and_spectrogram.py
```

결과는 `artifacts/audio_eda`에 저장된다.

- `AUDIO_EDA_결과요약.md`: 검사 결과의 쉬운 해석
- `audio_sanity_checks.csv`: 실제로 디코딩한 39개 프로필
- `audio_pipeline_quality.csv`: 길이·유효값·무음·패딩 검사
- `matched_waveform_logmel.png`: 같은 원곡 계열의 파형·Log-Mel 비교
- `previews/*.wav`: 직접 들을 수 있는 10초 예시 3개

## 9. 네 번째 모델링 단계: 수작업 특징 + 기계학습 기준선

먼저 60개 구간으로 특징 추출기가 작동하는지 시험한다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/04_extract_handcrafted_features.py \
  --limit 60 --output /tmp/echoes_features_smoke.npz
```

정상 작동하면 전체 9,784개 구간에서 266개 음향 특징을 추출한다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/04_extract_handcrafted_features.py
```

저장된 특징으로 Logistic Regression과 RBF-SVM을 학습한다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/05_train_ml_baselines.py
```

특징은 `artifacts/features`, 모델과 성능표는 `artifacts/ml_baseline`에
저장된다. 모델 설정과 threshold는 Validation으로 선택하며 Test 데이터는
마지막 평가에만 사용한다.

## 10. 다섯 번째 모델링 단계: 코덱 통제 실험

모든 REAL·FAKE 구간을 동일한 MP3 64kbps로 인코딩·디코딩한 별도 특징을
만든다. 기존 원본 조건 특징은 덮어쓰지 않는다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/04_extract_handcrafted_features.py \
  --codec-control mp3 --mp3-bitrate-kbps 64 \
  --output artifacts/features/handcrafted_features_mp3_64kbps.npz
```

같은 모델 선택 절차로 통제 조건 모델을 학습한다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/05_train_ml_baselines.py \
  --features artifacts/features/handcrafted_features_mp3_64kbps.npz \
  --output-dir artifacts/ml_codec_control_mp3_64kbps
```

마지막으로 파일 정보만 쓰는 metadata baseline과 원본·통제 조건을 비교한다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/06_compare_codec_control.py
```

최종 비교표와 보고서는 `artifacts/codec_control`에 저장된다.

## 11. 여섯 번째 모델링 단계: 미지 생성기 일반화

MusicGen과 Suno를 각각 Train·Validation에서 완전히 제외한 모델을 만들고,
같은 Test 하위집합에서 학습에 포함했던 기존 모델과 비교한다. 원본 인코딩과
공통 MP3 64kbps 조건을 모두 평가한다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/07_unseen_generator_experiment.py
```

모델, 예측, 일반화 격차, 그래프와 쉬운 결과 보고서는
`artifacts/unseen_generator`에 저장된다.

## 12. 일곱 번째 모델링 단계: Log-Mel CNN

프로젝트 전용 환경을 만들고 PyTorch를 설치한다. 현재 프로젝트에서는
`.venv`에 설치되어 있다.

```bash
/opt/anaconda3/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install torch
```

모든 코덱 통제 구간의 128-band Log-Mel 캐시를 만든다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  /opt/anaconda3/bin/python scripts/08_extract_logmel_cache.py
```

CNN 순전파·역전파가 가능한지 먼저 확인한다.

```bash
.venv/bin/python scripts/09_train_logmel_cnn.py --smoke-test
```

정상 작동하면 전체 CNN을 학습한다.

```bash
MPLCONFIGDIR=.mpl-cache XDG_CACHE_HOME=.cache \
  .venv/bin/python scripts/09_train_logmel_cnn.py --device auto
```

Log-Mel 캐시는 `artifacts/features`, CNN 모델·지표·그래프·보고서는
`artifacts/cnn_mp3_64kbps`에 저장된다.

현재 seed 42 최초 실행은 Validation Track EER 기준으로 5 epoch 모델이
선택되었고, 공통 MP3 64kbps Test에서 다음 곡 단위 결과를 얻었다.

| 모델 | EER (낮을수록 좋음) | ROC-AUC (높을수록 좋음) | Macro-F1 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 9.74% | 97.67% | 84.52% |
| RBF-SVM | 9.29% | 95.84% | 80.62% |
| Log-Mel CNN | 10.99% | 96.90% | 80.89% |

CNN이 이번 최초 실행에서 기계학습 기준선을 넘지는 못했다. 이는 실패가 아니라
같은 데이터·split·코덱 조건에서 모델 복잡도에 따른 차이를 비교할 수 있는
정상적인 연구 결과다. 다만 단일 seed 결과이므로 최종 결론 전 반복 실험이
필요하다.
