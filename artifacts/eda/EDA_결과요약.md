# 1단계 메타데이터 EDA 결과

## 이 단계에서 확인한 것

모델을 학습하기 전에 데이터의 수량, 길이, 장르, 생성기, 파일 형식과
Train/Validation/Test 누수 여부를 확인했다. 이 단계에서는 실제 음악의
내용을 분석하지 않고 manifest와 ffprobe 검증 결과만 사용했다.

## 데이터 요약

| label | tracks | unique_groups | duration_hours | mean_duration_sec | median_duration_sec | min_duration_sec | max_duration_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| REAL | 296 | 296 | 2.47 | 30.0 | 29.99 | 29.99 | 30.01 |
| FAKE | 3162 | 296 | 106.67 | 121.44 | 130.86 | 17.96 | 479.96 |

- 전체 트랙: 3,458곡
- REAL: 296곡
- FAKE: 3,162곡
- FAKE/REAL 트랙 수 비율: 10.68:1
- 원곡 그룹: 296개

## 품질 점검

| check | value | passed |
| --- | --- | --- |
| total_tracks | 3458 | True |
| duplicate_sample_ids | 0 | True |
| duplicate_audio_paths | 0 | True |
| missing_files | 0 | True |
| undecodable_files | 0 | True |
| shorter_than_10s | 0 | True |
| groups_in_multiple_splits | 0 | True |

모든 파일이 존재하고 디코딩 가능하며 10초 이상이다. 같은 원곡 그룹이
여러 split에 들어간 경우가 없으므로 현재 그룹 분할은 모델 평가에 사용할
수 있다.

## 가장 중요한 발견

1. **클래스 불균형**: FAKE가 REAL보다 10.68배 많다.
   학습 때 REAL/FAKE 균형 sampling 또는 sample weight가 필요하다.
2. **파일 형식 차이**: REAL은 mp3 296곡, FAKE는 mp3 2869곡, wav 293곡이다.
3. **MusicGen의 고유 형식**: MusicGen은 293곡, wav, 32000Hz, 1채널이다.
   모델이 음악의 생성 흔적 대신 WAV/MP3, 샘플링레이트, 채널 차이를
   지름길로 사용할 수 있다.
4. **길이 차이**: REAL은 거의 30초인 반면 FAKE는 생성기마다 길이가 다르다.
   트랙당 최대 3개의 10초 구간만 사용해야 긴 FAKE가 학습을 지배하지 않는다.

## 모델링 전에 확정할 처리

- 모든 오디오를 mono, 24,000Hz로 읽는다.
- 각 트랙에서 최대 3개의 10초 구간만 사용한다.
- Train 배치는 REAL/FAKE가 1:1에 가깝도록 구성한다.
- MusicGen 미지 생성기 실험은 원본 조건과 공통 MP3 변환 조건을 모두 보고한다.
- 코덱 증강은 REAL과 FAKE에 같은 확률과 설정으로 적용한다.

## 다음 단계

`track_manifest.csv`를 바탕으로 실제 학습 위치를 기록하는
`segment_manifest.csv`를 만든다. 원본 음원을 물리적으로 잘라 복사하지 않고
각 10초 구간의 시작 시각만 CSV에 저장한다.

## 생성된 그림

- `01_class_counts.png`: REAL/FAKE 곡 수
- `02_generator_counts.png`: AI 생성기별 곡 수
- `03_genre_class_counts.png`: 장르별 REAL/FAKE 분포
- `04_duration_by_class.png`: 클래스별 곡 길이
- `05_sample_rate_by_class.png`: 샘플링레이트 차이
- `06_split_class_counts.png`: split별 곡 수
