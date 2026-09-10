# 2단계 10초 세그먼트 구성 결과

## 이 단계에서 한 일

원본 음악 파일을 새로 자르거나 복사하지 않고, 각 트랙에서 읽을 10초 구간의
시작 위치를 `segment_manifest.csv`에 기록했다. 따라서 추가 저장공간은 거의
사용하지 않으며 원본 파일도 변경하지 않았다.

## 구간 선택 규칙

- 30초 이상: 앞, 중간, 끝에서 3개
- 20초 이상 30초 미만: 앞, 끝에서 2개
- 10초 이상 20초 미만: 가운데에서 1개
- 약 30초인 FMA는 MP3 인코딩 오차 0.02초를 허용하여 3개로 처리

구간 길이 계산에는 manifest의 반올림 값이 아니라 `ffprobe`가 측정한 실제
오디오 길이를 사용했다.

## 생성 결과

| split | label | tracks | unique_groups | segments | segment_hours |
| --- | --- | --- | --- | --- | --- |
| train | REAL | 207 | 207 | 621 | 1.725 |
| train | FAKE | 2237 | 207 | 6299 | 17.4972 |
| validation | REAL | 45 | 45 | 135 | 0.375 |
| validation | FAKE | 482 | 45 | 1355 | 3.7639 |
| test | REAL | 44 | 44 | 132 | 0.3667 |
| test | FAKE | 443 | 44 | 1242 | 3.45 |

- 전체: 9,784개
- REAL: 888개
- FAKE: 8,896개
- 10초 구간을 모두 이어 들었을 때: 27.18시간

## 트랙 길이에 따른 구간 수

| 트랙당_구간수 | 트랙수 |
| --- | --- |
| 1 | 2 |
| 2 | 586 |
| 3 | 2870 |

## 품질 검사

| check | value | passed |
| --- | --- | --- |
| total_segments | 9784 | True |
| duplicate_segment_ids | 0 | True |
| tracks_without_segments | 0 | True |
| tracks_outside_1_to_3_segments | 0 | True |
| segment_boundary_errors | 0 | True |
| groups_in_multiple_splits | 0 | True |
| max_overlap_seconds | 0.005715 | True |

약 30초인 FMA 일부는 실제 길이가 수 ms 짧아 앞·중간·끝 구간 사이에 최대
0.005715초의 미세한 겹침이 있다. 이는 허용치 0.01초보다 작다.

## CSV 한 행의 의미

예를 들어 `start_sec=20`, `duration_sec=10`이면 학습할 때 해당 음악 파일의
20초 지점부터 30초 지점까지 읽는다는 뜻이다. `label`은 REAL 또는 FAKE이고,
`split`은 공부용(train), 모의고사용(validation), 최종 시험용(test)을 뜻한다.

## 다음 단계

REAL과 FAKE 대표 구간을 실제로 읽어 모두 mono·24kHz로 변환할 수 있는지
검사하고, Log-Mel 스펙트로그램을 그려 눈으로 비교한다.
