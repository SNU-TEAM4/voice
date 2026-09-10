# 3단계 오디오 입력 및 스펙트로그램 결과

## 이 단계에서 한 일

앞 단계의 9,784개 구간 중 Train/Validation/Test 각각에서 인간 음악과
12개 AI 생성기 음악을 하나씩 뽑아 총 39개를 실제로 디코딩했다. MP3와 WAV,
16kHz부터 48kHz까지 서로 다른 원본이 모두 같은 모델 입력으로 변환되는지
검사했다.

## 모델에 들어가는 공통 모양

- 길이: 10초
- 채널: mono 1채널
- 샘플링레이트: 24,000Hz
- 숫자 개수: 240,000개
- 파형 자료형: float32

오디오의 원래 형식이 달라도 모델에는 항상 같은 모양의 숫자 배열이 들어간다.

## 입력 파이프라인 품질 검사

| check | value | passed |
| --- | --- | --- |
| tested_profiles | 39 | True |
| decode_failures | 0 | True |
| wrong_output_length | 0 | True |
| non_finite_waveforms | 0 | True |
| silent_waveforms | 0 | True |
| total_padding_samples | 0 | True |

39개 대표 프로필이 모두 디코딩되었고, 길이 부족으로 0을 덧붙인 구간도 없다.
앞 단계에서 전체 9,784개 구간의 파일 존재·디코딩 가능 여부·시작과 끝 경계를
검사했으므로 이 입력 방식을 전체 데이터에 적용할 수 있다.

## 같은 원곡 계열 비교

- 그룹: `grp_040350d9b464`
- 원곡 표기: `KOMFORT - voyageurs`
- 비교 대상: Human REAL, MusicGen FAKE, Suno FAKE
- 그림: `matched_waveform_logmel.png`
- 직접 들을 파일: `previews` 폴더의 WAV 3개

파형은 시간에 따른 소리의 세기를 보여주고, Log-Mel 스펙트로그램은 시간에
따라 저음부터 고음까지 에너지가 어떻게 분포하는지 색으로 보여준다. 그림에서
차이가 보여도 그것만으로 AI 생성의 일반적 특징이라고 결론 내리면 안 된다.
모델 학습과 전체 테스트 결과로 확인해야 한다.

## 다음 단계

각 10초 구간에서 MFCC와 스펙트럴 특징을 숫자로 추출하고, 첫 기계학습
기준선인 Logistic Regression과 SVM을 학습한다. Train으로 공부하고,
Validation으로 설정을 고른 뒤, Test는 마지막 평가에만 사용한다.
