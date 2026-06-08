# Ninja Gesture Recognition

웹캠에서 양손의 랜드마크를 추출하고, 학습된 모델로 제스처를 분류해 시각 효과와
연결한 팀 프로젝트입니다.

> 부트캠프 팀 프로젝트에서 제가 담당한 데이터 정제, 학습 최적화, 안정화 작업을
> 다시 검토해 포트폴리오용으로 정리한 저장소입니다.

## 동작 화면

![치도리 제스처 인식 데모](media/chidori-demo.gif)

양손의 랜드마크를 실시간으로 추적하고, 인식된 제스처 조합에 따라 치도리 효과를
실행합니다.

## 담당 역할

- 수집 데이터 정제와 잘못된 샘플 제거
- 학습 입력 형식 통일 및 데이터셋 구성
- 모델 학습과 조건별 결과 비교
- 실시간 추론 과정의 오인식과 실행 흔들림 안정화
- 제스처 인식 결과와 콘텐츠 실행 흐름 검증

## 주요 기술

- Python
- MediaPipe Hand Landmarker
- PyTorch
- OpenCV

## 프로젝트 구조

```text
.
|-- media/chidori-demo.gif  # 제스처 인식 동작 화면
|-- song/data_collect.py    # 데이터 수집
|-- song/train.py           # 모델 학습
|-- song/mp_test.py         # 랜드마크 및 추론 테스트
|-- song/cam.py             # 실시간 카메라 실행
|-- song/models/model.py    # 모델 정의
`-- song/hand_landmarker.task
```
