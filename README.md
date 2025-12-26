# x_ray_sty

흉부 X_ray 이미지를 이용해 정상 과 폐렴을 분류하는 딥러닝 개인 공부용 프로젝트입니다.
본 프로젝트는 모델 성능 극대화 보다, 불균형 데이터에서의 안정적 학습
파이프라인 구축과 실험 기록에 초점을 둡니다.

project Goals
-흉부 X_ray 데이터로 이진 분류 파이프 라인 구축
-클래스 불균형 상황에서의 안정적인 학습 확인
-딥러닝 모델 학습 과정애서 발생하는 과적합 문제 분석
-실험 과정을 Git/Github 커밋 단위 기록 이로 인한 Git/Github 사용법 숙지

### !!!! 본 프로젝트는 의료 진단 목적이 아닌 학습/연구 목적의 개인 프로젝트입니다.

## Dataset Overview
NORMAL: 약 1,341장
PNEUMONIA: 약 3,875장
클래스 비율: 약 1 : 3

데이터는 다음과 같은 구조를 가집니다.

```text
data/
 ├─ train/
 │   ├─ NORMAL/
 │   └─ PNEUMONIA/
 ├─ val/
 │   ├─ NORMAL/
 │   └─ PNEUMONIA/
 └─ test/
     ├─ NORMAL/
     └─ PNEUMONIA/
```

데이터 파일은 용량 및 혹시모를 라이선스 문제로 Github에 포함하지 않습니다.

## Project Structure
```text
chest_xray/
├─ data/                # 로컬 데이터 (Git ignore)
├─ src/
│   ├─ dataset.py       # Dataset 정의
│   ├─ model.py         # 모델 정의
│   └─ train.py         # 학습 스크립트
├─ README.md
├─ requirements.txt
└─ .gitignore
```



Approach
본 프로젝트는 다음과 같은 단계로 진행됩니다.
1. Baseline 파이프라인 구축
    -imageFolder 기반 Daraset
    -ResNet18 기반 간단한 분류 모델
2. 헉습 안정성 검증
    -Train/Validation loss 추이 확인
    -Epoch 증가 시 과적합 여부 관찰
3. 클래스 불균형 대응
    -Weighted Loss 적용
4. (추후) 모델 개선 및 시각화
    -Grad-CAM 기반 모델 해석

## Tech Stack
    -Python
    -PyTorch
    -Torchvision
    -Git / GitHub
    -GPT
    -OPENAI
Notes
-모든 실험은 작은 변경 단위로 커밋하며 진행합니다.
-성능 수치보다 학습 과정 자체를 중요하게 다룹니다.
======================================
### Authour - Lee_Seung_Jae 
======================================