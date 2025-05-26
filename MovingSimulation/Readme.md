# Robot Kinematics Simulation 🤖

**로봇 운동학 시뮬레이션 프로그램** - 교육용 로봇 운동학 학습 및 실험 도구

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com/your-username/robot-kinematics-simulation)

## 📋 프로젝트 개요

이 프로젝트는 **로봇 운동학(Robot Kinematics)**을 시각적으로 학습하고 실험할 수 있는 종합적인 시뮬레이션 도구입니다. 1DOF부터 6DOF까지 다양한 로봇 구조를 지원하며, DH(Denavit-Hartenberg) 파라미터를 기반으로 한 정기구학과 역기구학 계산을 제공합니다.

### 🎯 개발 목적

- **교육적 도구**: 로봇 공학 학습자들이 운동학 이론을 직관적으로 이해할 수 있도록 돕습니다
- **실험 플랫폼**: 다양한 로봇 구조와 파라미터를 실시간으로 변경하며 결과를 확인할 수 있습니다
- **시각화**: 3D 환경에서 로봇의 움직임과 작업공간을 실시간으로 관찰할 수 있습니다

## ✨ 주요 기능

### 🔄 운동학 계산
- **정기구학(Forward Kinematics)**: 관절 각도로부터 End-Effector 위치 계산
- **역기구학(Inverse Kinematics)**: 목표 위치로부터 관절 각도 계산
- **다중 해 지원**: 역기구학의 여러 해를 동시에 계산하고 선택 가능

### 🎛️ DH 파라미터 관리
- **실시간 편집**: GUI를 통한 직관적인 DH 파라미터 조정
- **프리셋 로봇**: 1DOF~6DOF 표준 로봇 구조 제공
- **YAML 지원**: 로봇 구성을 파일로 저장/로드

### 📊 3D 시각화
- **실시간 렌더링**: 로봇 링크, 관절, 좌표계 표시
- **궤적 추적**: End-Effector의 이동 경로 시각화
- **작업공간 분석**: 로봇의 도달 가능 영역 표시

### 🎯 시뮬레이션 기능
- **목표 지향적 이동**: 현재 위치에서 목표 위치까지의 최적 경로 계산
- **정밀도 추적**: mm 단위의 위치 오차 실시간 모니터링
- **충돌 검사**: 관절 제한 및 바닥 충돌 검증

## 📁 프로젝트 구조

```
robot-kinematics-simulation/
├── main.py                     # 메인 GUI 애플리케이션
├── robot_kinematics.py         # 운동학 계산 엔진
├── dh_parameters.py           # DH 파라미터 관리
├── trajectory_planner.py      # 궤적 계획 모듈
├── visualization.py           # 3D 시각화 도구
├── utils.py                   # 유틸리티 함수들
├── yaml/                      # 로봇 프리셋 파일들
│   ├── 1dof_revolute.yaml
│   ├── 2dof_planner.yaml
│   ├── 3dof_anthropomorphic.yaml
│   ├── 4dof_scara.yaml
│   ├── 5dof_articulated.yaml
│   └── 6dof_industrial.yaml
├── results/                   # 시뮬레이션 결과 저장 폴더
└── README.md
```

### 핵심 모듈 설명

#### 🔧 main.py
메인 GUI 애플리케이션으로, 4개의 핵심 문제를 해결한 완성된 버전입니다:
- ✅ 상태창 텍스트 줄간격 문제 완전 해결
- ✅ Auto Update 코드 완전 제거 (직관적인 Calculate 버튼으로 대체)
- ✅ 목표점 위치와 시뮬레이션 정확성 추적 시스템 구축
- ✅ 시뮬레이션 초기화 및 시각적 동작 일관성 보장

#### 🧮 robot_kinematics.py
로봇 운동학 계산의 핵심 엔진입니다:
- DH 변환 행렬 계산
- 정기구학 및 역기구학 해법
- 자코비언 행렬 계산
- 특이점 검출 및 회피

#### 📐 dh_parameters.py
DH 파라미터 관리 시스템입니다:
- 표준 로봇 구조의 DH 파라미터 데이터베이스
- YAML 파일 저장/로드 기능
- 파라미터 유효성 검증

#### 🎨 visualization.py
3D 시각화 도구입니다:
- 실시간 로봇 렌더링
- 링크, 관절, 좌표계 표시
- 궤적 및 작업공간 시각화

## 🎮 사용법

### 기본 사용법

1. **프로그램 실행** 후 좌측 제어판에서 로봇 설정
2. **DOF 선택**: 1~6DOF 중 원하는 자유도 선택
3. **모드 선택**: Forward 또는 Inverse 모드 선택
4. **파라미터 조정**: DH 파라미터를 실시간으로 변경
5. **시뮬레이션 실행**: "Run Simulation" 버튼으로 목표 위치로 이동

### Forward Kinematics 모드

```
1. DOF 설정 (예: 3DOF)
2. DH 파라미터 입력 또는 프리셋 로드
3. 관절 각도 조정 (슬라이더 또는 직접 입력)
4. 실시간으로 End-Effector 위치 확인
```

### Inverse Kinematics 모드

```
1. DOF 설정 및 DH 파라미터 입력
2. 목표 위치 및 자세 입력
3. "Calculate" 버튼으로 역기구학 해 계산
4. 여러 해 중 원하는 해 선택
5. "Run Simulation"으로 이동 시뮬레이션 실행
```

### 부가적 기능

- **프리셋 로봇 로드**: "Load Preset" 버튼으로 표준 로봇 구조 불러오기
- **YAML 저장/로드**: 사용자 정의 로봇 구성 저장
- **작업공간 분석**: 로봇의 도달 가능 영역 계산
- **결과 저장**: 시뮬레이션 결과를 CSV 파일로 저장

## 🔬 기술적 세부사항

### 운동학 알고리즘

**정기구학**: DH 변환 행렬의 연속 곱셈
```python
T_total = T_0 × T_1 × T_2 × ... × T_n
```

**역기구학**: 수치해석적 방법 (Newton-Raphson, Levenberg-Marquardt)
- 다중 초기값을 사용한 해 탐색
- 관절 제한 및 특이점 회피
- 해의 유효성 검증

### DH 파라미터 규약

표준 DH 파라미터 (Craig 규약) 사용:
- **a**: 링크 길이 (cm)
- **α**: 링크 트위스트 (도)
- **d**: 링크 오프셋 (cm)
- **θ**: 관절 각도 (도)

### 정밀도 관리

- **위치 허용 오차**: 5mm
- **각도 허용 오차**: 0.5도
- **실시간 오차 모니터링**: mm 단위 정확도 표시

## 📚 교육적 활용

### 학습 목표

이 프로그램을 통해 다음을 학습할 수 있습니다:

1. **DH 파라미터의 의미**: 각 파라미터가 로봇 구조에 미치는 영향 이해
2. **운동학 이론**: 정기구학과 역기구학의 차이점과 응용
3. **특이점 개념**: 로봇이 움직일 수 없는 특수한 자세 탐험
4. **작업공간**: 로봇의 도달 가능 영역과 제한사항 분석

### 실습 예제

#### 예제 1: 2DOF 평면 로봇
```yaml
# 2dof_planar.yaml
links:
  - a: 40.0, alpha: 0.0, d: 0.0, theta: 0.0  # 첫 번째 링크
  - a: 30.0, alpha: 0.0, d: 0.0, theta: 0.0  # 두 번째 링크
```

#### 예제 2: 6DOF 산업용 로봇
```yaml
# 6dof_industrial.yaml
links:
  - a: 0.0, alpha: 90.0, d: 15.0, theta: 0.0   # 베이스 회전
  - a: 25.0, alpha: 0.0, d: 0.0, theta: 0.0    # 어깨
  - a: 3.0, alpha: 90.0, d: 0.0, theta: 0.0    # 팔꿈치
  - a: 0.0, alpha: -90.0, d: 22.0, theta: 0.0  # 손목 요
  - a: 0.0, alpha: 90.0, d: 0.0, theta: 0.0    # 손목 피치
  - a: 0.0, alpha: 0.0, d: 6.0, theta: 0.0     # 손목 롤
```

### 성능 최적화

- **실시간 계산**: GUI 응답성을 위한 비동기 처리
- **메모리 효율성**: 대용량 작업공간 데이터의 효율적 관리
- **수치 안정성**: 특이점 근처에서의 안정적인 계산

## 📄 라이선스

이 프로젝트는 MIT License 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🏆 감사의 말

이 프로젝트는 로봇 공학 교육의 접근성을 높이고, 학습자들이 이론과 실제를 연결할 수 있도록 돕기 위해 개발되었습니다. 로봇 운동학의 복잡한 개념들을 시각적이고 직관적으로 이해할 수 있는 도구로 활용되기를 바랍니다.

---

**Happy Robot Programming! 🤖✨**

*"복잡한 수학을 간단한 시각화로, 추상적 이론을 실제 실험으로"*