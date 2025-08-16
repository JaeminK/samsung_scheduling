# AutoTP: Automatic Tensor & Pipeline Parallelism

## 프로젝트 개요

AutoTP는 단일 GPU HuggingFace 모델을 자동으로 Tensor Parallel과 Pipeline Parallel 모델로 변환하는 프로젝트입니다. 이 프로젝트는 분산 학습의 핵심 개념들을 구현하는 4가지 과제로 구성되어 있습니다.

## 🎯 구현 과제

이 프로젝트는 다음과 같은 4가지 핵심 과제로 구성되어 있습니다:

### Problem 1: Tensor Parallelism 구현
- **ColumnParallelLinear**: 입력을 여러 GPU로 분할하여 병렬 처리
- **RowParallelLinear**: 각 GPU가 출력의 일부를 계산하고 all_reduce로 합산

### Problem 2: Pipeline Parallelism 구현
- **PipelineParallelTransformerLayer**: 스테이지 간 데이터 전송을 위한 send/recv 연산
- 스테이지의 첫 번째/마지막 레이어에서만 통신 수행

### Problem 3: Layer Parallelization 구현
- **Attention Layer**: Query/Key/Value는 Column Parallel, Output은 Row Parallel
- **MLP Layer**: Up projection은 Column Parallel, Down projection은 Row Parallel

### Problem 4: Pipeline Parallel Stage Construction 구현
- 각 레이어를 PipelineParallelTransformerLayer로 래핑
- 스테이지 내에서의 위치 정보 설정

## 📁 프로젝트 구조

```
AutoTP/
├── src/autotp/
│   ├── layer.py          # Problem 1, 2 구현 위치
│   ├── utils.py          # Problem 3, 4 구현 위치
│   └── solutions.md      # 모든 문제의 정답과 상세 설명
├── benchmarks/           # 테스트 스크립트
│   ├── test_single.sh    # 단일 GPU 테스트
│   ├── test_tp_dist.sh   # Tensor Parallel 분산 테스트 (torchrun)
│   ├── test_pp_dist.sh   # Pipeline Parallel 분산 테스트 (torchrun)
│   ├── run_tp_rank0.sh   # Tensor Parallel Rank 0 디버깅용
│   ├── run_tp_rank1.sh   # Tensor Parallel Rank 1 디버깅용
│   ├── run_pp_rank0.sh   # Pipeline Parallel Rank 0 디버깅용
│   └── run_pp_rank1.sh   # Pipeline Parallel Rank 1 디버깅용
├── main.py              # 메인 실행 파일
├── requirements.txt     # 의존성 패키지
└── README.md           # 이 파일
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/AutoTP.git
cd AutoTP

# 패키지 설치
pip install .

# 의존성 설치
pip install -r requirements.txt
```

### 2. 단일 GPU 테스트

```bash
cd benchmarks
python test_single.sh
```

### 3. Tensor Parallel 분산 테스트

```bash
cd benchmarks
python test_tp_dist.sh
```

### 4. Pipeline Parallel 분산 테스트

```bash
cd benchmarks
python test_pp_dist.sh
```

### 5. 디버깅용 테스트 (개별 GPU 실행)

**Tensor Parallel 디버깅:**
```bash
cd benchmarks
# Rank 0 실행 (pdb 디버깅 가능)
./run_tp_rank0.sh

# Rank 1 실행 (다른 터미널에서)
./run_tp_rank1.sh
```

**Pipeline Parallel 디버깅:**
```bash
cd benchmarks
# Rank 0 실행 (pdb 디버깅 가능)
./run_pp_rank0.sh

# Rank 1 실행 (다른 터미널에서)
./run_pp_rank1.sh
```

> **참고**: `run_*_rank*.sh` 스크립트는 pdb 디버깅을 위한 개별 GPU 실행용이고, `test_*_dist.sh` 스크립트는 `torchrun`을 사용한 자동 분산 실행용입니다.

## 📋 요구사항

- Python 3.8+
- PyTorch 2.7.0
- Transformers
- CUDA 지원 GPU (분산 학습용)

## 🎓 학습 목표

이 프로젝트를 통해 다음을 학습할 수 있습니다:

1. **Tensor Parallelism**: 모델의 가중치를 여러 GPU에 분산하여 메모리 효율성 향상
2. **Pipeline Parallelism**: 모델을 여러 스테이지로 분할하여 처리량 향상
3. **분산 통신**: GPU 간 효율적인 데이터 전송 및 동기화
4. **모델 병렬화**: Transformer 아키텍처의 효율적인 분산 처리


---

**참고**: 구현 과제의 정답은 `src/autotp/solutions.md` 파일을 참조하세요.

