# M2 구현 지시서 — LIF 엔진 (torch, GPU)

총괄 세션에서 작성 (2026-09-13). **M1이 커밋되어 있어야 시작한다.** 없으면 멈추고 보고.
먼저 읽을 것: `CLAUDE.md`, `docs/m1-brief.md`, `docs/m1-report.md`, `flysim/graph/graph.py`.

의존성은 기존 그대로(torch cu124, numpy, pytest) + **`setuptools` 추가 승인 대기** (Triton import에 필요, "성능 설계" 절). 그 외 새 패키지가 필요하면 사용자에게 먼저 묻는다.

## 모델 (전부 "가정"으로 주석 표기)

전류 기반 LIF + 지수 시냅스. 상태 벡터 두 개(`v`, `i_syn`), float32, GPU 상주.

```
i_syn ← i_syn · exp(−dt/τ_syn) + g · (W_post @ s_prev)      # s_prev: 직전 스텝 스파이크 (0/1 float)
v     ← v + (dt/τ_m) · (v_rest − v) + (dt/τ_m) · R·(i_syn + i_ext)   # 불응기 중인 뉴런은 v_reset 고정
s     ← (v ≥ v_thresh) & ¬refractory
v[s]  ← v_reset ; refractory_until[s] ← t + t_ref
```

- `W_post @ s_prev`는 **수식상의 표기**다. 구현은 전체 SpMV가 아니라 **이벤트 구동 전파**로 한다
  (아래 "성능 설계" 절, 벤치마크 근거 포함). 이때 필요한 배열은 M1의 **전방 CSR(행=presynaptic)** 이다:
  발화한 pre 뉴런의 행을 훑어 post 뉴런 누산기에 더한다. 값은 부호 포함 시냅스 수 **int32**
  (원본이 정수이므로 손실 없음; 정수 누산이 결정론의 근거가 된다).
- 시냅스 지연: **없음** (1스텝 = dt). 가정으로 명시. 위상 잠금 모드에서 문제되면 M3에서 재검토.
- 노이즈: 기본 **없음**. 옵션으로 가우시안 전류 노이즈를 넣되 시드 고정 필수.
- 단위: v는 mV 스케일 그대로(v_rest −65, v_thresh −50, v_reset −65), i_syn는 임의 단위.
  `g`(시냅스 이득)가 사실상 유일한 자유 파라미터다. R=1로 두고 g에 흡수.

기본 파라미터 (문서화하고 `EngineParams` dataclass로):
`dt=1.0ms, τ_m=20ms, τ_syn=5ms, t_ref=2ms, v_rest=−65, v_reset=−65, v_thresh=−50, g=?`
g는 아래 스윕으로 정한다. dt=0.1ms 모드에서 같은 파라미터가 같은 동작을 내야 한다
(dt 의존 계수는 전부 `exp(−dt/τ)` 형태로 계산).

## 산출물

```
flysim/engine/
  params.py    # EngineParams dataclass (+ to_dict, 런 기록용)
  lif.py       # LIFEngine: __init__(graph, params, device), reset(seed), step(i_ext) -> spikes, run(...)
  recorder.py  # 스파이크 기록: GPU에서 (t, idx) 쌍을 청크로 모아 CPU로 옮김. 매 스텝 .cpu() 금지
  bench.py     # 성능 측정 스크립트
  sweep.py     # g 스윕 (완료 조건용)
tests/test_engine.py
docs/m2-report.md
```

### LIFEngine 요구사항
- `device="cuda"` 기본. CUDA 없으면 CPU 폴백하고 `warnings.warn`으로 느리다고 경고.
- `reset(seed: int)`: 상태 초기화 + `torch.manual_seed`. 결정론의 출발점.
- `step(i_ext: Tensor|None) -> Tensor[bool]`: 한 스텝. `i_ext`는 길이 n 또는 None.
- `run(n_steps, i_ext_fn=None, record=True)`: 루프. `i_ext_fn(t_step) -> Tensor` 콜백으로 입력 주입.
  M3의 sensory 어댑터가 이 콜백을 구현한다.
- 스파이크 기록은 `recorder.py`에 위임. 반환은 `(t_bin int32 array, neuron_idx int32 array)`.
- 스텝 내부에서 Python 스칼라 동기화(`.item()`, `if tensor:`) 금지. 전부 텐서 연산으로.

### 결정론 테스트 (`tests/test_engine.py`)
- 같은 시드·같은 입력으로 두 번 돌려서 스파이크 (t, idx) 배열이 **bit-identical**해야 한다.
- 다른 시드(노이즈 켰을 때)면 달라야 한다.
- 전파 누산은 int32 정수 atomic이므로 GPU에서도 bit-exact여야 한다. 아니면 **테스트를 완화하지 말고** 보고한다.
- 합성 그래프에서 Triton 전파 결과가 torch `index_add_` 참조 구현과 정확히 일치하는지도 테스트한다.
- NaN/Inf 없음, v가 [v_reset, v_thresh] 밖으로 안 나감.
- 작은 합성 그래프(뉴런 3개, A→B 흥분, A→C 억제)로 부호가 맞는지 확인.

### 성능 (`bench.py`, 결과는 `docs/m2-report.md`)
- 실제 그래프 전체(165,122 뉴런, 25.56M 엣지)로 측정. **서브그래프 금지.**
- `torch.cuda.synchronize()` 후 `time.perf_counter`, 워밍업 100스텝 제외, 10,000스텝 평균.
- 목표: **스텝당 100μs 미만** (dt=0.1ms 실시간). 설계는 아래 "성능 설계" 절을 따른다.
  그래도 안 되면 수치와 함께 보고. 그래프를 줄이지 않는다.
- 활동 수준별로 측정한다(스텝당 발화 뉴런 비율 0 / 0.1% / 1% / 5%). 이벤트 구동이라 비용이 활동에 비례한다.
- 기록기(recorder) 포함/제외를 분리 측정. 스텝당 GPU 메모리 사용량도 기록.

## 성능 설계 (2026-09-13 총괄 세션 벤치마크 후 확정)

같은 크기의 **합성** 그래프(165,122 뉴런, 25.56M 엣지, 무작위 연결)로 이 노트북의 RTX 4090 Laptop에서
잰 값이다. 스크립트: `docs/m2-prototype-step.py` (실제 그래프로 재측정하는 것이 `bench.py`의 일이다).

| 방식 | 스텝당 시간 | 판정 |
|---|---|---|
| `torch.mv(sparse_csr, s)` 전체 SpMV (float32 + int64 인덱스) | 592μs | 불가. 매 스텝 300MB를 읽으므로 대역폭 한계. int32 인덱스여도 ~400μs |
| torch 기본 연산 이벤트 구동 (`nonzero` → gather → `index_add_`) | 150~240μs | 불가. 커널 10여 개 런치 + `nonzero` 동기화가 고정 비용 |
| LIF 원소별 연산만 (6커널, 전파 제외) | 31μs (CUDA graph 19μs) | 참고 |
| **Triton 융합 커널 2개** (전파 + LIF), 프로그램당 뉴런 32개, eager | 46~64μs (활동 ≤5%) | **통과** |
| 같은 것, CUDA graph 재생 | 33~65μs (활동 ≤5%), 161μs (활동 20%) | **통과** (20% 활동은 폭주 영역이라 무관) |

따라서 **이벤트 구동 + 융합 커널이 필수**다. 설계:

1. `propagate` 커널: 그리드 = ⌈N/32⌉. 각 프로그램이 뉴런 32개를 순회하며 `s_prev[n] != 0`인 것만
   전방 CSR 행 `[indptr[n], indptr[n+1])`을 128개씩 읽어 `acc[post] += w` 를 `tl.atomic_add`(int32)로 누산.
   뉴런 1개당 프로그램 1개(그리드 = N)는 유휴 프로그램 비용 때문에 110μs가 나오므로 쓰지 않는다.
2. `lif` 커널: 누산기(int32→float32), `i_syn`, `v`, 불응기 카운터, 스파이크 판정, `acc` 0 초기화를 한 커널에서.
   스파이크 기록은 이 커널 안에서 `slot = atomic_add(counter, 1)` 로 `(t, idx)` 버퍼에 append (호스트 동기화 없음).
3. **결정론 근거**: 누산이 int32 정수 덧셈이므로 atomic 순서와 무관하게 bit-exact다 (벤치에서 확인).
   기록 버퍼의 슬롯 순서만 비결정적이므로 flush 시 `(t, idx)`로 정렬한다.
4. 기록기 flush: `chunk_steps`마다 한 번 counter를 읽고(이때만 동기화) 버퍼를 CPU로 옮긴 뒤 counter 0.
   counter가 `cap`을 넘으면 예외 (조용히 버리지 않는다).
5. 스텝 인덱스 `t`는 스칼라 인자가 아니라 **디바이스 텐서**로 넘기고 커널 뒤에서 `t_dev.add_(1)`.
   그래야 CUDA graph 캡처가 가능하다. `i_ext`도 사전 할당 버퍼에 `copy_`로 채운다
   (`i_ext_fn`은 새 텐서를 반환하지 말고 버퍼에 써 넣는 규약).
6. CUDA graph는 **선택 최적화**(`use_cuda_graph=True` 플래그). eager만으로도 목표를 넘는다.
   그래프 캡처 시 RNG(노이즈 옵션)는 캡처 밖에서 생성해 버퍼로 넣는다.
7. CPU 폴백은 torch 기본 연산(`nonzero` + `index_add_`)으로 같은 수식을 구현하고 느리다고 경고.
   GPU/CPU 간 bit-exact는 요구하지 않는다(부동소수 누적 순서 차이). GPU 내 반복 실행만 bit-exact 필수.

**의존성 결정 필요 (사용자 승인 대기):** Triton 3.2는 torch cu124 wheel의 의존성으로 이미 `.venv`에
설치되어 있지만, `import triton`이 `setuptools` 부재로 실패한다. `setuptools`를 프로젝트 의존성에
추가해야 한다(순수 파이썬, 새 외부 패키지 아님). 승인 전에는 `uv run --with setuptools`로 검증만 한다.
대안: nvcc 12.8이 시스템에 있어 `torch.utils.cpp_extension.load_inline`으로 CUDA C 커널을 써도
되지만 빌드 시간·복잡도가 커서 권하지 않는다.

프로토타입 코드는 `docs/m2-prototype-step.py`에 있다. **그대로 복사하지 말고** 읽고 이해한 뒤
`engine/`의 구조에 맞춰 다시 쓴다 (`t_ref`가 constexpr로 박혀 있는 등 벤치용 단순화가 있다).
LIF 갱신식은 이 문서 상단의 수식을 따른다 (프로토타입은 exp-Euler를 썼는데, 어느 쪽이든 `exp(−dt/τ)`
계수를 쓰면 되며 `docs/m2-report.md`에 선택을 적는다).

### g 스윕 (`sweep.py`) — 완료 조건의 핵심
사전 정의 판정 기준 (실행 전에 코드에 박고, 바꾸면 이력 남김):
- 자극: 무작위 뉴런 100개(시드 고정)에 처음 100ms 동안 상수 전류 주입, 이후 400ms 관찰. dt=1ms.
- 관찰 구간(마지막 200ms) 기준:
  - **소멸**: 스파이크 총수 0
  - **폭주**: 한 번이라도 발화한 뉴런 비율 ≥ 90%, 또는 마지막 50ms 발화율이 그 앞 50ms의 1.5배 이상
  - **유효**: 둘 다 아님. 발화 뉴런 비율이 0.1%~30% 사이면 "정상 범위"로 표기
- g를 로그 스케일로 스윕(예: 20점). 각 점의 (발화 뉴런 비율, 평균 발화율, 판정)을 표로 `docs/m2-report.md`에.
- 유효 g 범위를 `EngineParams` 기본값 근거로 문서화. **범위가 없으면 그것도 결과다.** 파라미터를
  더 만져서 만들어내지 말고 보고.
- 예상 소요 시간을 먼저 말한다 (500스텝 × 20점, 스텝당 <1ms면 수십 초).

## 완료 조건 (CLAUDE.md §6 M2)
- [ ] 유효 g 범위가 존재하고 `docs/m2-report.md`에 표와 함께 문서화됨
- [ ] dt=0.1ms에서 스텝당 100μs 미만 (측정치 기록)
- [ ] 동일 시드 → bit-identical 스파이크 (pytest로 강제)
- [ ] `pytest tests/` 전부 통과
- [ ] "M2: ..." 커밋 하나. sensory/probe 코드 섞지 않는다.

## 하지 말 것
- 서브그래프·엣지 임계값으로 성능 해결
- 결정론 테스트 완화
- 스윕 결과가 마음에 안 든다고 판정 기준 변경 (변경하면 이력 남기고 보고)
- 뉴런 타입명·영역명 사용 (M2는 무작위 뉴런만 쓴다)
