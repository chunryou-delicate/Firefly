# M2 구현 지시서 — LIF 엔진 (torch, GPU)

총괄 세션에서 작성 (2026-09-13). **M1이 커밋되어 있어야 시작한다.** 없으면 멈추고 보고.
먼저 읽을 것: `CLAUDE.md`, `docs/m1-brief.md`, `docs/m1-report.md`, `flysim/graph/graph.py`.

의존성은 기존 그대로(torch cu124, numpy, pytest). 새 패키지가 필요하면 사용자에게 먼저 묻는다.

## 모델 (전부 "가정"으로 주석 표기)

전류 기반 LIF + 지수 시냅스. 상태 벡터 두 개(`v`, `i_syn`), float32, GPU 상주.

```
i_syn ← i_syn · exp(−dt/τ_syn) + g · (W_post @ s_prev)      # s_prev: 직전 스텝 스파이크 (0/1 float)
v     ← v + (dt/τ_m) · (v_rest − v) + (dt/τ_m) · R·(i_syn + i_ext)   # 불응기 중인 뉴런은 v_reset 고정
s     ← (v ≥ v_thresh) & ¬refractory
v[s]  ← v_reset ; refractory_until[s] ← t + t_ref
```

- `W_post`: **post-major** CSR (행=postsynaptic, 열=presynaptic, 값=부호 포함 시냅스 수 float32).
  M1의 역방향(CSC) 배열이 곧 이것이다. 전방 CSR로 `W.T @ s`를 하지 말 것 — 매 스텝 전치 비용이 든다.
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
- GPU에서 `torch.sparse.mm`/`addmv`가 비결정적이면 **테스트를 완화하지 말고** 보고한다.
  (`torch.use_deterministic_algorithms(True)` 시도 → 그래도 안 되면 보고.)
- NaN/Inf 없음, v가 [v_reset, v_thresh] 밖으로 안 나감.
- 작은 합성 그래프(뉴런 3개, A→B 흥분, A→C 억제)로 부호가 맞는지 확인.

### 성능 (`bench.py`, 결과는 `docs/m2-report.md`)
- 실제 그래프 전체(165,122 뉴런, 25.56M 엣지)로 측정. **서브그래프 금지.**
- `torch.cuda.synchronize()` 후 `time.perf_counter`, 워밍업 100스텝 제외, 10,000스텝 평균.
- 목표: **스텝당 100μs 미만** (dt=0.1ms 실시간). 못 미치면 최적화 순서:
  1. `torch.sparse_csr_tensor` + `torch.mv` (cuSPARSE SpMV)인지 확인
  2. 스파이크가 희소하니 `s_prev`의 nonzero만 뽑아 CSC 열 gather-scatter (`index_add_`)로 바꿔보기
  3. 기록기가 병목인지 분리 측정
  그래도 안 되면 수치와 함께 보고. 그래프를 줄이지 않는다.
- 스텝당 GPU 메모리 사용량도 기록.

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
