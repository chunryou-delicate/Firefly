# M2b 구현 지시서 — 전도도 기반 시냅스 (엔진 확장)

총괄 세션 작성 (2026-09-14). 대상: M2 세션. 먼저 읽을 것: `CLAUDE.md`, `docs/m2-report.md` §4.4·§7,
`data-provenance/parameter-decisions.md` 전체, `docs/backlog.md`, `flysim/engine/lif.py`, `flysim/engine/params.py`, `tests/test_engine.py`.

의존성: 기존 그대로. 새 패키지 금지.

## 왜
현 모델(전류 기반)은 역전전위가 없어 점화 상태에서 v가 −4,000 mV까지 내려가고, 어떤 노이즈에서도 낮은 발화율 안정 상태가 없다
(parameter-decisions 2026-09-13 두 항목). 전도도 기반 시냅스는 v를 [E_inh, E_exc] 안에 자연히 가둔다. **결과 곡선을 만들기 위한 변경이 아니라
모델 결함 수정이다.** 이 문장을 리포트에 그대로 쓴다.

## 모델 (전부 "가정" 표기)

뉴런별 상태: `v`, `g_e`, `g_i` (무차원 전도도, 누설 전도도 1/τ_m 기준 단위 1/ms).

```
g_e ← g_e · exp(−dt/τ_e) + g · n_exc_contacts_in     # 흥분 pre가 발화한 접촉 수 합
g_i ← g_i · exp(−dt/τ_i) + g · n_inh_contacts_in     # 억제 pre가 발화한 접촉 수 합 (|w|)
G   = 1/τ_m + g_e + g_i
v_∞ = (v_rest/τ_m + g_e·E_e + g_i·E_i + i_ext/τ_m) / G
v   ← v_∞ + (v − v_∞) · exp(−dt·G)                    # exponential Euler, dt-강건
스파이크·리셋·불응기는 기존과 동일
```

- 역전전위: `E_e = 0 mV`, `E_i = −75 mV` (가정, 통상값). `v_rest = −65`, `v_thresh = −50`, `v_reset = −65`, `τ_m = 20`, `t_ref = 2` 기존 유지.
- `τ_e = 5 ms`, `τ_i = 10 ms` (가정: GABA-A 느림). 기존 `tau_syn`은 전류 모델 전용으로 남긴다.
- **단일 이득 g**: 흥분·억제 접촉당 같은 전도도 증분. 억제 배율 파라미터를 따로 두지 않는다(추가 자유도 금지). 부호는 M1 그대로.
- `i_ext`는 전류(mV/ms 스케일)로 기존과 같은 자리에 들어간다. 위 식처럼 v_∞에 `i_ext/τ_m`을 더하는 대신 기존 전류 모델과의 비교가 쉽도록
  구현 시 `i_ext`를 `(1−exp(−dt·G))·i_ext/(τ_m·G)`로 넣어도 된다. 어느 쪽인지 리포트에 적는다.
- 지연 없음, 노이즈 옵션은 기존 방식(전류 노이즈) 유지.

`EngineParams`에 `synapse: Literal["current","conductance"] = "current"`, `E_exc`, `E_inh`, `tau_e`, `tau_i`를 추가한다.
**기본값은 "current"**여서 기존 테스트·M3·M4 결과가 그대로 재현돼야 한다(회귀 테스트로 강제). `to_dict()`에 전부 포함.

## 구현
- propagate 커널: 누산기 두 개 `acc_e`, `acc_i` (int32). `w > 0`이면 `acc_e[post] += w`, `w < 0`이면 `acc_i[post] += −w`. 정수 atomic이므로 결정론 근거 유지.
  전류 모델에서는 기존처럼 하나만 써도 되고, 둘을 쓰되 `acc_e − acc_i`로 합쳐도 된다(성능 측정 후 선택, 리포트에 기록).
- lif 커널: `SYNAPSE` constexpr로 분기. 전도도 분기에서 `exp(−dt·G)`는 뉴런별로 달라 커널 안에서 `tl.exp` 계산.
- CPU/torch 백엔드에도 같은 식 구현 (참조 구현). 합성 3뉴런 그래프에서 Triton == torch 참조 확인.
- `v_floor`는 전도도 모델에서 불필요. 남겨두되 리포트에 "conductance에서는 미사용" 명시.

## 테스트 추가 (`tests/test_engine.py`에 추가하거나 `tests/test_engine_conductance.py`)
- 회귀: `synapse="current"` 기본값에서 기존 결정론 테스트의 스파이크 리스트가 M2와 **bit-identical** (M2 커밋 시점 결과를 `data-provenance/m2-regression-spikes.npz`로 저장해 대조).
- 전도도: 3뉴런 부호 테스트(억제 시 v가 E_i 아래로 안 내려감), v ∈ [E_i, v_thresh] 항상(i_ext 없을 때), NaN 없음, 결정론 bit-identical(eager/CUDA graph/새 엔진), dt 1 vs 0.1 스파이크 시각 일치(±1 ms), Triton == torch 참조.
- 성능: 전체 그래프 벤치 재실행, M2와 같은 표 형식(활동 0/0.1/1/5%, 기록기 on/off). 목표 100 μs 유지. 초과하면 수치와 원인 보고, 그래프 축소 금지.

## g 스윕 (전도도 모델) — 판정 기준을 실행 전에 코드에 박는다. M2 기준의 결함을 고친 버전이다.
프로토콜: 전체 그래프, dt 1 ms, 무작위 뉴런 100개(M2와 같은 시드 0/1/2)에 100 ms 동안 전류 30, 이후 400 ms 관찰. g 로그 그리드 20점(범위는 프로토타입으로 정하되 이유 기록).
각 g에 대해 기록: 자극 중 하류 스파이크 수(입력 뉴런 제외), 종료 후 0–50 / 50–200 / 200–400 ms 스파이크 수, 발화 뉴런 비율, 활성 뉴런 평균 발화율, min/max v.
판정 (세 시드 모두):
- **SILENT**: 자극 중 하류 스파이크 0
- **IGNITED**: 종료 후 200–400 ms 창에서 발화 뉴런 비율 ≥ 1 % (자기유지)
- **RESPONSIVE**: 하류 스파이크 > 0 이고 IGNITED 아님 ← 이것이 목표 영역
"정상 범위"라는 말은 쓰지 않는다. RESPONSIVE 구간의 하한·상한을 표로 보고한다. 점화가 일어나면 그 상태의 발화율·v 범위를 기록한다(유계인지가 관심).
**추가 스윕 (노이즈)**: RESPONSIVE 구간 중간의 g(기하평균)에서 `noise_sigma` 로그 그리드로 무입력 2 s 런. 판정: 평균 발화율(Hz/뉴런)과 점화 여부.
0.1–5 Hz 사이에서 점화 없이 안정(마지막 1 s 발화 수가 앞 1 s의 0.5–2배)인 σ가 있으면 그 범위를 보고. 없으면 없다고 보고. **어느 쪽이든 결과다.**

기본값 선택 규칙(실행 전 명시): `DEFAULT_G_CONDUCTANCE` = RESPONSIVE 구간의 기하평균(세 시드 교집합). 노이즈 기본값은 0 유지.

## 산출물
- 엔진 변경(`params.py`, `lif.py`, 필요 시 `recorder.py`), 테스트, `flysim/engine/sweep.py` 확장(`--synapse conductance`), `bench.py` 확장.
- `docs/m2b-report.md`: 모델식·가정, 회귀 확인, 벤치 표, g 스윕 표(3시드), 노이즈 스윕 표, 선택된 기본값과 규칙, Deviations.
- `data-provenance/m2b-sweep.json`, `m2b-noise-sweep.json`, `m2b-bench.json`, `m2-regression-spikes.npz`.
- "M2b: ..." 커밋 하나. `flysim/sensory`, `flysim/probe`, `flysim/apps`, 뷰어, 다른 브리프는 건드리지 않는다.
- 예상 시간: 스윕 20점×3시드×500스텝 + 노이즈 스윕은 1분 내. 커널 컴파일·벤치 포함 전체 수 분. 긴 런 전 예상 시간 출력.

## 완료 조건
- [ ] 기본값(current)에서 M2 결과 bit-identical 회귀 통과
- [ ] 전도도 모델 테스트 전부 통과, 결정론 bit-identical
- [ ] 전체 그래프 벤치 < 100 μs/스텝 (초과 시 수치 보고)
- [ ] RESPONSIVE g 구간 존재 여부와 범위가 3시드로 문서화됨 (없으면 없다고)
- [ ] 노이즈 스윕 결과 문서화 (낮은 발화율 상태 존재 여부)
- [ ] `pytest tests/` 전부 통과 (기존 53 + 신규)

## 하지 말 것
- 억제 배율, 뉴런별 이질성 등 자유 파라미터 추가
- 판정 기준을 결과 보고 바꾸기 (바꾸면 이력)
- 그래프 축소, 새 의존성, 다른 모듈 수정
