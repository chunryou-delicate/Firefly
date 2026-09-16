# M2c 구현 지시서 — 스파이크 빈도 적응 전류

총괄 세션 작성 (2026-09-16). 대상: 엔진 세션(M2·M2b를 한 세션이 적합). 먼저 읽을 것: `docs/m2b-report.md`, `data-provenance/parameter-decisions.md` 전체,

> **보고 대상:** 총괄 세션. 세션 이름은 재시작 때마다 바뀌므로 **이 지시를 전달한 메시지의 `from-name`**(또는 사용자가 알려준 이름)을 그대로 쓴다. 문서에 적힌 옛 이름은 무효다. 이름을 모르면 `ListAgents`로 `fruit-fly-*` 세션을 확인하고, 애매하면 사용자에게 묻는다.

`flysim/engine/{params,lif}.py`, `tests/test_engine_conductance.py`, `docs/backlog.md`.

의존성: 기존 그대로.

## 왜
전류·전도도 모델 모두 "무음 아니면 점화(150~245 Hz 포화)"의 쌍안정이고 낮은 발화율 상태가 없다. 포화를 뉴런 스스로 깎는 기제가 없기 때문이다.
적응 전류는 발화할수록 자기 억제가 쌓이는 보편적 기제다. **결과 곡선을 위한 튜닝이 아니라 포화 기제 추가**이며 파라미터는 아래 규칙으로 실행 전에 정한다.

## 모델 (가정 표기)
뉴런별 상태 `w` (전류 단위, `i_ext`와 같은 자리):
```
w ← w · exp(−dt/τ_w)            # 매 스텝
발화 시: w ← w + b
막전위 식의 외부 전류 자리에 (i_ext − w) 를 넣는다  # current·conductance 두 모델 공통
```
- `τ_w = 100 ms` 고정 (가정; 통상 50~300 ms). `b`는 스윕으로 정한다.
- `EngineParams`에 `adapt_b: float = 0.0`, `adapt_tau_w: float = 100.0`. **기본 b=0**이면 기존 커널 경로와 bit-identical(회귀 테스트, M2b의 `tests/regression_cases.py`에 케이스 추가: conductance g=3.162e-4 실그래프 100스텝).
- Triton: `ADAPT` constexpr 분기. `w`는 float32 상태 벡터 하나. 결정론 근거 불변(누산기 정수).
- torch 참조 백엔드에도 구현. 1뉴런 상수전류 테스트: b>0이면 ISI가 단조 증가하고 정상상태 발화율이 b=0보다 낮다. Triton == torch.

## b 스윕 (전도도 모델, g = DEFAULT_G_CONDUCTANCE = 3.162e-4, 실행 전 코드에 고정)
1. **응답 스윕**: M2b 프로토콜(100 뉴런 100 ms 자극, 3시드, SILENT/RESPONSIVE/IGNITED)을 b ∈ 로그 그리드 12점(범위는 프로토타입으로 정하고 이유 기록; 단위 EPSP 0.065 mV·임계 갭 15 mV를 참고)에서 실행. 점화가 일어나는 b에서는 점화 상태의 활성 비율·발화율을 기록한다(포화가 깎이는지).
2. **노이즈 스윕**: 각 b(또는 대표 4점)에서 σ 로그 그리드 무입력 2 s + off 0.4 s. 판정은 M2b와 동일. **목표 상태 정의(선기록)**: on 구간 마지막 1 s 평균 발화율 0.1~5 Hz/뉴런, 앞 1 s 대비 0.5~2배, off 후 200 ms 활성 < 1 %.
3. **g 재스윕**: 목표 상태가 존재하는 b 중 최소값에서 g 로그 그리드 12점으로 RESPONSIVE 창 폭을 다시 잰다(창이 넓어지는지).
기본값 규칙(선기록): `DEFAULT_ADAPT_B` = 목표 상태(2)가 존재하는 최소 b. 존재하지 않으면 `DEFAULT_ADAPT_B = 0`을 유지하고 "없음"을 보고한다. 노이즈 기본값은 계속 0.

## 산출물
- 엔진 변경, 테스트, `sweep_conductance.py` 확장(`--adapt`), 벤치 재실행(표 형식 동일, 100 μs 목표).
- `docs/m2c-report.md`, `data-provenance/m2c-*.json`, 회귀 npz 갱신(추가 케이스만).
- "M2c: ..." 커밋 하나. `flysim/live`, `flysim/sensory`, `flysim/probe`, `flysim/apps`, 뷰어는 건드리지 않는다.
- 끝나면 총괄 세션에 보고: 커밋, pytest, 회귀, 벤치, b 응답 스윕 표, 노이즈 스윕 표(목표 상태 존재 여부), g 재스윕 창, DEFAULT_ADAPT_B, Deviations.

## 하지 말 것
- τ_w 조정, 억제 배율 추가, 판정 기준 사후 변경, 그래프 축소, 새 의존성.
