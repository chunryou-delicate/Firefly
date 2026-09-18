# M2d 구현 지시서 — 적응을 전류가 아니라 전도도로

총괄 세션 작성 (2026-09-19).

> **보고 대상:** 총괄 세션. 이 지시를 전달한 메시지의 `from-name`을 쓴다.

먼저 읽을 것: `CLAUDE.md`(특히 §3.3), `docs/m2c-report.md`, `data-provenance/parameter-decisions.md`의 2026-09-14·09-16 항목, `flysim/engine/{params,lif}.py`, `flysim/engine/sweep_conductance.py`.

의존성: 기존 그대로.

## 왜
M2c의 적응 **전류**는 목표 상태를 만들긴 했지만 두 가지가 무너졌다. 막전위가 −834 mV까지 내려갔고(전류라 역전전위에 안 묶인다 — M2b가 고친 것과 같은 고장), 그 상태에서 뉴런의 96~100 %가 발화 중이었다(CLAUDE.md §3.3이 90 % 이상을 파라미터 실패로 규정한다).
실제 뉴런의 적응은 칼륨 전도도다. 전도도로 넣으면 구조적으로 역전전위에 묶인다. **결과를 만들려는 조정이 아니라 같은 종류의 모델 결함 수정**이며, 이 문장을 리포트에 그대로 쓴다.

## 모델 (전부 "가정" 표기)
뉴런별 상태 `g_a` (전도도, 1/ms 단위. 시냅스 전도도와 같은 단위):
```
g_a ← g_a · exp(−dt/τ_a)          # 매 스텝
발화 시: g_a ← g_a + b_g
G   = 1/τ_m + g_e + g_i + g_a
v_∞ = (v_rest/τ_m + g_e·E_e + g_i·E_i + g_a·E_a + i_ext/τ_m) / G
v   ← v_∞ + (v − v_∞) · exp(−dt·G)
```
- `E_a = −75 mV` 기본(= `E_inh`. 가정. 칼륨 역전전위는 통상 −80 ~ −90이지만 이 모델에 새 상수를 늘리지 않는다). 파라미터로 노출한다.
- `τ_a = 100 ms` 고정(M2c와 같게 둬서 비교 가능하게 한다).
- `EngineParams`에 `adapt_g_b: float = 0.0`, `adapt_tau_a: float = 100.0`, `E_adapt: float = -75.0`.
- **`synapse="conductance"`에서만 동작한다.** `synapse="current"`에 `adapt_g_b > 0`이면 `ValueError`. 전류 모델에는 역전전위가 없어 의미가 없다.
- M2c의 `adapt_b`(전류형)는 **지우지 말고 그대로 둔다.** 기본 0이고 이력이다. 둘을 동시에 켜면 `ValueError`.
- 기본값 0에서 기존 커널 경로와 **bit-identical**. `tests/regression_cases.py`에 conductance 케이스를 하나 추가해 강제한다.

## 사전 등록 스윕 (실행 전에 코드에 박는다)
1. **결합 격자**: `b_g` 로그 12점 × `g` 로그 8점(2e-4 ~ 2e-3 구간을 덮되 근거를 기록). 시드 0·1·2. 판정은 M2b와 동일한 SILENT / RESPONSIVE / IGNITED.
2. **노이즈 스윕**: 세 시드 모두 RESPONSIVE인 (b_g, g)에서 무입력 2 s + 노이즈 off 0.4 s. σ 로그 격자.
3. **"쓸 수 있는 상태" 정의 — M2c의 교훈을 반영해 조건을 하나 추가한다.** 네 조건을 **전부** 만족해야 한다.
   - 마지막 1 s 평균 발화율 0.1 ~ 5 Hz/뉴런
   - 마지막 1 s가 앞 1 s의 0.5 ~ 2배
   - 노이즈 off 후 200 ms 활성 < 1 %
   - **마지막 1 s 활성 뉴런 비율 ≤ 30 %** ← 신규. M2c는 이 값이 96~100 %였고 그건 §3.3 기준 실패다.
   활성 비율은 조건을 만족하든 안 하든 **항상 표에 적는다.**
4. **막전위 경계**: 모든 런에서 `v ∈ [min(E_a, E_i), v_thresh]`인지 검사하고 최저값을 기록한다. 이게 이 작업의 핵심 확인 사항이다.

기본값 규칙(스윕 전 명시): `DEFAULT_ADAPT_G_B` = 쓸 수 있는 상태가 존재하는 **최소** `b_g`, 그때의 `g`는 그 `b_g`에서 세 시드 공통 RESPONSIVE 구간의 기하평균. 없으면 둘 다 0/기존값을 유지하고 **"없음"을 결과로 보고한다.** 없는 것도 정상적인 결과다.

## 산출물·완료 조건
- 엔진 변경, 테스트(합성 3뉴런 부호·경계, 결정론 bit-identical, dt 1 vs 0.1 일치, Triton == torch 참조, current+adapt_g_b 조합 예외), `sweep_conductance.py` 확장, 벤치 재실행(표 형식 동일, 100 μs 목표).
- `docs/m2d-report.md`, `data-provenance/m2d-*.json`.
- [ ] 기본값에서 회귀 bit-identical
- [ ] 모든 런에서 v가 경계 안 (M2c의 −834 mV가 사라졌는지)
- [ ] 쓸 수 있는 상태의 존재 여부와 그때의 활성 비율이 표로 문서화
- [ ] `pytest tests/` 전부 통과 (현재 273개)
- [ ] "M2d: ..." 커밋 하나. **push 금지**

## 하지 말 것
- τ_a 조정, 억제 배율 추가, 판정 기준 사후 변경(바꾸면 이력), 그래프 축소, 새 의존성
- `flysim/live/`, `flysim-*.html`, `flysim/probe/`, `flysim/data/` 수정
