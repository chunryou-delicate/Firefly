# M4b 실행 지시서 — 전도도 모델로 M3 이득 재선택 + IPI 스윕 재실행

총괄 세션 작성 (2026-09-14). 대상: M3·M4를 구현한 세션(fruit-fly-ce). M2b(883c124)가 커밋되어 있어야 한다.
먼저 읽을 것: `docs/m2b-report.md`, `data-provenance/parameter-decisions.md` 마지막 두 항목, `flysim/engine/params.py`(`synapse`, `DEFAULT_G_CONDUCTANCE`),
`docs/m3-report.md`, `docs/m4-report.md`.

의존성: 기존 그대로. 엔진·프로브·어댑터 코드는 수정하지 않는다(옵션 전달만).

## 동작점 (총괄 결정)
- `EngineParams(synapse="conductance", g=3.162e-4)` (= `DEFAULT_G_CONDUCTANCE`, 규칙: 3시드 공통 RESPONSIVE 구간 기하평균). 그 외 M2b 기본값. 노이즈 0.
- 점화 감지는 M3/M4와 동일(자극 종료 후 100 ms 뒤 활성 ≥ 1 % → 실패 기록). **RESPONSIVE 창이 좁아 상관 입력에서 점화될 수 있다.** 점화되면 그 런은 실패로 표에 남기고 계속 진행한다. a_in을 내려서 점화를 피하는 것은 아래 규칙 안에서만 한다.

## 1단계: a_in 재선택 (M3 규칙 그대로)
`flysim/apps/m3_click.py`에 `--synapse conductance --g 3.162e-4 --tag b` 같은 옵션을 추가해 같은 프로토콜(IPI 35 ms 클릭, 4런, a_in 기본 그리드 → 첫 PASS까지 ×2 연장, 상한 5120)을 돌린다.
판정 기준 M3와 동일. **단, 점화된 점은 PASS가 아니다**(M3 기준에 명시 추가: late-window 활성 ≥1 %면 FAIL_IGNITED). 선택 규칙: 점화 없이 처음 PASS하는 최소 a_in.
PASS가 없으면(전부 NO_TRANSFER 또는 IGNITED) 그것을 결과로 기록하고 2단계는 "M3 이득 640/2560을 그대로 사용"으로 진행하되 그렇게 적는다.
출력: `runs/m3b-*/run.json`, `data-provenance/m3b-results.json`, `docs/m3b-report.md`(표만, 짧게).

## 2단계: IPI 스윕 (M4 프로토콜 그대로)
`flysim/apps/m4_ipi.py`에 같은 옵션을 추가해 9 IPI × 2 모드 실행. 1단계에서 선택된 a_in 사용.
출력: `runs/m4b-ipi/…`, `data-provenance/m4b-tuning.json`, `docs/m4b-tuning.html`(A단계와 같은 형식, 제목에 "phase B (conductance)"), `docs/m4b-report.md`.
리포트에 **A단계 표와 B단계 표를 나란히** 놓는다(spikes_per_pulse, 집합별). 해석은 한 문단, "가정" 표시.

## 산출물·완료 조건
- "M4b: ..." 커밋 하나. 테스트: 기존 76 통과 + `test_m4`에 B단계 tuning.json 스키마·파라미터 고정 검사 추가.
- [ ] 1단계 표(a_in × 판정, 점화 여부 포함)
- [ ] 18런 완료, 점화된 런 수 보고
- [ ] A/B 나란히 표, `m4b-tuning.html`
- [ ] 파라미터 전부 기록(synapse, g, E_exc, E_inh, τ_e, τ_i, a_in)

## 하지 말 것
- g·역전전위·시간상수 변경, 억제 배율 추가, 노이즈 추가
- 판정 기준 변경(점화 = FAIL 추가 외), 뷰어 수정, 그래프 축소
