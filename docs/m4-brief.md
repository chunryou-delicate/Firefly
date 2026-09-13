# M4 구현 지시서 (A단계) — IPI 튜닝 곡선, 현 모델 그대로

총괄 세션 작성 (2026-09-13). M3(f86d4f1)가 커밋되어 있어야 한다.
먼저 읽을 것: `CLAUDE.md` §6 M4 (특히 "피크가 안 나와도 정상"), `docs/m3-report.md`, `data-provenance/parameter-decisions.md` 전체,
`flysim/apps/m3_click.py`, `flysim/probe/export.py`.

의존성: 기존 그대로. matplotlib 없음. 곡선은 아래 방식으로 HTML에 인라인 SVG로 그린다.

## 알고 시작할 것 (총괄 세션 확인 사항)
- 현 모델(g=0.336, 노이즈 0)에서 전파는 첫 홉(JO_post)까지다. M3에서 SAD·AMMCtype·pC1은 모든 이득에서 0 Hz였다.
  배경 노이즈로 낮은 발화율 상태를 만들 수 없음도 확인됐다(parameter-decisions.md). **따라서 pC1 곡선이 전부 0이어도 그것이 이 단계의 정상 결과다.**
  곡선은 프로브 집합 전부(JO_AB, JO_post, SAD, AMMCtype, WED, pC1)에 대해 그린다. 첫 홉의 IPI 의존성 자체가 이 단계의 관찰 대상이다.
- 파라미터 고정: `g=0.336`, 뉴런 파라미터 M2 기본, `noise_sigma=0`, `v_floor=None`.
  `a_in`은 M3에서 선택된 값 고정: rate 640, phase-lock 2560. **어떤 파라미터도 스윕·조정하지 않는다.** 이 문장이 M4의 핵심 규칙이다.

## 스윕 설계 (실행 전에 코드에 박는다)
- IPI ∈ {20, 25, 30, 35, 40, 45, 50, 55, 60} ms. 9점.
- 자극: 앞 100 ms 무음 → 클릭 트레인 **고정 길이 700 ms** (펄스 수 = floor(700/IPI), IPI별로 다름) → 뒤 200 ms 무음. 총 1000 ms.
  펄스 파형·반송주파수·펄스 길이는 M3의 `click_train` 기본값 그대로. 값은 결과에 기록.
- 모드: rate(dt 1 ms)와 phase-lock(dt 0.1 ms) 둘 다. 총 18런. 각 런 전 예상 시간 출력(M3 기준 런당 1초 미만).
- 시드 고정(0). 노이즈가 없으니 시드는 기록용.
- 좌우는 같은 신호(ILD 0).

## 측정량 (집합별, 좌우 따로)
1. `rate_stim`: 자극 구간(100–800 ms) 평균 발화율 (Hz/뉴런)
2. `spikes_per_pulse`: 자극 구간 스파이크 수 / 펄스 수 (IPI별 펄스 수 차이를 보정한 값). **주 곡선은 이것.**
3. `latency_ms`: 첫 펄스 시작부터 그 집합의 첫 스파이크까지 (없으면 null)
4. `late_active_frac`: 800 ms 이후 발화 뉴런 비율 (점화 감지; > 1%면 그 런은 "ignited"로 실패 기록)
5. 뷰어 실패 규칙(발화 0 또는 90% 이상) 해당 여부

## 산출물
- `flysim/apps/m4_ipi.py` — 스윕 실행. 결과 `runs/m4-ipi/tuning.json`
  (`{meta: {params, adapter, stimulus, git_commit}, curves: {mode: {set: {ipi_ms: [...], spikes_per_pulse: [...], rate_stim: [...], latency_ms: [...]}}}, runs: [...]}`).
- IPI별 `run.json` 18개는 `runs/m4-ipi/<mode>-ipi<NN>/run.json` (뷰어 확인용, 미커밋).
- `runs/m4-ipi/tuning.html` — **단일 파일, 외부 의존 없음, 인라인 SVG.** 모드별 패널, 집합별 곡선(좌우 실선/점선), x축 IPI, y축 spikes_per_pulse.
  35 ms에 세로 참고선 하나(문헌값 표시일 뿐, 기대치가 아님). 파라미터 요약을 페이지 하단에 표로. 이 파일은 뷰어(`flysim-viewer.html`)가 아니라 리포트 그림이다. **뷰어는 수정하지 않는다.**
  같은 HTML을 `docs/m4-tuning.html`로 복사해 커밋한다(용량 작음).
- `tests/test_m4.py` — 스윕 설정이 고정값과 일치, tuning.json 스키마, 펄스 수 계산.
- `docs/m4-report.md` — 표 형태의 전체 수치, 판정, 관찰(피크 유무를 있는 그대로), 사용 파라미터 전부, 점화 여부.
- `data-provenance/m4-tuning.json` 에 tuning.json 복사본 커밋.
- "M4: ..." 커밋 하나.

## 완료 조건 (CLAUDE.md §6 M4)
- [ ] 18런 완료, 점화된 런 없음
- [ ] 곡선이 `tuning.html`에 그려지고 브라우저에서 열림 (세션에서 못 열면 HTML 파싱 테스트로 대체하고 그렇게 적는다)
- [ ] 모든 파라미터가 tuning.json과 리포트에 기록됨
- [ ] `pytest tests/` 통과
- [ ] 리포트에 "pC1 응답이 0이면 0이라고" 그대로 기록. 해석은 한 문단 이내, 원인 추정은 "가정"으로 표시.

## 하지 말 것
- g, a_in, 노이즈, 뉴런 파라미터, 펄스 파형 변경. 곡선 모양을 바꾸려는 어떤 시도도 금지. 시도했다면 되돌리고 이력을 리포트에 남긴다.
- 뷰어 수정, 그래프 축소, 새 의존성.
- 결과 해석에 "학습", "인식", "튜닝됨" 같은 단어 사용. 수치만.
