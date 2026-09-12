# M3 구현 지시서 — JO 어댑터 + 청각 경로 프로브 + 뷰어 출력

총괄 세션 작성 (2026-09-13). **M2가 커밋되어 있어야 시작한다.** 없으면 멈추고 보고.
먼저 읽을 것 (순서대로): `CLAUDE.md`, `docs/m2-report.md`(특히 §4.4), **`data-provenance/parameter-decisions.md`**, `flysim/engine/lif.py`(입력 주입 규약),
`data-provenance/roi-observed.md`, `docs/annotations-observed.md`, `flysim-viewer.html` 상단 주석(run.json 계약).

의존성: 기존 그대로. **scipy 없음** — 대역통과 필터는 numpy로 직접 구현(2차 IIR biquad 또는 FFT 기반).
새 패키지가 필요하면 멈추고 보고.

## 데이터에서 확인된 사실 (총괄 세션이 조회함. 추측 아님)

| 항목 | 값 |
|---|---|
| JO 뉴런 (유지 그래프, `type`이 `JO-`로 시작) | 672개. 그룹별 A 50 / B 88 / C 68 / D 8 / E 267 / F 78 / unclear 113 |
| JO-A/B 타입명 | `JO-A1..A4`, `JO-A-unclear`, `JO-B1_a/b/c`, `JO-B2`, `JO-B3`, `JO-B4_a/b`, `JO-B-unclear` (정확한 목록은 `search_type(r"^JO-[AB]")`로 뽑아 기록) |
| JO 좌우 | `somaSide` 전부 null. **`rootSide`**(L 348 / R 324) 또는 `instance` 접미사 `_L/_R`로 나눈다 |
| JO 좌표 | `somaLocation` 없음(0/672). `data/cache/neuron-roi-v1.parquet`의 시냅스 중심 `cx,cy,cz` 사용. JO-A/B 138개 중 **24개는 시냅스 사이트가 0개**(그래프에서 고립) → 제외하고 개수를 로그 |
| ROI 체계 | `primary_post` 146개 라벨. **AMMC 라벨 없음.** JO-A/B의 주 ROI는 `SAD`(94/138), AMMC-이름 타입(208개)의 주 ROI도 `SAD`(174/208). 즉 이 데이터셋에서 문헌의 "AMMC 단계" = **SAD ROI** |
| SAD 좌우 | `SAD`는 좌우 없는 단일 라벨. 좌우 분리는 뉴런의 side(`somaSide`→없으면 `rootSide`→없으면 instance 접미사)로 한다 |
| WED | ROI `WED(L)`, `WED(R)` 존재. 타입명 `WED*` 900개 중 457개가 주 ROI WED |
| pC1 | 타입 `^pC1` 156개(L/R 균형). 주 ROI는 `SIP(L/R)`가 다수 |
| JO-A/B 직접 출력 | 주로 `SAD*`, `GNG*`, `CB*`, `AN08B*` 타입으로 감. `AMMC*` 타입으로 가는 접촉은 JO-A 376, JO-B 2,656으로 소수 |

문헌의 "편측 약 480개 JO 뉴런"과 672개(양측)의 차이: 이 수치는 타입이 붙은 유지 뉴런만 센 것이다.
`docs/m3-report.md`에 이 차이를 기록한다. 채우려 하지 않는다.

## 엔진 동작점 (총괄 세션 결정, `data-provenance/parameter-decisions.md`)

- **`g = 0.336`** 고정. 엔진 기본값 `DEFAULT_G = 0.886`은 점화(포화) 상태라 **쓰지 않는다.** `EngineParams(g=0.336, ...)`로 명시.
- 그 외 뉴런 파라미터는 M2 기본값 그대로. `v_floor=None`, `noise_sigma=0`.
- **점화 감지**: 자극 종료 후 100 ms가 지나서도 전체 뉴런의 1% 이상이 발화 중이면 그 런은 "ignited"로 표시하고 실패로 기록. 결과로 제출하지 않는다.
- 엔진 API: `LIFEngine(graph, EngineParams(...), device="cuda")`, `reset(seed)`, `run(n_steps, i_ext_fn=fn, record=True)`.
  `fn(t_step, buf)`는 버퍼에 써 넣고 None을 반환한다. 반환값은 `(t_bin, neuron_idx)` int 배열. 자세한 건 `lif.py` docstring.
- 참고 수치(총괄 세션 사전 확인): JO-A/B 114개에 전류 30을 100 ms 주면 g=0.336에서 하류 136개 뉴런이 반응하고 종료 후 50 ms 안에 꺼진다.

## sensory 공통 인터페이스 (`flysim/sensory/base.py`)

```python
class SensoryAdapter:
    target_idx: np.ndarray            # 주입 대상 뉴런 인덱스
    def prepare(self, dt_ms, n_steps) # 신호를 스텝 단위 전류 테이블로 미리 계산 (GPU 텐서 [n_steps, len(target_idx)])
    def inject(self, t_step, buf)     # 엔진의 i_ext_fn 규약: buf에 써 넣고 None 반환 (새 텐서 반환 금지)
    def describe(self) -> dict        # run.json meta에 그대로 들어감: mode, 파라미터, 대상 뉴런 수, 좌우 개수
```

## JO 어댑터 (`flysim/sensory/jo.py`)

입력: 파형 `x[n]`(float, 샘플레이트 `fs`), 단위는 임의(정규화해서 최대 절대값 1).
대상: JO-A ∪ JO-B 중 시냅스 사이트가 있는 뉴런. 좌·우 각각 배열로 보관. 기본은 양측에 같은 신호.
옵션 `ild_db`로 좌우 진폭 차를 줄 수 있다(기본 0).

1. 대역통과 100–400 Hz (**가정**: CLAUDE.md §5.3의 대략치). 4차 Butterworth 상당의 biquad 2단, numpy 구현.
   필터 계수와 차단 주파수를 `describe()`에 기록.
2. **rate 모드** (dt = 1.0 ms): 대역통과 → 정류 → 포락선(1차 저역, τ_env = 2 ms, 가정) → dt 격자로 리샘플
   → `i_ext[JO] = a_in · env[t]`. 모든 대상 뉴런에 같은 값(뉴런별 이질성 없음, 가정).
3. **phase-lock 모드** (dt = 0.1 ms): 대역통과 → dt 격자로 리샘플 → `i_ext[JO] = a_in · max(x[t], 0)`
   (반파 정류, 가정: JO 뉴런은 한 방향 변위에 반응). 뉴런별 위상 오프셋 없음(가정).
4. `a_in`(입력 이득)은 자유 파라미터. 스윕 허용하되 **모든 값과 결과를 기록**. `g=0.336`은 건드리지 않는다. 참고: 전류 30이면 JO 뉴런이 약 60 Hz로 발화한다.
5. 두 모드 모두 구현. `describe()["mode"]` ∈ {"rate", "phase-lock"}.

자극 생성기 (`flysim/sensory/stimuli.py`):
- `click_train(ipi_ms, n_pulses, pulse_ms, carrier_hz, fs)` — 펄스 = `pulse_ms` 길이의 사인 버스트(해닝 창).
  기본 `carrier_hz`는 파라미터이며 값을 고를 때 근거를 `describe()`에 적는다. `ipi_ms=35`는 M3에서 그냥 하나의 값이다.
- `silence(duration_ms, fs)` — 0 배열. 음성 대조군.
- 파형과 `fs`는 run.json `input.envelope`(0~1로 정규화한 포락선)와 `input.label`로 나간다.

## probe (`flysim/probe/`)

`sets.py` — 뉴런 집합 정의. 전부 데이터 조회로 만들고, 각 집합의 크기·좌우 개수를 로그와 리포트에 남긴다. 0건이면 예외.

| 집합 이름 | 정의 |
|---|---|
| `JO_AB_L`, `JO_AB_R` | 어댑터 대상과 동일 |
| `JO_post_L`, `JO_post_R` | JO-A/B로부터 받는 시냅스 접촉 합 ≥ `k_min`(기본 5, 가정)인 직접 후시냅스 뉴런. 좌우는 후시냅스 뉴런의 side |
| `SAD_L`, `SAD_R` | `primary_roi == "SAD"` 인 뉴런, side로 분리. side 없는 것은 `SAD_unk`로 따로 |
| `AMMCtype_L/R` | `type`이 `AMMC`로 시작 (참고용 보조 집합) |
| `WED_L`, `WED_R` | `primary_roi` ∈ {WED(L)}, {WED(R)} |
| `pC1_L`, `pC1_R` | `type` 이 `pC1`로 시작, side로 분리 |
| `rest` | 위 어디에도 없는 나머지 전부 (배경 활동 감시용) |

`probe.py` — 엔진의 (t_bin, idx) 스파이크 기록을 받아 집합별·빈별 스파이크 수와 평균 발화율(Hz/뉴런)을 계산.
빈 폭 `bin_ms` 기본 1 ms. 원본은 parquet(`runs/<run_id>/spikes.parquet`, `rates.parquet`)로 저장.

`export.py` — **run.json 출력. 뷰어 상단 주석이 계약 원본이다.** 지킬 것:
- `neurons.x/y`: `soma_xyz` 있으면 그것, 없으면 ROI 중심(cx,cy). 둘 다 없는 뉴런(333개 + α)은 (nan) 대신 그래프 밖 고정 좌표로 두고 `meta`에 개수 기록. 투영 축은 (x, y) 그대로. 뷰어가 정규화한다.
- `neurons.region`: 위 프로브 집합 인덱스(집합에 안 들면 `rest`). `regions` 배열은 집합 이름 순서.
- `frames.rates`: 집합 × 빈, 0~1 정규화. 정규화 최대값(Hz)을 `meta.rate_norm_hz`에.
- `spikes`: 프레임당 최대 1,500개 균일 샘플. `meta.spike_sample_ratio`에 실제 비율.
- `meta.sensory_mode`, `meta.dt_ms`, `meta.bin_ms`, `meta.n_neurons`, `meta.run_id`, `meta.dataset="MaleCNS v1.0"`,
  `meta.engine_params`(M2 EngineParams.to_dict()), `meta.adapter`(describe()), `meta.stimulus`.
- neurons 좌표는 런마다 같으니 `runs/neurons-v1.json`으로 한 번만 쓰고, run.json에도 포함(뷰어가 단일 파일도 받는다).

## 실험 (`flysim/apps/m3_click.py`)

런 4개. 각 런 전에 예상 소요 시간을 출력한다.

| run_id | 자극 | 모드 | dt |
|---|---|---|---|
| `m3-click-rate` | click train IPI 35 ms, 500 ms 자극 + 앞 100 ms 무음 + 뒤 400 ms 무음 | rate | 1.0 |
| `m3-silence-rate` | 같은 길이 무음 | rate | 1.0 |
| `m3-click-phase` | 같은 자극 | phase-lock | 0.1 |
| `m3-silence-phase` | 무음 | phase-lock | 0.1 |

`a_in`은 로그 스케일 6점 정도 스윕해서, 아래 판정을 처음 통과하는 최소값과 그 위 한두 점을 기록한다.
스윕한 모든 점의 결과 표를 리포트에 남긴다.

**판정 기준 (실행 전에 코드에 박는다. 바꾸면 이력을 남긴다):**
- 자극 구간(100–600 ms) 대 대조군 같은 구간의 평균 발화율 비교. 대조군에 노이즈가 없으면 대조군은 0이어야 한다.
- **전달 관찰**: `JO_AB_*` 발화 > 0 이고, `JO_post_*` 자극 구간 발화율이 대조군의 같은 구간보다 크며 절대값 ≥ 1 Hz/뉴런. 좌우 각각.
- **무음 대조군 조용함**: 대조군 런에서 모든 프로브 집합의 스파이크 수 0 (노이즈 없음 기준).
- **실패 감지** (뷰어와 동일): 발화 뉴런 0 또는 전체의 90% 이상 → 그 런은 결과로 제출하지 않고 실패로 기록.
- SAD/WED/pC1 응답은 **관찰만** 기록한다. M3 완료 조건에 넣지 않는다. 안 나와도 정상.

## 산출물
- `flysim/sensory/{base,jo,stimuli}.py`, `flysim/probe/{sets,probe,export}.py`, `flysim/apps/m3_click.py`
- `tests/test_sensory.py`(필터 통과 대역 확인, 두 모드의 전류 테이블 shape·범위, describe 키), `tests/test_probe.py`(집합 크기 > 0, 겹침 없음, export가 계약 키를 전부 가짐)
- `runs/m3-*/run.json` 4개 (gitignore 대상이므로 커밋 안 함) — 대신 `docs/m3-report.md`에 집합 크기 표, a_in 스윕 표, 판정 결과, 사용한 파라미터 전부.
- "M3: ..." 커밋 하나.

## 완료 조건 (CLAUDE.md §6 M3)
- [ ] 클릭 트레인 → JO_AB → JO_post 전달이 두 모드 모두에서 관찰됨 (판정 기준 통과)
- [ ] 무음 대조군에서 프로브 집합이 조용함
- [ ] run.json 4개가 뷰어에서 열리고 실패 경고가 뜨지 않음 (파일을 열어 확인할 수 없으면 계약 키 검증 테스트로 대신하고 그렇게 적는다)
- [ ] `meta.sensory_mode` 기록됨
- [ ] `pytest tests/` 통과

## 하지 말 것
- 뷰어 수정·신규 제작
- JO 외 뉴런에 직접 주입해서 결과 만들기
- `g=0.336`이나 뉴런 파라미터 변경 (점화되면 실패로 기록하고 보고)
- 타입명·ROI명을 이 문서와 `annotations-observed.md`/`roi-observed.md` 밖에서 가져오기
