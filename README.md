# Firefly — flysim: MaleCNS 초파리 커넥톰 시뮬레이터

성체 수컷 초파리 중추신경계 전체 배선도(MaleCNS v1.0, 뉴런 165,122개 · 방향성 엣지 25,563,197개)를
GPU에 올려 임의의 감각 입력을 넣고 뉴런 활동을 관찰하는 취미 프로젝트. 1차 응용은 청각 경로
(존스턴 기관 JO → SAD/AMMC → WED → pC1)이며, 이후 다리 진동, 중앙복합체 방향 추정 등을 어댑터로 붙인다.

프로젝트 거버넌스와 절대 규칙은 [CLAUDE.md](CLAUDE.md)에 있다. 요약하면: **데이터를 지어내지 않는다,
그래프를 조용히 줄이지 않는다, 결과를 과장하지 않는다, 원하는 그림이 나오도록 파라미터를 만지지 않는다.**

## 데이터와 라이선스

- MaleCNS v1.0 (FlyEM/HHMI Janelia, Cambridge, MRC LMB, Google Research). Cell 2026, DOI 10.1016/j.cell.2026.08.015. **CC-BY 4.0.**
- 원본은 저장소에 없다. `python -m flysim.data.download`가 공개 GCS 버킷에서 받아 MD5를 검증하고 `data-provenance/hashes.json`에 기록한다.
- 출처·라이선스 고지: [THIRD_PARTY.md](THIRD_PARTY.md).

## 실행

```bash
uv sync                                   # torch cu124, pyarrow, pandas, numpy, pytest, setuptools(triton)
uv run python -m flysim.data.download     # 핵심 파일 3개, 1.1 GB
uv run python -m flysim.data.convert      # feather -> parquet, docs/annotations-observed.md 생성
uv run python -m flysim.data.verify       # CLAUDE.md §4.2 기준값 대조
uv run python -m flysim.data.download --files syn_partners   # 6.8 GB, 뉴런별 ROI·좌표용 (선택)
uv run python -m flysim.data.roi          # data/cache/neuron-roi-v1.parquet
uv run python -m flysim.graph.build       # CSR 캐시 data/cache/graph-v1.npz (약 25초)
uv run python -m pytest tests/ -q         # 78 tests
uv run python -m flysim.apps.m3_click     # 클릭 트레인 실험 -> runs/m3-*/run.json
uv run python -m flysim.apps.m4_ipi       # IPI 스윕 -> runs/m4-ipi/, docs/m4-tuning.html
```

환경: Ubuntu, Python 3.10, RTX 4090 Laptop 16 GB. 전체 그래프가 GPU에 상주하며(약 340 MB) 스텝당 36~85 μs(dt 0.1 ms)로 실시간 이상이다.

`run.json`은 [flysim-viewer.html](flysim-viewer.html)에 드래그하면 재생된다. 뷰어 상단 주석이 출력 계약의 원본이다.

## 구조

```
flysim/
  data/      다운로드·해시·parquet 변환·기준값 검증·ROI 집계
  graph/     CSR/CSC 그래프, 엣지 부호(신경전달물질 예측), 어노테이션 조회 API, 최단경로
  engine/    LIF 커널 (torch + Triton, 이벤트 구동, bit-exact 결정론). 전류/전도도 시냅스 선택
  sensory/   입력 어댑터. JO 어댑터: rate 모드(dt 1 ms) / phase-lock 모드(dt 0.1 ms)
  probe/     뉴런 집합별 발화율, run.json 출력
  apps/      실험 스크립트 (M3 클릭 트레인, M4 IPI 스윕)
docs/            마일스톤 지시서(mN-brief.md)와 리포트(mN-report.md), 백로그
data-provenance/ 해시, 유지 규칙, 엣지 부호, 파라미터 결정 이력, 스윕 원본 JSON
```

## 진행 상태

| 단계 | 내용 | 커밋 | 결과 |
|---|---|---|---|
| M0 | 데이터 확보, 해시, parquet | 762fca3 | 기준값 4개 모두 1 % 이내 ([retention.md](data-provenance/retention.md)) |
| M1 | CSR 그래프, 조회 API | 08d6138 | [m1-report.md](docs/m1-report.md) |
| M2 | LIF 엔진 (전류 시냅스) | 8bae9d3 | [m2-report.md](docs/m2-report.md). 네트워크가 쌍안정임을 발견 |
| M3 | JO 어댑터, 청각 프로브, 뷰어 출력 | f86d4f1 | [m3-report.md](docs/m3-report.md). JO→직접 후시냅스 전달 확인, 무음 대조군 0 |
| M4 A | IPI 20~60 ms 스윕 (전류 모델) | 67f85c8 | [m4-report.md](docs/m4-report.md), [m4-tuning.html](docs/m4-tuning.html). pC1 = 0 |
| M2b | 전도도 기반 시냅스 (역전전위) | 883c124 | [m2b-report.md](docs/m2b-report.md). 전압은 유계, 쌍안정은 그대로 |
| M4 B | IPI 스윕 (전도도 모델) | 9dfb23a | [m4b-report.md](docs/m4b-report.md), [m4b-tuning.html](docs/m4b-tuning.html). 홉 2까지 전파, pC1 = 0 |

### 지금까지 알게 된 것 (수치만)

- 유지 그래프의 흥분 접촉 7,730만 대 억제 4,680만. 시냅스 이득 g에 대해 네트워크는 "무음"과 "점화(뉴런 12~35 %가 150~245 Hz)" 두 상태만 가지며 1차 전이를 보인다. 배경 노이즈를 넣어도 낮은 발화율의 안정 상태는 생기지 않는다(전류·전도도 모델 모두).
- 점화 아래 영역에서 JO 클릭 입력은 첫 홉(전류 모델) 또는 둘째 홉(전도도 모델)까지 전달되고 pC1에는 닿지 않는다. 구애노래 IPI 35 ms 근처의 피크는 어느 집합에도 없다. CLAUDE.md M4 규칙대로 이것은 정상 결과다.
- 모델 가정(뉴런 파라미터, 부호 규칙, 역전전위, 지연 없음 등)과 그 결정 시점·이유는 전부 [parameter-decisions.md](data-provenance/parameter-decisions.md)에 있다.

### flysim-live 조종석 (2026-09-16 완료)

엔진을 상주시키고 브라우저에서 실시간으로 조종·관찰한다. 계약은 [m5-protocol.md](docs/m5-protocol.md) v1.1.

```bash
uv run python -m flysim.live.server        # ws 8765 + http 8080
# 브라우저에서 http://127.0.0.1:8080/  (원격은 터널 경유, 인증 없음 — 공개망 금지)
```

| 부분 | 커밋 | 내용 |
|---|---|---|
| 서버 | 8c2c2cf | 웹소켓 제어·텔레메트리, 오디오 스트리밍, 제어 로그 재생으로 결정론 보장, 정적 서빙 ([m5a-report.md](docs/m5a-report.md)) |
| 조종석 | 11452e3, 3cf2ba2 | 단일 HTML. 자극·파라미터 조종, 실시간 뉴런 필드, 하류 2홉 추적, 좌우 차이, 스냅샷 ([m5b-report.md](docs/m5b-report.md)) |

총괄 세션이 실서버로 확인: 무음은 전 영역 0.000 Hz, 35 ms 클릭 트레인은 JO_AB → JO_post(57/50 Hz) → WED(4.5 Hz)로 홉 2까지 전달, 스냅샷은 뷰어 계약을 만족한다. speed 1이 dt 1.0 ms·0.1 ms 모두 유지된다.

### 적응 전류 (2026-09-16, 기록만 하고 미채택)

[m2c-report.md](docs/m2c-report.md), 커밋 c5d7178. 낮은 발화율 상태가 생기긴 하지만 세 가지 이유로 채택하지 않았다 — 막전위가 −834 mV까지 내려가고(M2b가 없앤 것과 같은 종류의 인공물), 그 상태에서 뉴런의 96~100 %가 발화 중이며(CLAUDE.md §3.3의 실패 기준), 선택된 b와 기존 g가 함께 쓸 수 없는 쌍이다. `adapt_b`는 0으로 남아 있어 기존 결과는 그대로다. 사유는 [parameter-decisions.md](data-provenance/parameter-decisions.md)에.

### 다음 후보
- 다리 기계수용 구심신경 어댑터(바닥 진동), 중앙복합체 EPG 방향 추정, 버섯체 KC→MBON 가소성 — [CLAUDE.md §6](CLAUDE.md) M5 이후

## 작업 방식

총괄 세션 하나가 지시서(`docs/mN-brief.md`)를 쓰고 커밋하면, 마일스톤별 구현 세션이 그 지시서대로 구현·테스트·커밋하고 결과를 보고한다.
총괄이 완료 조건을 재검증한 뒤 다음 지시서를 쓴다. 지어내기 쉬운 식별자(세포 유형명, ROI명)는 총괄이 데이터에서 직접 조회해
지시서에 박아 넣고, 그 밖의 식별자는 코드에 쓰지 않는다. 판정 기준은 실행 전에 코드에 박고, 바꾸면 이력을 남긴다.
