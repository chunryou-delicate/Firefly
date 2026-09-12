# M1 구현 지시서 — CSR 그래프 + 어노테이션 조회 API

총괄 세션에서 작성 (2026-09-12). 구현 세션은 이 문서와 아래 파일을 먼저 읽는다.

- `CLAUDE.md` (거버넌스, 절대 규칙)
- `data-provenance/retention.md`, `data-provenance/m0-verify.json`
- `docs/annotations-observed.md` (실제 컬럼·값. 여기 없는 식별자는 쓰지 않는다)

M0 상태: `data/parquet/{annotations,neurotransmitters,weights}.parquet` 존재. 가상환경 `.venv`
(uv). 의존성은 pyarrow·pandas·numpy·torch(cu124)·pytest뿐이며 추가 시 사용자에게 물어본다.

## 결정된 사항 (총괄 세션에서 사용자 승인)

1. **유지 규칙**: `annotations.status == "Traced"` 뉴런만, 엣지는 양 끝이 모두 유지 뉴런인 것만.
   가중치 임계값 없음. 165,122 뉴런 / 25,563,197 엣지가 나와야 한다. 다르면 중단하고 보고.
2. **엣지 부호** (NT 컬럼은 `neurotransmitters.consensus_nt`, 키는 `body`):
   - `acetylcholine` → +1
   - `gaba`, `glutamate` → −1
   - `unclear`(3,100), NT 테이블 누락(502) → **+1로 가정** (ACh 다수 근거). 코드 주석과
     `data-provenance/edge-signs.md`에 "가정"이라고 명시하고 개수를 기록한다.
   - `histamine`(5,910), `dopamine`, `octopamine`, `serotonin` → 시냅스 전달로는 **+1 가정**하되
     별도 마스크로 표시해 나중에 신경조절 처리를 붙일 수 있게 한다. 히스타민은 초파리 광수용체
     전달물질로 억제성으로 알려져 있으나, 이 프로젝트 1차 대상(청각)과 무관하므로 우선 +1 통일하고
     문서에 남긴다. 가정임을 명시.
   - 부호는 **presynaptic 뉴런** 기준으로 엣지에 부여한다 (Dale 원칙 가정).
3. **정규화 안 함**: 가중치는 시냅스 접촉 수 정수 그대로 저장 (float32 변환만). 스케일링은
   엔진(M2) 파라미터로 한다.

## 산출물

```
flysim/graph/
  build.py      # parquet → CSR 캐시 (data/cache/graph-v1.npz 또는 torch .pt)
  graph.py      # Graph 클래스: 로드, bodyId↔index, CSR 접근, 최단경로
  annotations.py# 조회 API
tests/test_graph.py
docs/m1-report.md   # 통계, 히스토그램 이미지 경로, 걸린 시간
```

### Graph 클래스 요구사항
- `bodyId → idx` (dict 또는 정렬 배열 + searchsorted), `idx → bodyId` 배열.
- CSR: `indptr(int64)`, `indices(int32)`, `weight(float32, 부호 포함)` — 전방(pre→post).
  역방향 CSC도 만들어 둔다 (M4 이후 상류 추적용).
- 뉴런 부가 배열: `side`(L/R/M/null → 정수 코드), `nt`(정수 코드), `has_soma_xyz`, `soma_xyz`.
- `shortest_path(src_bodyId, dst_bodyId, max_hops=8)` — 가중치 무시 BFS. 경로 없으면 예외.
- 캐시 빌드 시 로그: 필터 조건, 전후 개수, 부호별 엣지 수.

### 조회 API (`annotations.py`)
- `by_type(name: str, side: str|None=None) -> np.ndarray[idx]`
- `by_class(name)`, `by_superclass(name)`, `by_fru_dsx(value)`
- `search_type(pattern: str)` — 정규식/부분일치로 후보 목록 반환 (탐색용, 0건이면 예외)
- **모든 함수는 0건이면 `LookupError`를 던진다.** 빈 배열 반환 금지.
- 기본은 `type` 컬럼 exact match. `instance`는 `_L`/`_R` 접미가 붙은 경우가 있으니
  side 필터에 활용하되, 먼저 `somaSide` 컬럼을 쓴다.

### 통계 (`docs/m1-report.md`)
- 입차수/출차수 히스토그램 (로그 스케일, PNG). matplotlib이 필요하면 **사용자에게 먼저 묻는다.**
  묻기 싫으면 텍스트 히스토그램으로 대체 가능.
- 부호별 엣지 수·시냅스 합. 자기 루프 수(원본에 123개 있었음, 유지 후 개수 보고).
- CSR 메모리 사용량(바이트).

## 완료 조건 (CLAUDE.md §6 M1)
- [ ] 임의의 두 뉴런 최단 경로 조회 가능
- [ ] `by_type("DNp01")`이 결과를 내고, `by_type("없는이름")`이 예외를 낸다 (테스트로 강제)
  - `DNp01`은 annotations-observed 기준 존재 확인된 값. 다른 타입명은 데이터에서 찾아 쓴다.
- [ ] 입출력 차수 히스토그램 산출
- [ ] `pytest tests/` 통과
- [ ] 커밋 메시지 "M1: ..." 하나로 커밋. M2 코드 섞지 않는다.

## 하지 말 것
- 그래프 축소, 임계값 필터 기본 적용
- 뉴런 타입명·영역 약어를 기억에서 쓰는 것 (JO, AMMC, pC1 등은 M3에서 데이터로 찾는다)
- 뷰어 새로 만들기
