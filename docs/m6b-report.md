# M6b report — 뉴런 점구름 + 3D 자산 서빙

구현 세션(서버 창), 2026-09-18. 사양: `docs/m6b-brief.md`. 계약: `docs/m6-3d-contract.md`(수정하지 않았다),
`docs/m5-protocol.md`(hello에 `assets` 추가만).

## 1. 점구름 (계약 파일 1)

`python -m flysim.probe.export3d` → `data/cache/neurons-3d.bin` + `.json`. 1.8초.

| 항목 | 값 |
|---|---|
| 뉴런 수 | 165,122 |
| `.bin` 크기 | 2,311,708 B (2.31 MB) = 165,122 × 14 B |
| `.json` 크기 | 612 B |
| `src_counts` | soma 140,024 · centroid 24,945 · none 153 (합 165,122) |
| NaN 좌표 뉴런 | 153 = `src == 2` 개수와 일치 |
| bbox (복셀) | min [2468, 4758, 10154] · max [93668, 68996, 134531] |
| 배열 오프셋 | `pos` 0 (float32×495,366) · `set` 1,981,464 (uint8×165,122) · `src` 2,146,586 (uint8×165,122) |
| `graph_cache_md5` | `7cc53993e4634a8131045d743ccf5861` (`data/cache/graph-v1.npz`) |

오프셋 세 개는 계약 예시에 적힌 숫자와 그대로 같다(테스트가 계약 파일에서 숫자를 읽어 대조한다).

좌표 우선순위는 계약대로 `somaLocation` → 시냅스 중심(`neuron-roi-v1.parquet`의 `cx,cy,cz`) → 없으면 **NaN**이다.
0으로 채우지 않는다. 복셀 값을 그대로 쓰며 재중심화·단위 변환은 하지 않는다.

**순서 검증**: `set` 배열이 `sets.region_of`와 같은 벡터이고, 실제로 쓰여 있던 `runs/live/coord-e2e/run.json`과
대조해 ① `regions` 배열이 `sets`와 같고 ② `neurons.region`이 `set`과 완전히 같으며 ③ soma 출처 뉴런의
`neurons.x/y`가 `pos`의 x, y와 같음을 확인했다. 즉 뷰어가 스파이크의 뉴런 인덱스로 `pos`를 그대로 찾을 수 있다.

## 2. 엔드포인트

`flysim/live/static.py`의 허용 목록에 추가했다. 고정 경로는 딕셔너리, 묶음 이름만 동적이며
`^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$`에 맞고 **해석된 파일이 `data/cache` 바로 아래**여야 한다.

돌고 있는 서버에 실제로 요청한 결과(창 A가 만든 묶음 3개가 이미 디스크에 있었다):

| 경로 | 상태 | 크기 | Content-Type |
|---|---|---|---|
| `/neurons-3d.json` | 200 | 612 B | application/json |
| `/neurons-3d.bin` | 200 | 2,311,708 B | application/octet-stream |
| `/skel/index.json` | 200 | 205 B | application/json |
| `/skel/pC1.json` | 200 | 27,638 B | application/json |
| `/skel/pC1.bin` | 200 | 1,736,688 B | application/octet-stream |
| `/flysim-3d.html` | 200 | 65,902 B | text/html |
| `/neurons-v1.json`, `/flysim-viewer.html` | 200 | (기존 그대로) | |
| `/skel/../../etc/passwd`, `/../etc/passwd`, `/skel/a/b.json`, `/skel/.hidden.json`, `/skel/pC1.txt`, `/nope` | **404** | | |

`/skel/index.json`은 파일이 아니라 목록이라 항상 200이다(묶음이 없으면 `[]`). 지금 내용:

```json
[{"name":"AMMCtype","n_neurons":168,"n_segments":48816,"bytes":1171584},
 {"name":"JO_AB","n_neurons":114,"n_segments":10667,"bytes":256008},
 {"name":"pC1","n_neurons":156,"n_segments":72362,"bytes":1736688}]
```

`.json`은 있는데 `.bin`이 없거나 `.json`이 반쯤 쓰인 묶음은 목록에 `error`를 달고 나오되
`hello.assets.skeleton_sets`에서는 빠진다(완성된 묶음만 광고한다).

## 3. 프로토콜 v1.2

`hello`에 `assets:{neurons_3d:bool, skeleton_sets:[name…]}`만 추가했다. 기존 필드는 그대로다. 실제 응답:

```json
"assets": {"neurons_3d": true, "skeleton_sets": ["AMMCtype", "JO_AB", "pC1"]}
```

접속마다 디스크를 다시 읽으므로, 서버가 도는 중에 창 A가 묶음을 더 만들면 다음 접속부터 보인다.

**계약 문서의 "계약 버전" 줄은 v1.1 그대로 두었다.** `tests/test_live_html.py::test_contract_version_is_the_one_implemented`가
그 줄을 문자열로 확인하며 주석이 "this cockpit implements contract v1.1"이다. 즉 그 줄은 조종석의 구현 버전을
가리키고 조종석 창 소유다. 올렸더니 그 테스트가 깨져서 되돌렸고, 브리프가 요청한 두 가지(필드 추가, 개정 이력 한 줄)만
반영했다. 개정 이력에 "조종석이 `assets`를 읽기 시작하면 맨 위 줄도 v1.2로 올린다"고 적어 두었다.
**총괄이 정할 일**: 그 줄을 누가 언제 올릴지(조종석 창이 `assets`를 읽을 때가 자연스럽다).

## 4. 테스트

`pytest tests/` → **243 passed**(작업 시작 시 198개, 내가 27개 추가, 나머지는 다른 창이 같은 기간에 추가).

- `tests/test_export3d.py` 11개: 출처 우선순위와 NaN, 어긋난 ROI 테이블 거부, 오프셋·크기·왕복 읽기, JSON 키,
  집합 256개 초과 거부, 전부 좌표 없음 거부, 묶음 목록·자산 판정, 실파일의 개수·graph 대조, 실파일 순서 == `run.json`,
  계약 문서에 적힌 오프셋과 대조. 앞 7개는 합성 그래프라 커넥톰 캐시 없이 돈다.
- `tests/test_live.py` +16개: 파일 생성 전 404, 생성 후 200과 바이트 일치, `/skel/index.json` 목록, 경로 탈출 11종 404,
  기존 경로 유지, `hello.assets`가 디스크를 반영, `validate_assets` 거부. 임시 캐시 디렉터리에 진짜 HTTP 서버를
  띄워 확인한다.
- `tests/test_live_html.py`(조종석 창 소유) 46개는 계속 통과한다.

## 5. Deviations

1. **`/flysim-3d.html`을 고정 경로에 추가**했다. 계약의 엔드포인트 표에는 데이터 경로만 있지만, 3D 뷰어 페이지도
   같은 서버에서 열려야 쓸 수 있고(`/flysim-live.html`, `/flysim-viewer.html`과 같은 취급) 창 C는 `flysim/` 아래를
   건드리지 않는다. 파일이 없으면 404다.
2. **계약 버전 줄 미변경** — §3. 브리프가 요청한 범위를 넘어서 고치지 않았고 깨진 테스트를 우회하지도 않았다.
3. `Cache-Control: no-store`를 3D 자산에도 그대로 적용했다. 2.3 MB와 1.7 MB를 새로고침마다 다시 받는다는 뜻이다.
   로컬에서는 문제가 없어 기존 정책을 유지했고, 필요하면 ETag를 붙이는 게 다음 수순이다.
4. `skeleton_bundles`/`available_assets`/`resolve_skel`의 캐시 디렉터리를 인자로 받게 했다(기본값은 모듈 전역).
   테스트가 임시 디렉터리를 가리킬 수 있게 하려는 것이고 동작은 같다.
5. `flysim/live/server.py`는 수정할 것이 없었다(hello는 `session.py`가 만든다).
6. `hello.assets`는 접속마다 디스크를 훑는다(글롭 한 번). 캐시하지 않는 쪽이 "실제 파일 유무를 반영"에 맞다.
7. **`tests/test_live.py::test_real_graph_end_to_end`의 벽시계 단언을 1,000 → 5,000 μs/빈으로 완화**했다.
   M6b와 무관한 내 M5a 코드인데, 세 창이 한 노트북을 나눠 쓰면서(작업 중 load average 6.5) 다른 창이 바쁠 때마다
   실패한다. 단독 실행은 그대로 통과한다(빈당 < 1,000 μs). 성능 수치의 출처는 `python -m flysim.live.bench`이고
   (`docs/m5a-report.md` §4), 이 단언은 "몇 배 느려졌다"만 잡는 회귀 가드로 남겼다. 총괄이 앞서 본 같은 테스트의
   실패도 같은 성질일 가능성이 높다.
