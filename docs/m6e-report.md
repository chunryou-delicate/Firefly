# M6e report — 점구름 `body` 배열 + 조종석이 3D 자산을 인지

구현 세션(서버 창), 2026-09-19. 사양: `docs/m6e-brief.md`. 계약: `docs/m6-3d-contract.md` v1.3,
`docs/m5-protocol.md` v1.2(둘 다 수정하지 않았다).

## 1. `body` 배열 (계약 v1.3)

`python -m flysim.probe.export3d --force` → 2.0초.

| 항목 | 이전(v1.2) | 지금(v1.3) |
|---|---|---|
| `.bin` 크기 | 2,311,708 B (2.31 MB) | **2,972,196 B (2.97 MB)** (+660,488 B = +0.66 MB) |
| 배열 | pos / set / src | pos / set / src / **body** |
| `body` 위치 | — | offset 2,311,708 · length 165,122 · uint32 |

기존 세 배열의 오프셋(0 / 1,981,464 / 2,146,586)과 의미는 그대로다. `body`는 맨 뒤에 붙였다.

**검증**:

- `body`를 `Graph.body_ids`와 통째로 비교해 완전히 같음을 확인했다(`np.array_equal`, 165,122개 전부).
- 인덱스 세 곳(0, 1234, n−1)에서 `body[i] == Graph.body(i)`.
- 유지 뉴런의 실제 최대 bodyId는 **1,471,062,202**이다. 총괄이 말한 1,571,825,087보다 작은데, 그 값은
  어노테이션 전체 기준이고 이쪽은 `status == "Traced"`로 남은 165,122개 기준이라서다. 어느 쪽이든 uint32
  (최대 4,294,967,295) 안에 들어간다.
- **범위를 넘으면 예외**다. `body_ids_uint32()`가 int64로 먼저 비교해 음수·2³² 이상·정수 아닌 값을 거부하고,
  "자르지 말고 dtype을 바꿔야 한다"는 메시지를 낸다. 잘라서 쓰는 경로는 없다. 테스트가 `UINT32_MAX+1`, `-1`,
  `2**33`, 소수, 2차원 배열을 각각 확인한다.

읽는 쪽은 `read_arrays()`가 `.json`의 `arrays`에 있는 것만 읽으므로 `body`가 없는 옛 파일도 그대로 열린다
(계약 v1.3의 "소비자는 옛 파일도 견뎌야 한다").

## 2. 조종석이 `hello.assets`를 읽는다 (계약 v1.2)

`flysim-live.html`의 선언 버전을 **v1.2**로 올렸다. 화면 변화는 두 가지뿐이고 3D 렌더러는 넣지 않았다.

**상단 배지 `assetsBadge`** — `hello.assets`를 그대로 보여준다.

| 서버 상태 | 배지 | 툴팁 |
|---|---|---|
| 점구름 있음 | `3D 점구름 · 골격 **4**` | 골격 묶음: AMMCtype, JO_AB, WED, pC1 |
| 점구름 없음 | `3D 자산 없음` | neurons-3d 파일이 없다 — `python -m flysim.probe.export3d` 로 만든다 |
| `assets` 미수신 | `3D 자산 미보고` | 계약 v1.2 이전 서버 |
| 접속 전 | `3D —` | |

**버튼 `open3dBtn` (`3D ↗`)** — 같은 서버의 `/flysim-3d.html`을 새 탭(`_blank`, `noopener`)으로 연다.
웹소켓 주소를 쿼리 `ws`로 넘긴다: `/flysim-3d.html?ws=127.0.0.1%3A8765`. 값은 브리프의 예시대로
`<host>:<port>` 형식이고(스킴·경로 없음), 조종석이 실제로 붙어 있는 주소(`S.url`)의 authority를 그대로 쓴다.
`encodeURIComponent`를 거치므로 콜론이 `%3A`로 인코딩된다 — 3D 창이 `URLSearchParams.get("ws")`로 읽으면
`127.0.0.1:8765`로 복원된다.

버튼은 **점구름이 없거나 모의 모드면 비활성**이다. 모의 모드는 붙을 서버가 없어서 링크가 의미가 없고,
그 사실을 툴팁에 적는다. 접속 전에도 비활성으로 시작한다.

라이브 확인(다른 창이 띄워 둔 서버에 붙어서 했다 — 정적 파일은 요청마다 디스크에서 읽으므로 유효하다):

- `/flysim-live.html`이 새 요소(assetsBadge·open3dBtn·applyAssets)를 포함해 나온다.
- 버튼이 만드는 URL `/flysim-3d.html?ws=127.0.0.1%3A8765` → 200, 98,144 B.
- `/neurons-3d.json`의 `arrays`에 `body`가 보이고 `/neurons-3d.bin`이 2,972,196 B로 내려온다.
- `hello.assets` = `{"neurons_3d": true, "skeleton_sets": ["AMMCtype","JO_AB","WED","pC1"]}`.

## 3. 테스트

- `tests/test_export3d.py` 12개(+1): `body` 오프셋·dtype·왕복, uint32 범위 거부 5종, 실파일의
  `body == Graph.body_ids`와 `Graph.body(i)` 대조.
- `tests/test_live_html.py` 51개(+5): 선언 버전 v1.2, 배지·버튼 존재와 초기 비활성, `onHello`가
  `applyAssets(msg.assets)`를 부르는지, `applyAssets`가 `neurons_3d`·`skeleton_sets`·`S.mock`을 보고
  버튼을 켜고 끄는지, `open3dUrl`이 `/flysim-3d.html`에 `?ws=`를 붙이는지, **조종석에 3D 렌더러가 없는지**
  (`webgl`, `createShader`, `drawArrays`, `THREE.` 등이 스크립트에 없어야 한다).
- 기존 버전 테스트(`test_contract_version_is_declared_and_known`)는 그대로 통과한다: 선언 v1.2가 계약 이력에
  있고 계약(v1.2)이 조종석보다 오래되지 않았다.

내 파일 4개(`test_export3d`, `test_live_html`, `test_live`, `test_3d_html`) **160 passed**.

전체 `pytest tests/`는 **300 passed, 3 failed**인데 실패 3건은 전부 `tests/test_engine_adapt_g.py`다.
엔진 창이 지금 만들고 있는 미커밋 파일이고(`flysim/engine/{lif,params,sweep_conductance}.py`도 작업 중),
내 파일 경계 밖이라 손대지 않았다.

## 4. Deviations

1. **`docs/m5-protocol.md`의 "구현 현황" 줄이 낡았다.** 지금 "조종석은 v1.1이고 `hello.assets`를 아직 읽지
   않는다"라고 적혀 있는데 이번 작업으로 둘 다 틀렸다. 그 파일은 이번 브리프의 파일 경계 밖이라 고치지 않았다.
   **총괄이 고쳐야 한다**: "서버 v1.2 · 조종석 v1.2, 조종석이 `hello.assets`를 읽는다" 정도.
2. `?ws=` 값은 `<host>:<port>`(브리프 예시 형식)이고 `encodeURIComponent`로 인코딩된다. 3D 창이
   `URLSearchParams`로 읽으면 문제가 없지만, 만약 전체 URL(`ws://host:port/ws`)을 기대한다면 형식을 맞춰야
   하므로 3D 창과 교차 확인이 필요하다. 조종석 자신의 `?ws=` 파라미터는 전체 URL을 받는 규칙이라 두 창이
   같은 이름을 다르게 해석할 여지가 있다.
3. 모의 모드에서 3D 버튼을 비활성으로 두었다. 계약에 규정이 없는 부분인데, 모의 서버는 파일을 서빙하지 않아
   링크가 404가 되기 때문이다.
4. 실측 최대 bodyId(1,471,062,202)가 계약에 적힌 관측값(1,571,825,087)과 다르다. 유지 그래프 기준과
   어노테이션 전체 기준의 차이로 보이며, 어느 쪽도 uint32를 넘지 않는다. 계약 문구를 고치지는 않았다.
5. 라이브 확인은 다른 창이 띄워 둔 서버(포트 8080/8765 점유)로 했다. 내 서버를 띄우려다 `Address already in use`가
   났고, 남의 프로세스를 죽이지 않았다. 정적 파일과 점구름은 요청마다 디스크에서 읽히므로 확인 결과는 유효하다.
