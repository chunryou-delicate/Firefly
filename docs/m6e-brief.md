# M6e 구현 지시서 — bodyId 배열 + 조종석이 3D를 인지

총괄 세션 작성 (2026-09-19).

> **보고 대상:** 총괄 세션. 이 지시를 전달한 메시지의 `from-name`을 쓴다.

먼저 읽을 것: `docs/m6-3d-contract.md`(v1.3), `docs/m5-protocol.md`(v1.2 및 맨 위 "구현 현황"), `flysim/probe/export3d.py`(네가 만든 것), `flysim-live.html`, `tests/test_live_html.py`, `docs/backlog.md`의 M6 항목.

의존성: 없음.

## 할 일 1 — 점구름에 `body` 배열 (계약 v1.3)
`flysim/probe/export3d.py`가 `body`(uint32, 길이 n, bodyId)를 추가로 낸다. 관측 최대 bodyId가 1,571,825,087이라 uint32에 들어간다. **넘는 값이 있으면 조용히 자르지 말고 예외.**
`arrays`에 오프셋·길이를 넣고, 기존 세 배열의 위치·의미는 그대로 둔다. 파일이 약 0.66 MB 커진다.

## 할 일 2 — 조종석이 `hello.assets`를 읽는다
`flysim-live.html`이 계약 v1.2를 구현한다.
- `hello.assets.neurons_3d`와 `skeleton_sets`를 받아 상단에 3D 자산 유무를 표시한다.
- **3D 화면으로 넘어가는 버튼**을 단다. 같은 서버의 `/flysim-3d.html`을 새 탭으로 열되, 실시간 모드로 붙도록 웹소켓 주소를 쿼리로 넘긴다(예: `?ws=<host>:8765`). 3D 화면 쪽 구현은 다른 창이 하고 있으니 **쿼리 이름만 이 문서대로 맞춘다**: `ws`.
- 파일 맨 위 주석의 선언 버전을 `v1.2`로 올린다. `tests/test_live_html.py`의 버전 테스트가 선언 버전과 계약 이력을 대조하므로 그대로 통과해야 한다.
- 조종석에 3D 렌더러를 넣지 마라. 링크만 건다.

## 완료 조건
- [ ] `neurons-3d.bin`에 `body` 추가, 인덱스 → bodyId가 `Graph.body()`와 일치(테스트)
- [ ] 조종석이 자산 유무를 표시하고 3D 버튼이 `?ws=`를 붙여 연다
- [ ] 조종석 선언 버전 v1.2, 버전 테스트 통과
- [ ] `pytest tests/` 전부 통과 (현재 273개)
- [ ] "M6e: ..." 커밋 하나. **push 금지**

## 파일 경계
`flysim/probe/export3d.py`, `flysim-live.html`, `tests/{test_export3d,test_live_html}.py`, `docs/m6e-report.md`만. `flysim-3d.html`과 `flysim/engine/`은 다른 창 소관이다.
