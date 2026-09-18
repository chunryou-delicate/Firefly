# M6b 구현 지시서 — 뉴런 점구름 + 3D 자산 서빙

총괄 세션 작성 (2026-09-18).

> **보고 대상:** 총괄 세션. 세션 이름은 재시작마다 바뀌므로 이 지시를 전달한 메시지의 `from-name`을 쓴다.

먼저 읽을 것: `CLAUDE.md`, **`docs/m6-3d-contract.md`(계약, 수정 금지)**, `flysim/probe/export.py`(좌표·집합 처리 방식), `flysim/probe/sets.py`, `flysim/live/static.py`, `flysim/live/server.py`, `docs/m5-protocol.md`.

의존성: 기존 그대로. 새 패키지 금지.

## 할 일 1 — 점구름 생산
`flysim/probe/export3d.py`. 계약의 파일 1을 만든다.

- 좌표 우선순위: `somaLocation` → 없으면 `data/cache/neuron-roi-v1.parquet`의 시냅스 중심 → 둘 다 없으면 NaN + `src=2`.
  총괄이 확인한 개수: soma 140,024 / 유지 뉴런 165,122. 나머지는 중심 좌표로 채워지고 333개는 시냅스 사이트가 아예 없다. 실제 개수를 `src_counts`에 기록한다.
- 순서는 반드시 **그래프 뉴런 인덱스 순**. run.json의 뉴런 인덱스와 같아야 한다. 테스트로 강제한다.
- CLI: `python -m flysim.probe.export3d`.

## 할 일 2 — 서빙
`flysim/live/static.py`에 계약의 엔드포인트를 **허용 목록 방식 그대로** 추가한다. 경로 탈출은 지금처럼 404여야 하고, 기존 테스트가 깨지면 안 된다.
`/skel/index.json`은 `data/cache/skel-*.json`을 훑어 목록을 만든다. 파일이 없으면 빈 배열이지 오류가 아니다.

## 할 일 3 — 프로토콜 v1.2
`docs/m5-protocol.md`에 **추가만** 한다: `hello`에 `assets:{neurons_3d:bool, skeleton_sets:[name…]}`. 개정 이력에 v1.2 한 줄.
`flysim/live/protocol.py`와 `session.py`의 hello 생성부를 맞춘다. **기존 메시지·필드는 건드리지 마라.** 조종석의 계약 대조 테스트(`tests/test_live_html.py`)가 계약 파일을 직접 읽으므로, 네 변경이 그 테스트를 깨면 안 된다. 깨면 멈추고 보고.

## 완료 조건
- [ ] `neurons-3d.bin`/`.json` 생성. 크기 약 2.3 MB. `src_counts` 합이 165,122
- [ ] NaN 좌표 뉴런이 `src=2`와 개수 일치
- [ ] 서버 기동 후 네 엔드포인트가 200/404를 계약대로 반환
- [ ] `hello.assets`가 실제 파일 유무를 반영
- [ ] `tests/test_export3d.py` + `tests/test_live.py` 보강: 인덱스 순서가 run.json과 같음, 배열 오프셋·길이 정확, 엔드포인트 허용 목록, 경로 탈출 404
- [ ] `pytest tests/` 전부 통과 (현재 198개)
- [ ] "M6b: ..." 커밋 하나. **push 금지**

## 파일 경계
너는 `flysim/probe/export3d.py`, `flysim/live/{static,protocol,session,server}.py`(3D 관련 부분만), `tests/test_export3d.py`, `tests/test_live.py`, `docs/m5-protocol.md`(추가만), `docs/m6b-report.md`만 만진다.
`flysim/data/skeletons.py`(다른 창), `flysim-*.html`(다른 창), `flysim/engine/`은 건드리지 않는다.
