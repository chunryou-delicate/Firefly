# M7 준비 — 프로브 프로필과 탈출 회로 골격

총괄 세션 작성 (2026-10-23). `docs/m7-game-ladder.md`의 "M7e 준비 결정" 1·2번 구현.

> **보고 대상:** 총괄 세션. 이 지시를 전달한 메시지의 `from-name`을 쓴다.

먼저 읽을 것: `docs/m7-game-ladder.md`(M7e 준비 결정, M7a 결과), `flysim/probe/sets.py`, `flysim/probe/export.py`, `flysim/data/skeletons.py`, `docs/m6-3d-contract.md`.

## 할 일
1. **`flysim/probe/sets.py`에 프로필.** 지금 집합 정의를 이름 붙은 프로필로 감싼다.
   - `auditory` — 지금 것 그대로. **기본값.** 기존 함수 호출부는 한 줄도 바뀌지 않아야 하고 결과가 bit-identical이어야 한다(회귀 테스트: 집합별 인덱스 배열이 프로필 도입 전과 같음).
   - `escape` — `LC4_L`, `LC4_R`, `LPLC2_L`, `LPLC2_R`, `DNp01_L`, `DNp01_R`, `TTMn_L`, `TTMn_R`, `rest`. 좌우는 M7a와 같은 규칙. 집합 크기가 M7a(71/55, 94/91, 1/1, 1/1)와 같아야 한다(테스트). 0건이면 예외.
   - 함수 시그니처는 `profile: str = "auditory"` 인자를 **추가만** 한다. 기존 인자·반환형 유지.
2. **`flysim/data/skeletons.py --profile escape`.** 그 프로필의 집합(rest 제외)을 묶음으로 낸다. 좌우는 합쳐 `LC4`, `LPLC2`, `DNp01`, `TTMn` 네 묶음. 계약 v1.3 그대로(허용오차 40, 20 MB 상한, 뉴런당 상한 없음). 기존 `--set` 동작 유지.
3. `export3d.py`는 **건드리지 않는다**(결정 1에서 제외됨).

## 완료 조건
- [ ] `auditory` 회귀 bit-identical, 기존 테스트 전부 통과
- [ ] `escape` 집합 크기 테스트
- [ ] 네 묶음 생성, 각 크기·선분 수·missing 보고
- [ ] `pytest tests/` 전부 통과 (현재 349)
- [ ] "M7: probe profiles ..." 커밋 하나. **push 금지**

## 파일 경계
`flysim/probe/sets.py`, `flysim/data/skeletons.py`, `tests/test_probe.py`(프로필 테스트 추가), `tests/test_skeletons.py`, `docs/m7-profiles-report.md`만.
서버 창이 지금 `flysim/apps/games/`, `flysim/sensory/looming.py`, `flysim/probe/motor.py`를 작업 중이다. 그 파일들은 건드리지 않는다.
