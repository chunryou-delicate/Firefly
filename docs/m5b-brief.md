# M5b 구현 지시서 — flysim-live 조종석 (브라우저 클라이언트)

총괄 세션 작성 (2026-09-16). 대상: 새 세션. **`docs/m5-protocol.md`가 계약이다.** 먼저 읽을 것: 계약서, `flysim-viewer.html` 전체(시각 언어와 그리기 방식을 이어받는다), `CLAUDE.md` §5.5,

> **보고 대상:** 총괄 세션. 세션 이름은 재시작 때마다 바뀌므로 **이 지시를 전달한 메시지의 `from-name`**(또는 사용자가 알려준 이름)을 그대로 쓴다. 문서에 적힌 옛 이름은 무효다. 이름을 모르면 `ListAgents`로 `fruit-fly-*` 세션을 확인하고, 애매하면 사용자에게 묻는다.

`docs/backlog.md`("Viewer normalisation" 항목), `docs/m3-report.md`(집합 이름).

의존성: 없음. **단일 HTML 파일 `flysim-live.html`**, 외부 스크립트·CSS 없음(폰트는 뷰어와 같은 Google Fonts 링크 허용). 빌드 없음.
M5a 서버가 아직 없을 수 있으므로 **내장 모의 서버 모드**(`?mock=1`: 계약서 형식의 가짜 `hello`/`frame`을 브라우저 안에서 생성)로 개발·확인한다.

## 위치와 관계
- 기존 `flysim-viewer.html`은 **결과 재생기**로 그대로 둔다(수정 금지). `flysim-live.html`은 **조종석**이며 별개 파일이다. 색 규칙(차가운 색=입력, 따뜻한 색=출력), 폰트, 레이아웃 리듬, 실패 경고 문구를 그대로 잇는다. 뉴런 필드의 ImageData 가산 합성 방식은 가져다 써도 된다(복사 허용, 출처 주석).
- 저장소 루트에 둔다. 서버는 `http://host:8080/flysim-live.html`로 서빙한다.

## 화면
1. **상단 바**: 연결 상태(host, `params_version`, 상태 running/paused/lagging/disconnected), 모드(rate/phase-lock), dt, speed, 안전 경고(뷰어와 같은 문구 + `ignited`).
2. **자극 패널**(왼쪽): 종류 선택(silence / click_train / tone / audio), IPI·반송주파수·펄스 길이·ILD 슬라이더, **오디오**: 마이크(getUserMedia) 또는 파일(`<input type=file>`→ AudioContext 디코드) → ScriptProcessor/AudioWorklet로 20 ms float32 청크를 바이너리 프레임으로 전송, 입력 레벨 미터.
3. **파라미터 패널**: g(로그 슬라이더), a_in(로그), noise_sigma, synapse 토글, adapt_b(서버가 거절하면 비활성), speed, dt. 변경 → `set_params` → `ack`로 동기화. 각 변경이 로그됨을 UI에 표시("모든 조작은 기록됨").
4. **뉴런 필드**(중앙, 뷰어 방식): 실시간 스파이크 가산 합성 + 잔광. 클릭 → 가장 가까운 뉴런 → `get_hops k=2` → 하류 층을 색으로 강조(토글), Shift+클릭 → `inject`(amp·duration은 패널 값).
5. **영역 히트맵/타임라인**(하단): 최근 5 s 롤링. **정규화 선택**: 전역 최대 / 영역별 최대 / 로그 스케일 / 고정 Hz 상한(입력). 기본은 **영역별 최대**(백로그 문제 해결). JO 집합은 별도 행 그룹으로 분리 표시.
6. **좌우 창**: 집합별 L−R 차이(Hz)를 부호 있는 발산 색으로. 값이 0인 집합은 회색.
7. **스냅샷 버튼**: `snapshot {run_id, seconds}` → 완료 시 경로 표시. 결과 제출용은 아니라는 문구.

## 동작 규칙
- 서버 원시 Hz를 받아 정규화는 전부 클라이언트. 낙관적 갱신 금지(ack 후 반영).
- 재접속 2 s. 끊기면 마지막 프레임 위 "disconnected".
- 프레임이 몰리면(60 fps 초과) 렌더는 requestAnimationFrame당 최신 프레임만, 히트맵은 전부 누적.
- 400 px 폭에서도 깨지지 않게(패널 접힘).

## 테스트·검증 (브라우저 자동화 없음)
- `tests/test_live_html.py`: 파싱, 외부 스크립트 0, 계약서의 모든 서버→클라이언트 `type` 문자열이 코드에 처리 분기로 존재, 모든 클라이언트→서버 `type`이 전송 코드에 존재, `?mock=1` 분기 존재.
- 모의 모드에서의 동작은 리포트에 텍스트로 기술(무엇을 눌렀을 때 무엇이 보이는지). 실제 서버 연동 확인은 M5a 완료 후 총괄이 한다.

## 산출물·완료 조건
- `flysim-live.html`, `tests/test_live_html.py`, `docs/m5b-report.md`(화면 설명, 정규화 방식, 오디오 경로, 한계).
- "M5b: ..." 커밋 하나. 다른 파일 수정 금지(특히 `flysim-viewer.html`, `docs/m5-protocol.md`). 계약이 부족하면 총괄 세션에 보고.
- 끝나면 총괄 세션에 커밋 해시, 테스트, 구현된 패널 목록, 미구현 항목을 보고.
