# M7d 1부 사전 등록 — 아타리 게임 연결과 비교 봇 (초파리는 아직 안 넣는다)

총괄 세션 작성 (2026-10-25). **의존성 `gymnasium`·`ale-py` 사용자 승인(2026-10-25).** 아타리 게임을 한 판도 돌리기 전에 쓴 문서다. 결과를 보고 고치지 않는다.

> **보고 대상:** 총괄 세션. 이 지시를 전달한 메시지의 `from-name`을 쓴다.

먼저 읽을 것: `docs/m7-game-ladder.md`(규칙 1~6), `docs/m7b-report.md`, `flysim/apps/games/base.py`(특히 `ALE_FULL_ACTION_INDEX`), `flysim/apps/games/agents.py`, `flysim/apps/games/runner.py`, `flysim/sensory/looming.py`.

초파리 뇌는 이 단계에 넣지 않는다. 해독기가 재대결(`docs/m7c2-brief.md`)에서 정해진 뒤 2부에서 넣는다. 이 단계는 **아타리 게임이 우리 틀에서 돌아가고, 같은 눈이 무엇을 보며, 비교 봇이 몇 점을 내는지**를 확정한다.

## 0. 설치와 라이선스 (먼저, 막히면 멈추고 보고)
- `.venv`에 `gymnasium`, `ale-py` 설치(uv). 설치된 정확한 버전을 `pyproject.toml`/`uv.lock`에 반영한다. 다른 패키지는 추가하지 않는다(화면 축소도 numpy로 한다).
- `THIRD_PARTY.md`에 두 패키지의 라이선스를 적는다. **롬:** `ale-py` 패키지 안에 롬이 들어 있는지, 그 배포 근거가 패키지의 어느 파일에 어떻게 적혀 있는지 찾아 경로와 문구를 인용한다. **근거 문구를 찾지 못하면 게임을 돌리지 말고 보고한다**(사다리 규칙 6).

## 1. 연결 방식 (실행 전 확정)
- `gym.make("ALE/<이름>-v5", obs_type="grayscale", frameskip=1, repeat_action_probability=0.0, full_action_space=True)`.
  - `frameskip=1`: 아타리 한 화면 = 우리 한 틱(60 Hz). 뇌의 틱당 16/17 스텝과 그대로 맞는다.
  - `full_action_space=True`: 행동 번호가 모든 게임에서 같아야 `base.py`의 고정표 하나로 끝난다.
  - 반복 행동 확률 0: 같은 시드면 같은 판(결정론 테스트).
- **행동은 `base.ALE_FULL_ACTION_INDEX` 그대로** — 아무것도 안 함 = NOOP(0), 점프 = UP(2), 오른쪽 = RIGHT(3), 왼쪽 = LEFT(4). 이 표는 M7b에서 고정됐고 바꾸지 않는다. 그래서 **발사(FIRE) 버튼은 초파리 근육에 없다.** 발사가 필요한 게임에서 모든 봇이 못하는 것은 그대로 결과다.
- 화면: 210×160 회색조 → 84×84. **면적 평균**(행·열 방향 겹침 비율 행렬 두 개를 곱함, 정규화). 반올림해 uint8. 다른 전처리(잘라내기, 색 반전, 프레임 최대값 합성)는 하지 않는다 — 같은 눈 규칙.
- 판 길이: 게임 오버 또는 **3,600틱(게임 시간 60 s)** 중 먼저. 점수 = 그 안의 보상 합. 공개 점수(대개 5분, 프레임 건너뛰기 4)와 조건이 다르다는 것을 표에 적는다.
- `flysim/apps/games/atari.py`: 위 설정으로 `GameEnv`와 같은 호출(`reset(seed) → frame`, `step(action) → frame, reward, done, info`)을 갖는 감싸개. 에이전트 코드는 고치지 않는다.

## 2. 후보 게임 (이 목록 밖은 없다, 점수로 고르지 않는다)
Pong, Breakout, SpaceInvaders, Freeway, Kaboom, DemonAttack, Skiing, Boxing — 다가오는 물체가 있는 게임 위주로 실행 전에 고른 8개.
`ale-py`에 없는 게임은 "없음"으로 기록하고 뺀다. 나머지는 전부 돌린다.

### 2.1 게임 조사표 (봇을 돌리기 전에 먼저 만든다)
게임마다: 최소 행동 집합의 이름 목록, 우리 네 행동(NOOP/UP/RIGHT/LEFT)이 그 집합에 들어 있는지, 즉 **초파리 근육이 이 게임에서 실제로 움직일 수 있는 것**. 하나도 없으면 "근육이 닿지 않음"으로 표시하되 봇은 그대로 돌린다(점수가 0이면 0이 결과다).

### 2.2 같은 눈이 보는 것 (기록, 선택에 안 씀)
게임마다 학습 시드 0~9에서 무작위 봇이 플레이한 프레임에 감지기를 돌려: 0이 아닌 `loom` 비율, 좌우 평균, 99번째 백분위수. 감지기는 "어두운 영역이 늘어나는 것"을 재므로 배경이 검은 게임에서 무엇을 보는지가 여기서 드러난다.

## 3. 비교 봇 (M7b·M7c와 같은 절차)
- 무작위 봇(우리 네 행동 균일), 한 방향 봇(항상 왼쪽, 항상 오른쪽 — 높은 쪽이 기준), 뇌 없는 봇(같은 한 줄 규칙, `THETA_GRID`에서 학습 시드 0~9로 θ 선택, 평가 시드 10~19).
- `runner.py`의 함수들을 가져다 쓰고 고치지 않는다. 결과는 `data-provenance/m7d-baselines.json`(시드별 점수 포함).
- **공개 인간·무작위 점수**는 원 논문 표에서 출처(논문, 표 번호)를 달아 인용만 한다. 원문을 직접 확인할 수 없으면 비워 둔다. 기억으로 숫자를 쓰지 않는다.

## 4. 보이는 결과
- `docs/m7d-survey.png`: 게임마다 한 줄, 뇌 없는 봇 평가 시드 10의 84×84 화면 8장(1 s 간격)과 그 시각의 `loom_L`/`loom_R` 막대. 박사님이 "초파리 눈에 이 게임이 이렇게 보인다"를 한눈에 볼 수 있게.
- `docs/m7d-report.md`: 조사표, 감지기 표, 점수표(무작위 / 한 방향 / 뇌 없는 봇 / 공개 점수), 판 길이 분포.

## 5. 산출물·완료 조건
- [ ] 설치, `THIRD_PARTY.md`(롬 배포 근거 인용 포함)
- [ ] `flysim/apps/games/atari.py`, `flysim/apps/games/m7d.py`(실행 스크립트), `tests/test_atari.py`(결정론, 84×84 uint8, 행동 표가 `base.py` 것인지, 3,600틱 상한)
- [ ] `data-provenance/m7d-baselines.json`, `docs/m7d-report.md`, `docs/m7d-survey.png`
- [ ] `pytest tests/` 전부 통과
- [ ] "M7d: ..." 커밋 하나. push 금지

## 하지 말 것
- 게임별 전처리·행동표·감지기 설정, 후보 밖 게임 추가, 점수 보고 게임 빼기
- `flysim/apps/games/` 기존 파일(`base.py`, `agents.py`, `runner.py`, `brain.py`, `m7c.py` 등), `flysim/probe/`, `flysim/engine/`, `flysim-*.html` 수정 — 다른 창이 재대결 작업 중이다
- 초파리 뇌 연결(2부)

## 시간
뇌가 없어서 빠르다. 뇌 없는 봇의 θ 격자 41점 × 학습 10판 × 8게임이 가장 크다(판당 최대 3,600틱). 한 게임을 재서 전체 예상 시간을 총괄에 먼저 알린다.
