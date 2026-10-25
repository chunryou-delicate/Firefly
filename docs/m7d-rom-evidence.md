# M7d 롬 배포 근거 조사 — 사다리 규칙 6 미충족, 사용자 판단 대기

조종석 창 작성 (2026-10-25). 아래 원문은 모두 **2026-10-25에 조회**했다. 인용문은 원문 그대로 옮겼다(영어 원문의 오타 포함).

## 질문과 인정 기준

사다리 규칙 6(`docs/m7-game-ladder.md`): "직접 만든 게임, 또는 정식으로 배포가 허락된 롬만 쓴다."
`docs/m7d-brief.md` §0: 롬 배포 근거를 찾지 못하면 게임을 돌리지 않고 보고한다.

인정 기준은 총괄 세션이 조사 전에 정했다.
- **근거로 인정:** 권리자에게서 배포 허가나 라이선스를 받았다고 명시한 문구, 또는 해당 롬의 저자가 공개했다고 명시한 문구.
- **근거가 아님:** "사용자가 책임진다", "동봉돼 있다", "연구용으로 쓰인다" 같은 문구.
- 찾을 곳: 설치된 `ale-py` 패키지(1차), Farama의 ALE 공식 문서(ale.farama.org), Arcade-Learning-Environment 저장소의 README·docs·이슈·릴리스 공지, AutoROM 저장소와 PyPI 패키지의 LICENSE·README.

## 결론

**후보 8개(Pong, Breakout, SpaceInvaders, Freeway, Kaboom, DemonAttack, Skiing, Boxing) 모두 인정할 근거 문구가 없다.** 게임별 차이도 없다. 오히려 이 롬들의 배포가 허락되지 않았다고 말하는 Farama 측 발언이 두 건 있다(§2). 근거가 적힌 롬은 후보 밖의 테트리스 하나뿐이다(§5).

→ 규칙 6 미충족. **사용자 판단 대기.** 그때까지 롬을 에뮬레이터에 불러오지 않고, 환경도 만들지 않는다(§7).

총괄 세션이 §2의 세 인용(AutoROM #7, ALE #513의 두 댓글)을 GitHub API로 따로 조회해 원문과 일치함을 확인했다.

## 1. 설치된 패키지 (`ale-py` 0.12.1, `.venv/lib/python3.10/site-packages/`)

- 롬이 들어 있다: `ale_py/roms/`에 `.bin` 108개, `md5.json` 108항목. 그 밖의 파일은 `__init__.py`와 `__pycache__`뿐이다.
- 롬에 관한 문구는 `ale_py-0.12.1.dist-info/METADATA`의 두 줄뿐이다. 둘 다 동봉 사실만 말한다.
  - 66행: "- Atari roms are packaged within the pip package."
  - 101행: "For simplicity for installing ale-py with Gymnasium, `pip install "gymnasium[atari]"` shall install all necessary modules and ROMs."
- 라이선스 파일은 `ale_py-0.12.1.dist-info/licenses/LICENSE.md` 하나이고 GNU GPL v2 전문이다(METADATA `License-Expression: GPL-2.0-only`). 에뮬레이터 코드의 라이선스이고 롬은 언급하지 않는다.
- 근거 문구를 찾았으나 없었던 곳: `ale_py/roms/__init__.py`(해시 확인 코드뿐), `RECORD`, `sboms/auditwheel.cdx.json`(롬 항목 0개), `_ale_py.cpython-310-x86_64-linux-gnu.so`의 문자열("licen", "permission", "AutoROM", "redistribut", "Atari, Inc", "Atari Interactive"로 검색).

## 2. 반대 증거 — 배포가 허락되지 않았다는 발언

**AutoROM 이슈 #7, jkterry1, 2021-09-16.** 작성자는 AutoROM 저장소에 LICENSE를 추가한 PR #1(2020-03-21)의 작성자다.
https://github.com/Farama-Foundation/AutoROM/issues/7#issuecomment-920895605

> It is not legal for us to distribute ROMs, which is why Atari-Py (now ALE-Py) do not ship them anymore. However, it is legal for us to have a library that automatically installs them from a 3rd party (even if that third parties hosting of them may or may not be legal) in the same way that torrent clients are legal.

**ALE PR #513 "Includes roms within build for pypi and testing", 2024-02.** 롬을 PyPI 패키지에 넣기로 한 논의다.

- pseudo-rnd-thoughts, 2024-02-23. PR #513과 #516을 병합하고 v0.9.0 릴리스를 올린 사람이다(GitHub API의 `merged_by`, 릴리스 `author`).
  https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/513#issuecomment-1961687629
  > On the accept-rom-license, to my knowledge, we never explicitly explain what rom license people are accepting.
- jjshoots, 2024-02-28. PR #513의 작성자다.
  https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/513#issuecomment-1969706478
  > Discussed with @pseudo-rnd-thoughts on Discord, we agreed that packaging the roms as a tar.gz file within the PyPI package and unpacking after is the most ideal way to skirt around this grey area.

**PR #513과 PR #516(2024-04-23 병합)이 모두 병합됐다.**
https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/516
#516 본문 원문은 "* Includes AutoROM roms within `pypi` (but not github)"이다. 본문과 댓글 어디에도 허가나 라이선스 언급이 없다. 그 뒤 v0.9.0(2024-05-20)으로 릴리스됐다(§4).

## 3. 사용자 책임 문구 (기준상 근거가 아님)

- **AutoROM 동의 문구.** `packages/AutoROM/src/AutoROM.py` 288–292행, 커밋 5bb4474db00b5b6ab7510ca272546a14aa07fa6a.
  https://github.com/Farama-Foundation/AutoROM/blob/5bb4474db00b5b6ab7510ca272546a14aa07fa6a/packages/AutoROM/src/AutoROM.py#L288-L292
  > I own a license to these Atari 2600 ROMs.
  > I agree to not distribute these ROMs and wish to proceed:
- **예전 오류 메시지.** 사용자가 ALE 이슈 #564(2024-09-30) 본문에 붙인 것이다.
  https://github.com/Farama-Foundation/Arcade-Learning-Environment/issues/564
  > Note: Gymnasium no longer distributes ROMs. If you own a license to use the necessary ROMs for research purposes you can download them via `pip install gymnasium[accept-rom-license]`.
- **AutoROM 이슈 #49 (2024-04-22).** 이 질문은 지금까지 열려 있고 댓글이 0개다.
  https://github.com/Farama-Foundation/AutoROM/issues/49
  > I don't work for Atari. How can I answer yes without lying?

## 4. 근거가 없음을 확인한 곳

**ale.farama.org (조회한 사이트 페이지).**
- 페이지: https://ale.farama.org/ , /getting-started/ , /environments/ , /env-spec/ , /faq/ , /gymnasium-interface/ , /python-interface/
- 롬과 관련된 법적 문구가 없다. 저작권 문구는 사이트 자체의 "Copyright © 2023 Farama Foundation"뿐이다.

**ALE 저장소 `main` (커밋 816fd1591f1b73133c269fe753f10a0490ab4a63).**
- `README.md` 26행: "- Atari roms are packaged within the pip package." (METADATA와 같은 문장)
- `ACKNOWLEDGMENTS.md`: Stella와 ALE 코드의 GPL 고지뿐이다.
- `LICENSE.md`: GPL v2.
- `docs/` 전체.
- 후보 8개의 게임 문서 `docs/environments/{pong,breakout,space_invaders,freeway,kaboom,demon_attack,skiing,boxing}.md`: "licen/permission/copyright/distribut/rights"가 0건이다. 이 파일들은 사이트 게임별 페이지의 원본인데, 사이트의 게임별 페이지는 따로 받지 않았다.
- 게임 코드 `src/ale/games/supported/{Pong,Breakout,SpaceInvaders,Freeway,Kaboom,DemonAttack,Skiing,Boxing}.cpp`: 코드의 GPL 고지뿐이다.

**릴리스 공지 23개** (https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases). 롬을 말한 곳은 동봉 사실뿐이다.
- v0.9.0 (2024-05-20)
  > An important change in this update is that the Atari ROMs are packaged within the PyPI installation such that users no longer require `pip install "gym[accept-rom-license]"` (AutoROM) or `ale-import-roms` for downloading or loading ROMs. This should significantly simplify installing Atari for users.
- v0.12.1 (2026-08-16)
  > Plus, as v0.12.0 was accidentally released outside ROMs installed within the wheels, v0.12.0 on PyPI will be yanked.

**이슈와 PR.**
- 검색어: license, legal, copyright, permission. 대상 저장소: ALE, AutoROM.
- 본문과 댓글을 전부 읽은 것: ALE #437 #513 #516 #536 #564 #630, AutoROM #1 #7 #49.
- 인정 기준에 맞는 문구는 없다.

**AutoROM** (저장소 커밋 5bb4474db00b5b6ab7510ca272546a14aa07fa6a, PyPI `AutoROM` / `AutoROM.accept-rom-license` 0.6.1).
- `README.md` 5행:
  > **Aug 11, 2025: This project has been deprecated due to a lack of wide spread community use (and largely no longer being needed), and is no longer planned to receive any additional updates or support.**
- `README.md` 9행:
  > AutoROM automatically downloads the needed Atari ROMs from ROM hosting websites into the ALE-Py folder and Multi-Agent-ALE-py folder in a very simple manner:
- `LICENSE.txt`: MIT, "Copyright (c) 2020 Farama Foundation". AutoROM 코드 자체의 라이선스다. `packages/AutoROM/LICENSE.txt`와 `packages/AutoROM.accept-rom-license/LICENSE.txt`도 같은 파일이다.
- PyPI 두 패키지: 라이선스는 MIT이고 설명은 README와 같다.

## 5. 예외 — 저자 공개 근거가 있는 롬 (후보 밖)

ALE 릴리스 v0.7.2 (2021-10-07): https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/tag/v0.7.2

> Package Tetris by Colin Hughes. This ROM is made publicly available by the author. This is useful for other open-source packages to be able to unit test against the ALE. (@tfboyd)

`ale_py/roms/tetris.bin`이 설치본에 있다. 후보 8개 밖이고, 지시서가 후보 밖 게임 추가를 금지하므로 쓰지 않는다.

## 6. 게임별 정리

| 게임 | 설치본에 있음 | md5 (`ale_py/roms/`) | 인정 근거 |
|---|---|---|---|
| Pong | `pong.bin` | 60e0ea3cbe0913d39803477945e9e5ec | 없음 |
| Breakout | `breakout.bin` | f34f08e5eb96e500e851a80be3277a56 | 없음 |
| SpaceInvaders | `space_invaders.bin` | 72ffbef6504b75e69ee1045af9075f66 | 없음 |
| Freeway | `freeway.bin` | 8e0ab801b1705a740b476b7f588c6d16 | 없음 |
| Kaboom | `kaboom.bin` | 5428cdfada281c569c74c7308c7f2c26 | 없음 |
| DemonAttack | `demon_attack.bin` | f0e0addc07971561ab80d9abe1b8d333 | 없음 |
| Skiing | `skiing.bin` | b76fbadc8ffb1f83e2ca08b6fb4d6c9f | 없음 |
| Boxing | `boxing.bin` | c3ef5c4653212088eda54dc91d787870 | 없음 |

md5는 파일 바이트만 읽어 계산했고, 에뮬레이터에 불러오지 않았다.

## 7. 현재 상태

- **의존성.** `gymnasium` 1.4.0, `ale-py` 0.12.1을 설치했다. 함께 `cloudpickle` 3.1.2와 `farama-notifications` 0.0.6이 들어왔다.
  - 방법: `uv add --no-sync` 뒤 `uv sync --inexact`. 기존 잠금 버전은 하나도 바뀌지 않았다.
  - 의존성 자체는 승인됐다. `pyproject.toml`과 `uv.lock`은 미커밋으로 둔다. 근거가 끝내 없으면 되돌릴지를 그때 정한다.
- **롬과 환경.** 롬을 불러온 적이 없고 `gym.make`도 호출하지 않았다. 게임 화면이나 점수는 하나도 없다.
- **코드.** `atari.py`와 `tests/test_atari.py`는 총괄 지시로 작성 전에 멈췄다. 사용자가 롬 없이 가는 쪽을 고르면 버려질 작업이기 때문이다.
- **원문 사본.** 이슈와 릴리스 JSON, 사이트 HTML, 저장소 압축본은 조사 세션의 임시 폴더에만 두었고 저장소에는 넣지 않았다. 다시 확인하려면 위 URL과 커밋 해시를 쓴다.
- **정정.** 첫 보고에서 롬 개수를 "111개"라고 했다. 이는 롬이 아닌 항목 3개(`__init__.py`, `md5.json`, `__pycache__`)를 포함한 수이고, 롬은 108개다.
