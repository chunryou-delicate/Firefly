# flysim-live 프로토콜 계약 (M5) — 서버(M5a)와 조종석 클라이언트(M5b)의 공통 원본

총괄 세션 작성 (2026-09-16). **이 문서가 계약이다.** 서버와 클라이언트를 다른 세션이 만들므로 여기 없는 메시지는 쓰지 않는다.

> **보고 대상:** 총괄 세션. 세션 이름은 재시작 때마다 바뀌므로 **이 지시를 전달한 메시지의 `from-name`**(또는 사용자가 알려준 이름)을 그대로 쓴다. 문서에 적힌 옛 이름은 무효다. 이름을 모르면 `ListAgents`로 `fruit-fly-*` 세션을 확인하고, 애매하면 사용자에게 묻는다.

바꿔야 하면 바꾸지 말고 총괄 세션에 보고한다.

## 전송
- 제어·상태: 웹소켓 `ws://<host>:8765/ws`, JSON 텍스트 프레임. 모든 메시지는 `{"type": "...", ...}`.
- 오디오 업로드: 같은 웹소켓의 **바이너리 프레임** = float32 little-endian 모노 PCM 청크(길이 자유, 권장 20 ms). 샘플레이트는 앞서 보낸 `audio_start`가 정한다.
- 정적 파일: `http://<host>:8080/` 에서 `flysim-live.html`, `neurons-v1.json`(뉴런 좌표·영역, 뷰어 계약의 `neurons`+`regions`와 같은 형식) 제공. 서버가 함께 띄운다.
- 시간 단위 ms, 인덱스는 유지 그래프의 뉴런 인덱스(0..165121).

## 서버 → 클라이언트
| type | 필드 | 언제 |
|---|---|---|
| `hello` | `n_neurons`, `regions:[str]`, `sets:{name:[idx…]}`(프로브 집합, 크기 표시용은 `set_sizes`), `params`(아래 params 객체), `dt_ms`, `bin_ms`, `engine:{synapse, git_commit}`, `neurons_url` | 접속 직후 |
| `frame` | `t_ms`, `step`, `rates:[float…]`(영역별 Hz/뉴런, **정규화 안 함**), `spikes:{idx:[int…], n_total:int, sample_ratio:float}`(빈당 상한 1,500 균일 샘플), `active_frac_100ms`(최근 100 ms 발화 뉴런 비율), `input_level`(0~1, 현재 자극 포락선), `status`(`"running"`/`"paused"`), `speed`, `params_version` | 빈마다(bin_ms=1이면 1 ms마다; 클라이언트 부하를 위해 서버는 `frame_every`(기본 1) 빈마다 묶어 보낼 수 있고 그때 `rates`/`spikes`는 묶음 평균/합집합) |
| `ack` | `req_id`, `ok:bool`, `msg`, `params_version` | 모든 제어 메시지에 대해 |
| `hops` | `req_id`, `src`, `k`, `layers:[[idx…],…]` | `get_hops` 응답 |
| `snapshot_done` | `req_id`, `run_id`, `path`, `n_frames` | `snapshot` 완료 |
| `warning` | `code`(`"ignited"`, `"all_silent"`, `"runaway"`, `"recorder_overflow"`), `msg` | 감지 시 1회, 해제 시 `code+"_cleared"` |
| `error` | `req_id?`, `msg` | 잘못된 메시지 |

`params` 객체: `{synapse:"current"|"conductance", g:float, a_in:float, noise_sigma:float, adapt_b:float, adapt_tau_w:float, dt_ms:1.0|0.1, speed:float, mode:"rate"|"phase-lock"}`.
`adapt_*`는 M2c 전에는 0 고정이며 서버가 `ack ok:false`로 거절한다.

## 클라이언트 → 서버 (모두 `req_id:int` 포함)
| type | 필드 | 의미 |
|---|---|---|
| `set_params` | `params`의 부분집합 | 다음 스텝부터 적용. `dt_ms`·`synapse` 변경은 엔진 재생성이라 `reset`을 동반(서버가 수행하고 ack에 명시) |
| `stimulus` | `kind:"silence"|"click_train"|"tone"|"audio"`, `ipi_ms`, `carrier_hz`, `pulse_ms`, `duration_ms`(0=계속), `ild_db` | 자극 생성기 교체. `audio`는 이후 바이너리 PCM을 JO 어댑터 스트리밍 경로로 투입 |
| `audio_start` | `sample_rate:int`, `channels:1` | 오디오 스트림 시작 |
| `audio_stop` | | 오디오 종료 → 무음 |
| `inject` | `targets:{set:name} \| {idx:[…]}`, `amp:float`, `duration_ms:float` | 지정 뉴런에 전류 주입(자극과 별개, 합산) |
| `pause` / `resume` / `step` | `step.n:int` | 정지·재개·n스텝 진행 |
| `reset` | `seed:int` | 상태 초기화(파라미터 유지) |
| `get_hops` | `src:int`, `k:int(≤4)`, `min_contacts:int` | 하류 k홉 뉴런 집합(전방 CSR, 가중치 절대값 ≥ min_contacts) |
| `snapshot` | `run_id:str`, `seconds:float(≤10)` | 최근 `seconds`의 링버퍼를 뷰어 계약 `run.json`으로 저장(`runs/live/<run_id>/run.json`) + `meta.live_control_log` |
| `set_frame_every` | `n:int` | 프레임 묶음 크기 |

## 서버 의무
- **모든 제어 메시지를 로그**: `runs/live/<session_id>/control-log.jsonl` (수신 시각, step, 메시지 원문, 적용 결과). 스냅샷 `meta`에 이 로그 경로와 현재 `params`를 넣는다. 조종석은 탐색용이고 재현은 로그로 한다.
- 실시간 페이싱: `speed=1`이면 시뮬 1 ms당 벽시계 1 ms. 못 따라가면 `frame.status="lagging"`으로 표시하고 건너뛰지 않는다.
- 안전 신호: 최근 100 ms 발화 비율 ≥ 90 % → `runaway`; 자극·주입·노이즈가 모두 0인데 최근 200 ms 발화 비율 ≥ 1 % → `ignited`; 자극이 있는데 300 ms 동안 발화 0 → `all_silent`.
- 클라이언트 없이도 돈다(무음). 다중 클라이언트는 모두 같은 프레임을 받고, 제어는 누구나 가능(잠금 없음, 취미용).

## 클라이언트 의무
- `hello`의 `regions`/`sets` 순서를 그대로 쓴다. 정규화·색은 전부 클라이언트가 한다(서버는 원시 Hz).
- 연결이 끊기면 마지막 프레임 위에 "disconnected" 표시. 자동 재접속 2 s 간격.
- 모든 슬라이더 변경은 `set_params` 하나로 보내고 `ack`의 `params_version`으로 화면을 동기화한다(낙관적 갱신 금지).
