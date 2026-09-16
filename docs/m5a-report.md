# M5a report — flysim-live server

구현 세션(M1 서버 창), 2026-09-16. 사양: `docs/m5a-brief.md`. 계약: `docs/m5-protocol.md` **v1.1**(커밋 0f11e2e)
— 계약 문서는 수정하지 않았다. 측정 하드웨어: RTX 4090 Laptop(16 GB), torch 2.6.0+cu124, triton 3.2.0,
websockets 16.1.1. 새 의존성 없음.

## 1. 무엇이 들어갔나

```
flysim/live/
  protocol.py        계약을 코드로 (메시지 검증 + 서버 메시지 빌더). 거부 경로 2종
  stream.py          JO 스트리밍 입력 경로 (상태 유지 biquad·포락선, 자극 생성기, 오디오 버퍼)
  session.py         LiveSession: 엔진 소유, 실시간 루프(별도 스레드), 링버퍼, 안전 신호, 스냅샷, 제어 로그
                     + compute_hops(), replay_control_log()
  server.py          엔트리. 웹소켓 + 정적 서버 기동, 클라이언트별 송신 큐
  static.py          허용 목록 기반 정적 서빙 (flysim-live.html / flysim-viewer.html / neurons-v1.json)
  export_neurons.py  LiveContext 구성 + runs/live/neurons-v1.json
  bench.py           페이싱 측정 (§4의 표를 그대로 출력)
tests/test_live.py   54개
```

기동:

```
.venv/bin/python -m flysim.live.server            # ws://127.0.0.1:8765/ws + http://127.0.0.1:8080/
  [--host 0.0.0.0] [--ws-port 8765] [--http-port 8080] [--synapse conductance] [--g ...]
  [--dt 1.0|0.1] [--mode rate|phase-lock] [--a-in 640] [--speed 1.0] [--bin-ms 1.0]
  [--noise-sigma 0] [--frame-every 1] [--seed 0] [--device cuda] [--no-cuda-graph]
```

기본 동작점은 M4b와 같다: `synapse=conductance`, `g=DEFAULT_G_CONDUCTANCE=3.162e-4`, 노이즈 0,
`a_in=640`(M3 rate 선택값), `dt=1 ms`, `bin=1 ms`. 컨텍스트(그래프·프로브 집합·좌표·JO 대상) 적재는 1.2 s.

**엔진·감각·프로브·뷰어 코드는 한 줄도 고치지 않았다.** `stream.py`는 `flysim.sensory.jo`의
`biquad_coeffs`·`BAND_HZ`·`TAU_ENV_MS`·`MODES`를 import해서 상태 유지 버전을 새로 쓴 것이라, 총괄에게 보고했던
"공유 함수 추출"은 필요하지 않았다. 배치 경로와 같은 결과를 내는지는 테스트로 강제한다(§5).

## 2. 구현한 메시지 (계약 v1.1 전부)

**클라이언트 → 서버 12종** — `set_params`, `stimulus`, `audio_start`, `audio_stop`, `inject`,
`pause`, `resume`, `step`, `reset`, `get_hops`, `snapshot`, `set_frame_every`.

**서버 → 클라이언트 8종** — `hello`, `frame`, `ack`, `params`, `hops`, `snapshot_done`, `warning`, `error`.

거부 경로를 둘로 나눴다. 형식 자체가 틀린 메시지(객체 아님, 모르는 `type`, `req_id` 없음/정수 아님, JSON 파싱 실패)는
`error`, 형식은 맞지만 값을 거절하는 경우(범위 초과, 없는 집합 이름, 엔진이 지원하지 않는 파라미터)는 `ack ok:false`다.
계약의 "잘못된 메시지 → error"와 "adapt_*는 ack ok:false로 거절"을 둘 다 만족한다.

세부 사항:

- `set_params`: 다음 스텝 경계에서 적용. `dt_ms`·`synapse` 변경은 엔진 파라미터 교체 + `reset`을 동반하고 ack에
  "engine reset (dt_ms/synapse changed)"로 명시한다. 엔진 객체를 새로 만들지 않고 `EngineParams`만 교체하는데,
  두 값 모두 계수·커널 분기에만 영향을 주고 상태 버퍼 모양은 같기 때문이다(재생성보다 빠르고 결과는 동일).
- `params` 브로드캐스트: `params_version`이 오르는 유일한 지점(`_bump_params_version`)에서 전 클라이언트에 보낸다.
- `frame.n_bins`: 묶인 빈 수. `reset`·`pause` 경계에서는 부분 묶음을 먼저 내보내므로 `frame_every`보다 작은 값이 나온다.
- `inject`: 여러 개가 동시에 살아 있을 수 있고(각자 만료 시각), 자극과 **합산**된다. 집합 이름 또는 인덱스 목록 모두 가능.
- `get_hops`: 전방 CSR BFS, 층당 상한 5,000(초과 시 접촉 수 상위로 자르고 `truncated[i]=true`).
  시뮬 스레드가 아니라 실행기 스레드에서 처리해 페이싱을 방해하지 않는다.
- `snapshot`: 링버퍼(최근 10 s)를 뷰어 계약 `run.json`으로 저장. `run_id`는 `[A-Za-z0-9._-]{1,64}`로 제한한다
  (경로 성분이기 때문).
- 오디오: `audio_start`가 큐에 들어가 스텝 경계에서 적용되기를 기다리는 사이에 도착한 PCM도 버리지 않는다.
  웹소켓 스레드가 즉시 버퍼를 만들고(`prepare_audio`) 신호 경로 연결만 스텝 경계에서 한다.

## 3. 입력 경로가 배치 경로와 같음

라이브 입력은 `PCM → 대역통과(100–400 Hz, jo의 계수) → [rate: 정류 + τ=2 ms 포락선] → dt 격자 평균 →
[phase-lock: 반파 정류]` 순으로, `jo.JOAdapter.drive()`의 합성과 같다. 어느 샘플이 어느 스텝에 속하는지는
`stream.sample_end()`가 `jo.bin_mean`의 `floor(n/fs*1000/dt + 1e-9)`을 정확히 역으로 계산해 정한다. 덕분에
스텝 주기의 정수배가 아닌 샘플레이트(dt=1 ms에 44.1 kHz)도 같은 격자에 떨어진다.

`tests/test_live.py::test_stream_matches_batch`가 같은 파형을 1–256 샘플의 불규칙한 청크로 밀어 넣고 배치 결과와
비교한다(두 모드, 오차 1e-5 이내, 언더런 0). `test_stream_click_train_source_matches_batch_waveform`은 끝없는
클릭 트레인 생성기가 `stimuli.click_train`과 같은 샘플을 내는지 본다(1e-12 이내). 기존 `tests/test_sensory.py`는
그대로 통과한다(배치 경로 무변경).

생성 자극은 항상 M3의 10 kHz로 만든다. 오디오 스트림이 44.1 kHz로 바뀐 뒤에 클릭 트레인을 틀어도 M3와 같은 파형이다.

## 4. 페이싱 측정

`python -m flysim.live.bench --seconds 3` (전체 그래프 165,122 뉴런 / 25.56 M 엣지, conductance, g 3.162e-4,
a_in 320, 워밍업 50빈 제외, 타이트 루프 = 페이싱 sleep 없음):

| dt (ms) | 모드 | 자극 | 스파이크/빈 | 빈당 μs | +JSON μs | 합계 μs | 실시간 배수 |
|---|---|---|---|---|---|---|---|
| 1.0 | rate | 무음 | 0 | 208 | 10 | 218 | 3.95× |
| 1.0 | rate | 클릭 트레인 | 14 | 263 | 14 | 277 | 3.12× |
| 0.1 | phase-lock | 무음 | 0 | 558 | 10 | 568 | 1.65× |
| 0.1 | phase-lock | 클릭 트레인 | 5 | 634 | 14 | 647 | 1.45× |

- **dt = 1 ms에서 목표(빈당 < 1 ms, speed 1 유지)를 만족한다.** 여유는 3배 이상이다.
- **dt = 0.1 ms에서도 speed 1이 가능하다**(1.45–1.65×). 브리프는 가능 여부만 측정하라고 했는데, 두 가지를 고치니
  가능해졌다: ① 프레임 계산에서 `index_add_`(영역 15칸에 165 k 원자연산이 몰려 250 μs)를 `nonzero` 후 발화 뉴런만
  다루는 희소 경로로 바꿨고(빈당 419 → 150 μs), ② 엔진의 CUDA 그래프 캡처를 켰다(`use_cuda_graph`, CUDA에서 기본값).
  ②만으로 dt 0.1 ms가 0.76× → 1.49×로 올라간다. M2 보고서가 캡처와 eager의 스파이크가 bit-identical임을 이미
  확인했고, 라이브 세션 수준에서도 실그래프 200스텝을 두 방식으로 돌려 같은지 테스트로 강제한다.

실제 서버(페이싱 sleep 포함, speed 1, 40 s):

| 조건 | 빈당 평균 μs | lagging 빈 | 비고 |
|---|---|---|---|
| 클라이언트 없음 | 404 | 498 / 37,538 (1.3 %) | lagging은 전부 기동 직후 |
| 클라이언트 2개, frame_every 7 | 444 | 494 / 26,553 (1.9 %) | 모든 프레임 `running` |
| 클라이언트 1개, frame_every 1, 클릭 트레인 | — | 0 | 5 s에 4,997 프레임(=1,000/s), 전부 `running` |

두 가지를 기록해 둔다.

1. **기동 직후 한 번 400 ms 정도 멈춘다.** Triton 커널 컴파일과 CUDA 그래프 캡처가 첫 빈에 몰린다(`max_bin_us`
   ≈ 4.0e5). 스텝을 건너뛰지 않으므로 그만큼 뒤처졌다가 3–4배속으로 따라잡고, 그동안 `status="lagging"`이 뜬다.
2. **빈마다 `time.sleep`을 부르면 오히려 못 따라간다.** 1 kHz로 sleep을 부르면 syscall 오버헤드와 GPU 클럭 하강이
   겹쳐 빈당 818 μs까지 올라가고 34 %의 빈이 lagging이었다. 2 ms 이상 앞설 때만 자게 바꾸니 404 μs / 1.3 %가 됐다
   (`SLEEP_MIN_MS`). 평균 페이싱은 절대 시각 기준이라 이 변경으로 드리프트가 생기지 않는다.

`snapshot`은 실행기 스레드에서 돌지만 JSON 직렬화가 GIL을 잡아 0.3–0.4 s의 lag 스파이크를 만든다(2 s / 1 ms 빈 기준).
스텝은 건너뛰지 않고 `status="lagging"`으로만 표시된다.

## 5. 테스트

`.venv/bin/python -m pytest tests/ -q` → **190 passed, 1 failed**. 실패한 하나는 `tests/test_live_html.py`
(M5b 조종석 파일의 계약 v1.1 대응이 진행 중)이고 서버 쪽 테스트는 전부 통과한다. `tests/test_live.py`는 54개:

| 묶음 | 내용 |
|---|---|
| 프로토콜 | 12종 전부 파싱·기본값, 형식 오류 7종 → `ProtocolError`, 값 거절 18종 → `Reject`, 적응 파라미터 분기, 서버 빌더 |
| 스트림 | `sample_end`가 `jo.bin_mean`의 역인지(4개 조합), 스트리밍 == 배치(2모드), 클릭 트레인 파형 일치, 오디오 언더런·상한, describe 키 |
| 세션 | hello 계약 키, `set_params` 적용 시점·버전·reset 동반, 자극/주입이 `i_ext`에 도달·만료·ILD, 거절, 프레임·링버퍼, `frame_every` 묶음과 `n_bins`, `params` 브로드캐스트, 안전 신호 발생/해제, pause·step·resume, 제어 로그 기록, 오디오 선버퍼링 |
| 결정론 | 제어 로그 재생 = 같은 스파이크 (§6) |
| 스냅샷 | 뷰어 계약 키 전부 + `live_control_log`·`live_params`, 데이터 없을 때 거절 |
| hops | 층 분리·중복 없음·`min_contacts`·상한 잘림·범위 밖 거절 |
| 루프 | 스레드 기동·프레임 생성·페이싱 메트릭 |
| 실그래프(CUDA) | 500빈 구동 + CUDA 그래프 on/off 스파이크 일치 + 스냅샷 + 빈당 < 1 ms |

## 6. 결정론(제어 로그 재생)

제어 로그 `runs/live/<session_id>/control-log.jsonl`의 첫 줄이 세션 시작 시점의 엔진·라이브 파라미터·시드를 담고,
이후 줄마다 **적용된 스텝 번호**와 메시지 원문·적용 결과가 남는다. `replay_control_log(ctx, log, n_steps)`가 같은
스텝에서 같은 메시지를 다시 적용하며 돌린다. 테스트는 300스텝 동안 클릭 트레인 → 주입 → `a_in` 변경 → 톤으로 바꾸는
런을 돌린 뒤 재생해 `(t, idx)` 배열이 완전히 같은지 본다(두 번 재생해 재생끼리도 같은지까지).

**오디오 입력은 재현되지 않는다.** PCM 페이로드는 로그에 남기지 않으므로(용량), `audio` 자극을 쓴 구간은 재생 시
무음이 된다. 로그에는 `audio_start`/`audio_stop`과 수신 샘플 수만 남는다. 결과로 제출할 런은 계약대로 스크립트로
재현한다.

## 7. 스냅샷

`snapshot`은 링버퍼를 `flysim/probe/export.py::write_run_json`에 그대로 넘긴다. 즉 기존 뷰어가 여는 파일과
같은 계약이다(이 세션에서 브라우저로 열어보지는 못했고, `validate_run_doc` + 계약 키 테스트로 갈음한다).
`meta`에 추가로 들어가는 것: `live_control_log`(로그 경로), `live_session_id`, `live_params`, `live_seconds`,
`live_step_range`, `safety_thresholds`. `spike_sample_ratio`는 실제 값으로 덮어쓴다 — 샘플링이 내보내기 시점이
아니라 라이브 중 빈마다(상한 1,500) 이미 일어났기 때문이다. `active_neurons`는 샘플 기준이라는 주석을 함께 넣는다.

크기 주의: 빈 1 ms에서 2 s 스냅샷은 2,000 프레임이고, 활동이 많으면 파일이 14 MB 정도가 된다(대부분 `spikes`).

## 8. 안전 신호

계약의 세 가지를 빈마다 판정하고, 발생 시 1회 `warning`, 해제 시 `code+"_cleared"`를 보낸다. 임계값은
`session.SAFETY`에 모여 있고 스냅샷 `meta`에도 들어간다.

| code | 조건 |
|---|---|
| `runaway` | 최근 100 ms 발화 뉴런 비율 ≥ 90 % |
| `ignited` | 자극·주입·노이즈가 모두 0인데 최근 200 ms 발화 비율 ≥ 1 % |
| `all_silent` | 최근 300 ms 안에 입력이 있었는데 그 구간 발화가 0 |
| `recorder_overflow` | 라이브 경로는 엔진 recorder를 쓰지 않으므로(스텝마다 직접 읽는다) 발생하지 않는다. 방어용으로만 남겨뒀다 |

스모크 런에서 `all_silent`(a_in=0), `ignited`(a_in=640 클릭 트레인이 전도도 모델을 점화시킨 뒤 자극 종료)가 실제로
떴다가 해제되는 것을 확인했다.

## 9. 원격 접속

인증이 없다. 8080/8765는 공개망에 열지 말 것. 노트북 밖에서 조종석을 쓰려면 터널을 쓴다.

Tailscale이면 노트북과 클라이언트 기기에 각각 설치·로그인한 뒤, 서버를 테일넷 주소에 바인딩하면 된다.
`tailscale ip -4`로 노트북의 100.x 주소를 확인하고 `python -m flysim.live.server --host 100.x.y.z`로 띄운 다음,
다른 기기 브라우저에서 `http://100.x.y.z:8080/`을 연다(웹소켓 주소는 조종석이 같은 호스트명으로 만든다).
`--host 0.0.0.0`은 카페 와이파이 같은 곳에서 그대로 노출되므로 피한다. SSH 포트 포워딩
(`ssh -L 8080:localhost:8080 -L 8765:localhost:8765 노트북`)도 같은 일을 하며, 이 경우 `--host 127.0.0.1`
기본값 그대로 두면 된다.

## 10. Deviations / 미구현

1. **`status`에 `"lagging"`을 포함**시켰다. v1.0 표에는 `running`/`paused`만 있었고 서버 의무 절에만 `lagging`이
   있었는데, v1.1에서 계약에 반영됐다.
2. **부분 묶음 flush**: `reset`·`pause` 경계에서 `frame_every`에 못 미친 묶음을 먼저 내보낸다. 계약 v1.1의
   "reset·pause 경계에서는 실제로 묶인 수"를 그렇게 읽었다.
3. **일시정지 중 하트비트 프레임**을 0.2 s마다 보낸다(`status="paused"`, `rates` 전부 0, `spikes` 없음, `n_bins=0`).
   계약에 정지 중 프레임 규정이 없어서 정한 것이다. 조종석이 이 프레임을 시간축에 쌓지 않도록 `n_bins=0`으로 표시했다.
4. **`ack` 순서**: 값이 거절된 메시지는 웹소켓 스레드가 즉시 답하고, 수용된 메시지는 시뮬 스레드가 스텝 경계에서
   답한다. 따라서 `req_id` 순서가 뒤바뀔 수 있다(계약은 순서를 규정하지 않는다).
5. **느린 클라이언트**는 프레임을 잃는다. 클라이언트마다 256칸 송신 큐를 두고, 큐가 찼을 때 프레임은 버리고
   `ack`·`warning`·`params`는 대신 가장 오래된 항목을 밀어내고 넣는다. 한 브라우저가 느려도 다른 클라이언트와
   시뮬레이션은 영향을 받지 않는다.
6. **`.cpu()`는 빈당 한 번**이지만 동기화는 두 번이다(`nonzero`가 출력 크기를 호스트로 알려야 한다). 희소 경로가
   훨씬 빨라서 이 편을 택했고, 옮기는 값(영역별 카운트·창 합계·샘플 인덱스)은 float32 한 덩어리에 담아 한 번에 옮긴다
   (뉴런 인덱스 < 2²⁴이라 float32로 정확히 표현된다).
7. **영역별 카운트는 빈당 뉴런 하나가 최대 한 번 발화한다고 가정**한다(`last_spike_step` 기반). `t_ref`(2 ms)가
   `bin_ms`(1 ms)보다 크므로 성립하고, `t_ref`는 라이브 파라미터가 아니다.
8. **`input_level`의 기준**은 200 Hz 전대역 톤이 만드는 drive다(모드·dt별로 기동 시 1회 계산해 `hello.engine.
   drive_full_scale`에 넣는다). 계약이 "0~1"만 요구해서 정한 것이며, 세션 중에 기준이 바뀌지 않는다.
9. **CUDA 그래프 기본 켜짐**(CUDA일 때). 문제가 있으면 `--no-cuda-graph`로 끈다.
10. **`bench.py`는 브리프의 파일 목록에 없다.** §4의 수치를 누구나 다시 측정할 수 있게 넣었다.
11. `adapt_b`/`adapt_tau_w`는 엔진에 해당 필드가 있으면 통과시키고, 없으면 0 이외의 값을 `ack ok:false`로 거절한다
    (`P.ENGINE_HAS_ADAPT`). 이 리포트를 쓰는 시점에 M2c가 엔진에 들어와 있어 실제로 수용된다.
12. 미구현 없음 — 계약 v1.1의 모든 메시지와 서버 의무를 구현했다.
