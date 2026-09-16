# M5a 구현 지시서 — flysim-live 서버

총괄 세션 작성 (2026-09-16). 대상: 새 세션. **`docs/m5-protocol.md`가 계약이다.** 먼저 읽을 것: 계약서, `CLAUDE.md`, `flysim/engine/lif.py`(run/step, i_ext 규약, recorder),
`flysim/sensory/{base,jo,stimuli}.py`, `flysim/probe/{sets,probe,export}.py`, `docs/m3-report.md`.

의존성: `websockets`(승인됨, 설치돼 있음) + 표준 라이브러리(`http.server`, `asyncio`, `threading`, `json`). 그 외 금지.

## 산출물
```
flysim/live/
  server.py     # 엔트리: python -m flysim.live.server [--host 0.0.0.0] [--ws-port 8765] [--http-port 8080] [--synapse conductance] [--g ...]
  session.py    # LiveSession: 엔진 소유, 실시간 루프(별도 스레드), 파라미터 큐, 자극/주입 합성, 링버퍼, 안전 신호, 스냅샷
  protocol.py   # 메시지 dataclass·검증(계약서 표를 코드로), 프레임 직렬화
  stream.py     # JO 스트리밍 경로: 상태 유지 biquad + 포락선, 임의 길이 PCM 청크 → dt 격자 전류. flysim/sensory/jo.py의 계수·모드와 동일해야 함(공유 함수로 뽑아 재사용, 기존 배치 경로의 출력이 바뀌면 안 됨 — test_sensory 회귀)
  static.py     # http 정적 서빙 (flysim-live.html은 M5b가 만든다; 없으면 안내 페이지)
  export_neurons.py  # runs/live/neurons-v1.json 생성 (probe/export의 좌표·영역 로직 재사용)
tests/test_live.py
docs/m5a-report.md
```

## 요구사항
- 엔진은 `EngineParams` 기본(synapse는 옵션, 기본 `conductance`, g는 `DEFAULT_G_CONDUCTANCE`)으로 시작. 파라미터 변경은 큐로 받아 스텝 경계에서 적용하고 `params_version`을 올린다. `dt_ms`/`synapse` 변경은 엔진 재생성+reset.
- 실시간 루프: `speed` 페이싱, 밀리면 `lagging`. 프레임 계산(영역별 Hz, 샘플 스파이크, 활성 비율)은 GPU에서 하고 한 번만 `.cpu()`.
  집합은 `probe/sets.py`의 M3 집합을 그대로 쓴다(JO_AB_L … rest). `hello.sets`에는 `rest`를 제외하고 크기만 보낸다(인덱스는 크기 ≤ 2,000인 집합만).
- 자극: `stimuli.py` 생성기를 스트리밍으로 감싼다(무한 클릭 트레인은 주기 생성). `audio`는 `stream.py`. `inject`는 별도 버퍼로 합산. 모든 입력은 엔진 `i_ext` 버퍼에 **디바이스에서** 써 넣는다(호스트 왕복 최소화).
- 링버퍼: 최근 10 s의 (t, sampled idx)와 영역별 rates. `snapshot`은 `probe/export.py`로 뷰어 계약 run.json을 쓴다(`meta.sensory_mode`, `rate_norm_hz`, `spike_sample_ratio` 포함, `meta.live_control_log` 경로 추가).
- 제어 로그 jsonl. 안전 신호 3종 + recorder overflow.
- `get_hops`: 전방 CSR BFS, 층별 인덱스, 각 층 상한 5,000(초과 시 접촉 수 상위로 자름, ack에 표시).
- 클라이언트 없이 무음으로 돈다. 다중 클라이언트 브로드캐스트.
- 성능 목표: dt 1 ms에서 speed 1 유지, 프레임 직렬화 포함 빈당 < 1 ms(측정해 리포트). dt 0.1 ms(phase-lock)는 speed ≤ 1 보장 여부만 측정·보고.

## 테스트 (GPU 없이도 돌아야 하는 것은 `CSRGraph` 합성 그래프로)
- protocol: 계약서의 모든 메시지 타입 파싱·검증, 잘못된 메시지 → `error`.
- session(합성 그래프, torch 백엔드 CPU): set_params 적용 시점·버전, pause/step 결정론(같은 제어 로그 재생 → 같은 스파이크), inject·click_train이 i_ext에 들어가는지, 스냅샷 run.json이 계약 키를 갖는지, 제어 로그 기록.
- stream: 스트리밍 경로 출력 == 배치 경로 출력(같은 파형을 청크로 나눠 넣었을 때, 부동소수 오차 1e-5 이내). 배치 경로 회귀(test_sensory) 통과.
- 실그래프 GPU 통합 테스트 1개(있으면 실행, 없으면 skip): 서버 기동 → hello → click_train 500 ms → frame 수신 → snapshot.

## 완료 조건
- [ ] `python -m flysim.live.server`로 기동, 브라우저(또는 `websockets` 클라이언트 스크립트)로 hello·frame 수신
- [ ] 계약서의 모든 메시지 구현, 테스트 통과(기존 78 + 신규)
- [ ] 제어 로그 재생으로 동일 스파이크(결정론) 확인
- [ ] 스냅샷이 기존 `flysim-viewer.html`에서 열림(HTML 파싱·계약 키 테스트로 대체 가능, 그렇게 적는다)
- [ ] 벽시계 페이싱 측정치 리포트
- "M5a: ..." 커밋 하나. `flysim/engine`, 뷰어, `docs/m5-protocol.md` 수정 금지. 계약이 부족하면 fruit-fly-4c에 보고.

## 원격 접속 (코드 아님, 리포트에 한 절)
Tailscale 등 터널로 노트북의 8080/8765에 붙는 방법을 한 단락으로 적는다. 인증은 없다(취미용, 공개망에 노출 금지 문구).
