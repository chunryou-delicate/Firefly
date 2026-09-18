# M6 3D 데이터 계약 — 세 창의 공통 원본

총괄 세션 작성 (2026-09-18). **이 문서가 계약이다.** 세 창이 각자 만들므로 여기 없는 파일·필드·메시지는 쓰지 않는다.
바꿔야 하면 바꾸지 말고 총괄 세션(이 지시를 전달한 메시지의 `from-name`)에 보고한다.

## 좌표계 (총괄이 데이터로 확인함, 추측 아님)

- 모든 3D 좌표는 **MaleCNS 원본 복셀**이다. 1 복셀 = **8 nm**. 파일에서 단위 변환을 하지 않는다. 뷰어가 표시할 때만 ×0.008 해서 μm로 읽는다.
- 확인된 사실: SWC 골격의 좌표와 어노테이션 `somaLocation`이 같은 공간이다. bodyId 10001·10002·98113에서 soma → 가장 가까운 골격 노드 거리가 113~154 복셀(≈1 μm)이었다.
- 유지 뉴런 전체의 시냅스 중심 좌표 범위: x 4,673~92,111 · y 6,069~68,657 · z 10,778~133,549 복셀. z가 긴 것은 복부신경삭이 포함되기 때문이다.
- SWC 컬럼은 `id type x y z radius parent`이고 `parent = -1`이 뿌리다. 주석줄은 `#`로 시작한다.

## 파일 1 — 뉴런 점구름 (창 B가 생산)

`data/cache/neurons-3d.bin` + `data/cache/neurons-3d.json`

`.bin`은 헤더 없는 연속 바이트이며 `.json`의 `arrays`가 오프셋과 길이를 준다. 순서는 **유지 그래프의 뉴런 인덱스 순**(M1의 `Graph` 인덱스)이고 이 순서가 run.json의 뉴런 인덱스와 같다.

| 배열 | dtype | 길이 | 내용 |
|---|---|---|---|
| `pos` | float32 | 3n | x, y, z 복셀. 뉴런당 3개 |
| `set` | uint8 | n | `sets` 배열의 인덱스. 프로브 집합 소속 |
| `src` | uint8 | n | 0 = somaLocation, 1 = 시냅스 중심, 2 = 좌표 없음 |

```json
{
  "n": 165122, "voxel_nm": 8, "order": "graph neuron index",
  "bbox": {"min": [x,y,z], "max": [x,y,z]},
  "sets": ["JO_AB_L", "...", "rest"],
  "src_counts": {"soma": 0, "centroid": 0, "none": 0},
  "arrays": {"pos": {"offset": 0, "length": 495366, "dtype": "float32"},
             "set": {"offset": 1981464, "length": 165122, "dtype": "uint8"},
             "src": {"offset": 2146586, "length": 165122, "dtype": "uint8"}},
  "generated": "ISO8601", "graph_cache_md5": "..."
}
```

`src = 2`인 뉴런은 `pos`에 **NaN**을 넣는다. 0으로 채워서 원점에 뭉치게 하지 않는다. 뷰어가 걸러낸다.

## 파일 2 — 골격 묶음 (창 A가 생산)

집합 이름마다 `data/cache/skel-<name>.bin` + `data/cache/skel-<name>.json`.

`.bin`은 **선분 배열**이다. float32로 `x1 y1 z1 x2 y2 z2`가 반복된다. 뷰어는 `gl.LINES`로 그대로 그린다.
뉴런별 구간은 `.json`이 준다.

```json
{
  "set": "pC1", "voxel_nm": 8, "n_neurons": 156, "n_segments": 28114,
  "bbox": {"min": [x,y,z], "max": [x,y,z]},
  "decimation": {"method": "rdp-per-branch", "tolerance_voxels": 40, "max_segments_per_neuron": 200},
  "neurons": [{"body_id": 98113, "idx": 12345, "seg_offset": 0, "seg_count": 187,
               "soma": [x,y,z] | null, "n_nodes_raw": 1653}],
  "source": {"url_prefix": "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc",
             "md5_manifest": "data-provenance/skeleton-hashes.json"},
  "generated": "ISO8601"
}
```

- `idx`는 유지 그래프의 뉴런 인덱스다. 그래프에 없는 bodyId는 묶음에 넣지 않는다.
- 골격 파일이 없는(404) 뉴런은 조용히 건너뛰지 말고 `.json`의 `missing: [body_id…]`에 적는다.
- 파일 크기 상한: 묶음 하나가 **20 MB를 넘으면 예외**를 던지고 상한을 낮추라고 안내한다.

## 파일 3 — 활동

기존 `run.json` 그대로 쓴다. 뷰어는 `frames.rates`(영역×시간)와 `spikes`(샘플링된 뉴런 인덱스)를 읽고,
뉴런 인덱스로 파일 1의 `pos`를 찾는다. **run.json 계약은 수정하지 않는다.**

## HTTP 엔드포인트 (창 B가 flysim/live에 추가)

기존 정적 서버에 허용 목록으로 추가한다. 경로 탈출은 지금처럼 404.

| 경로 | 내용 |
|---|---|
| `/neurons-3d.json`, `/neurons-3d.bin` | 파일 1 |
| `/skel/<name>.json`, `/skel/<name>.bin` | 파일 2. 없으면 404 |
| `/skel/index.json` | 서버가 가진 묶음 목록 `[{name, n_neurons, n_segments, bytes}]` |

## 프로토콜 (v1.2, 창 B가 추가)

`docs/m5-protocol.md`에 **추가만** 한다. 기존 메시지는 건드리지 않는다.

| 방향 | type | 필드 |
|---|---|---|
| 서버→클라 | `hello` | 기존에 `assets:{neurons_3d:bool, skeleton_sets:[name…]}` 추가 |

골격을 요청하는 메시지는 이번에 만들지 않는다. 미리 만들어 둔 묶음만 HTTP로 가져간다.

## 금지

- 그래프 축소, run.json 계약 수정, 뷰어(`flysim-viewer.html`)·조종석(`flysim-live.html`) 수정
- 새 파이썬 의존성. 3D 뷰어는 **외부 스크립트 없이 WebGL2 직접** 사용(three.js 등 금지)
- 좌표 재중심화·단위 변환을 파일에 넣는 것
