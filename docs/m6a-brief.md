# M6a 구현 지시서 — 골격 다운로드·간략화 파이프라인

총괄 세션 작성 (2026-09-18).

> **보고 대상:** 총괄 세션. 세션 이름은 재시작마다 바뀌므로 이 지시를 전달한 메시지의 `from-name`을 쓴다.

먼저 읽을 것: `CLAUDE.md`, **`docs/m6-3d-contract.md`(계약, 수정 금지)**, `flysim/data/download.py`(해시 검증 방식), `flysim/graph/graph.py`, `flysim/probe/sets.py`.

의존성: 기존 그대로. 새 패키지 금지.

## 할 일
`flysim/data/skeletons.py` 하나를 만든다.

1. **다운로드**: `https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/<bodyId>.swc`.
   총괄이 확인한 사실 — 인증 없이 받아지고, 서버가 `x-goog-hash`로 MD5를 준다. `download.py`와 같은 방식으로 검증하고 `data-provenance/skeleton-hashes.json`에 기록한다. `data/raw/skeletons/`에 캐시하고 재실행 시 해시가 맞으면 건너뛴다.
   404는 예외가 아니라 `missing` 목록에 적는다(계약 참조). 동시 다운로드는 스레드 4~8개 정도, 실패는 3회 재시도 후 보고.
2. **파싱**: SWC 컬럼 `id type x y z radius parent`, `parent=-1`이 뿌리. 여러 뿌리가 있을 수 있으니 숲으로 다룬다.
3. **간략화**: 분기점과 끝점은 반드시 보존하고, 그 사이의 곧은 구간만 Ramer–Douglas–Peucker로 줄인다. 기본 허용오차 40복셀(≈0.32 μm), 뉴런당 선분 상한 200. 상한을 넘으면 허용오차를 올려 재시도하고 최종 허용오차를 `.json`에 기록한다.
4. **묶음 출력**: 계약의 파일 2 형식. `build_bundle(body_ids, name, tolerance_voxels=40, max_segments_per_neuron=200) -> (bin_path, json_path)`.
5. **CLI**: `python -m flysim.data.skeletons --set JO_AB --set pC1 --set WED --set AMMCtype` 처럼 프로브 집합 이름으로 받는다. 집합 정의는 `flysim/probe/sets.py`를 그대로 쓴다. 좌우는 합쳐서 한 묶음으로 낸다(`JO_AB_L`+`JO_AB_R` → `JO_AB`).

## 예상 용량 (총괄이 표본으로 측정)
JO 138개 평균 22 KB · pC1 156개 평균 124 KB · AMMC 208개 평균 80 KB · WED 900개 평균 141 KB. 원본 합계 약 160 MB.
간략화 후 묶음은 집합당 수 MB여야 한다. 20 MB를 넘으면 계약대로 예외.

## 완료 조건
- [ ] 네 집합(JO_AB, pC1, WED, AMMCtype) 묶음 생성, 각 20 MB 미만
- [ ] 재실행 시 해시 일치로 다운로드를 건너뜀
- [ ] 간략화 전후 노드 수와 최대 편차를 `docs/m6a-report.md`에 표로
- [ ] `tests/test_skeletons.py`: SWC 파싱(합성 데이터), 분기점 보존, 허용오차 준수, 묶음 json이 계약 키를 전부 가짐, 그래프에 없는 bodyId 제외, 20 MB 상한 예외
- [ ] `pytest tests/` 전부 통과 (현재 198개)
- [ ] "M6a: ..." 커밋 하나. **push 금지** — 원격 푸시는 총괄이 한다.

## 파일 경계
너는 `flysim/data/skeletons.py`, `tests/test_skeletons.py`, `docs/m6a-report.md`, `data-provenance/skeleton-hashes.json`만 만진다.
`flysim/probe/`, `flysim/live/`, `flysim-*.html`, 계약 문서는 다른 창 소관이다.
