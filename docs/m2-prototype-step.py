"""M2 prototype (planning session, 2026-09-13): fused Triton step on a SYNTHETIC graph.
Reference only — see docs/m2-brief.md "성능 설계". Not part of the flysim package.
Run: uv run --no-sync --with setuptools python docs/m2-prototype-step.py
"""
import torch, time, triton, triton.language as tl
torch.manual_seed(0); dev='cuda'
N, NNZ = 165_122, 25_563_197
pre = torch.randint(0, N, (NNZ,), device=dev).sort().values
post = torch.randint(0, N, (NNZ,), device=dev, dtype=torch.int32)
w_i = (torch.randint(1, 20, (NNZ,), device=dev) * torch.where(torch.rand(NNZ, device=dev) < 0.3, -1, 1)).to(torch.int32)
indptr = torch.zeros(N + 1, device=dev, dtype=torch.int64); indptr[1:] = torch.bincount(pre, minlength=N).cumsum(0)
indptr32 = indptr.to(torch.int32)

@triton.jit
def propagate1(spk_ptr, indptr_ptr, idx_ptr, w_ptr, acc_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.load(spk_ptr + pid)
    if s != 0:
        start = tl.load(indptr_ptr + pid); end = tl.load(indptr_ptr + pid + 1)
        for off in range(start, end, BLOCK):
            e = off + tl.arange(0, BLOCK); m = e < end
            col = tl.load(idx_ptr + e, mask=m, other=0); w = tl.load(w_ptr + e, mask=m, other=0)
            tl.atomic_add(acc_ptr + col, w, mask=m)

@triton.jit
def propagateB(spk_ptr, indptr_ptr, idx_ptr, w_ptr, acc_ptr, N, NB: tl.constexpr, BLOCK: tl.constexpr):
    base = tl.program_id(0) * NB
    for k in range(NB):
        n = base + k
        if n < N:
            s = tl.load(spk_ptr + n)
            if s != 0:
                start = tl.load(indptr_ptr + n); end = tl.load(indptr_ptr + n + 1)
                for off in range(start, end, BLOCK):
                    e = off + tl.arange(0, BLOCK); m = e < end
                    col = tl.load(idx_ptr + e, mask=m, other=0); w = tl.load(w_ptr + e, mask=m, other=0)
                    tl.atomic_add(acc_ptr + col, w, mask=m)

@triton.jit
def lif(v_ptr, isyn_ptr, acc_ptr, iext_ptr, ref_ptr, spk_ptr, cnt_ptr, out_t_ptr, out_i_ptr, t_ptr, N, cap,
        decay_m, decay_s, g, v_rest, v_reset, v_th, t_ref: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < N
    t = tl.load(t_ptr)
    v = tl.load(v_ptr + i, mask=m, other=0.0); isyn = tl.load(isyn_ptr + i, mask=m, other=0.0)
    acc = tl.load(acc_ptr + i, mask=m, other=0).to(tl.float32); iext = tl.load(iext_ptr + i, mask=m, other=0.0)
    ref = tl.load(ref_ptr + i, mask=m, other=0)
    isyn = isyn * decay_s + g * acc
    vn = v_rest + (v - v_rest) * decay_m + (1.0 - decay_m) * (isyn + iext)
    vn = tl.where(ref > 0, v_reset, vn)
    s = vn >= v_th
    vn = tl.where(s, v_reset, vn)
    ref = tl.where(s, t_ref, tl.maximum(ref - 1, 0))
    tl.store(v_ptr + i, vn, mask=m); tl.store(isyn_ptr + i, isyn, mask=m); tl.store(ref_ptr + i, ref, mask=m)
    tl.store(spk_ptr + i, s.to(tl.int8), mask=m); tl.store(acc_ptr + i, tl.zeros_like(acc).to(tl.int32), mask=m)
    sm = s & m
    slot = tl.atomic_add(cnt_ptr + tl.zeros_like(i), 1, mask=sm)
    ok = sm & (slot < cap)
    tl.store(out_t_ptr + slot, tl.zeros_like(i) + t, mask=ok); tl.store(out_i_ptr + slot, i, mask=ok)

v = torch.full((N,), -65.0, device=dev); isyn = torch.zeros(N, device=dev); acc = torch.zeros(N, device=dev, dtype=torch.int32)
ref = torch.zeros(N, device=dev, dtype=torch.int32); spk = torch.zeros(N, device=dev, dtype=torch.int8); spk_force = torch.zeros(N, device=dev, dtype=torch.int8)
cap = 1 << 24; cnt = torch.zeros(1, device=dev, dtype=torch.int32); out_t = torch.zeros(cap, device=dev, dtype=torch.int32); out_i = torch.zeros(cap, device=dev, dtype=torch.int32)
iext = torch.zeros(N, device=dev); t_dev = torch.zeros(1, device=dev, dtype=torch.int32)
BLOCK = 1024
def step(variant):
    if variant == 1: propagate1[(N,)](spk_force, indptr32, post, w_i, acc, N, BLOCK=128, num_warps=4)
    else: propagateB[(triton.cdiv(N, 32),)](spk_force, indptr32, post, w_i, acc, N, NB=32, BLOCK=128, num_warps=4)
    lif[(triton.cdiv(N, BLOCK),)](v, isyn, acc, iext, ref, spk, cnt, out_t, out_i, t_dev, N, cap,
                                  0.995, 0.98, 0.0, -65.0, -65.0, -50.0, t_ref=20, BLOCK=BLOCK)
    t_dev.add_(1)
def timeit(fn, iters=500):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6
for variant in (1, 2):
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for _ in range(3): step(variant)
    torch.cuda.current_stream().wait_stream(s_)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): step(variant)
    for p in (0.0, 0.001, 0.01, 0.05, 0.2):
        spk_force.copy_((torch.rand(N, device=dev) < p).to(torch.int8)); torch.cuda.synchronize()
        print(f"variant {variant} graph replay p_active={p}: {timeit(gr.replay):.0f} us   (eager: {timeit(lambda: step(variant), 100):.0f} us)")
# sanity: t counter advanced and spike log empty (g=0, no input)
print("t_dev", int(t_dev), "spikes logged", int(cnt), "mem MiB", torch.cuda.max_memory_allocated() // 2**20)
