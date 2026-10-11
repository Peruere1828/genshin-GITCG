"""Native-Windows GPU stress via ctypes + nvcuda.dll (no CUDA toolkit needed).

Complements ``scripts/gpu_stress.cu`` for the Windows side: runs a pure-FP32 FMA
burn kernel (sm_75 PTX embedded) from a *native* Windows process, so clocks/power
can be watched while Windows (not WSL) drives the GPU. Verified to match the WSL
behaviour (see reports/LOCAL_BASELINE.md).

Run from Windows (the script may live on the WSL filesystem):
    python gpu_stress_native.py 60        # 60 s (default 60); Ctrl-C to stop early

Watch in parallel (PowerShell / cmd):
    nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,power.draw,enforced.power.limit --format=csv -l 1
"""

import ctypes
import sys
import time

PTX = r"""
.version 8.3
.target sm_75
.address_size 64
.visible .entry burn(
    .param .u64 burn_param_0,
    .param .u64 burn_param_1
)
{
    .reg .pred 	%p<7>;
    .reg .f32 	%f<56>;
    .reg .b32 	%r<3>;
    .reg .b64 	%rd<14>;
    ld.param.u64 	%rd8, [burn_param_0];
    ld.param.u64 	%rd9, [burn_param_1];
    mov.u32 	%r1, %tid.x;
    cvt.rn.f32.u32 	%f25, %r1;
    fma.rn.f32 	%f54, %f25, 0f3A83126F, 0f3F800000;
    mov.u32 	%r2, %ctaid.x;
    cvt.rn.f32.u32 	%f26, %r2;
    fma.rn.f32 	%f53, %f26, 0f3A83126F, 0f40000000;
    setp.eq.s64 	%p1, %rd9, 0;
    mov.f32 	%f55, 0f40490FD0;
    @%p1 bra 	$L__BB0_7;
    add.s64 	%rd10, %rd9, -1;
    and.b64  	%rd1, %rd9, 3;
    setp.lt.u64 	%p2, %rd10, 3;
    mov.f32 	%f55, 0f40490FD0;
    @%p2 bra 	$L__BB0_4;
    sub.s64 	%rd12, %rd9, %rd1;
    mov.f32 	%f55, 0f40490FD0;
$L__BB0_3:
    fma.rn.f32 	%f30, %f54, %f53, %f55;
    fma.rn.f32 	%f31, %f53, %f55, %f30;
    fma.rn.f32 	%f32, %f55, %f30, %f31;
    fma.rn.f32 	%f33, %f30, %f31, %f32;
    fma.rn.f32 	%f34, %f31, %f32, %f33;
    fma.rn.f32 	%f35, %f32, %f33, %f34;
    fma.rn.f32 	%f36, %f33, %f34, %f35;
    fma.rn.f32 	%f37, %f34, %f35, %f36;
    fma.rn.f32 	%f38, %f35, %f36, %f37;
    fma.rn.f32 	%f54, %f36, %f37, %f38;
    fma.rn.f32 	%f53, %f37, %f38, %f54;
    fma.rn.f32 	%f55, %f38, %f54, %f53;
    add.s64 	%rd12, %rd12, -4;
    setp.ne.s64 	%p3, %rd12, 0;
    @%p3 bra 	$L__BB0_3;
$L__BB0_4:
    setp.eq.s64 	%p4, %rd1, 0;
    @%p4 bra 	$L__BB0_7;
    neg.s64 	%rd13, %rd1;
$L__BB0_6:
    fma.rn.f32 	%f54, %f54, %f53, %f55;
    fma.rn.f32 	%f53, %f53, %f55, %f54;
    fma.rn.f32 	%f55, %f55, %f54, %f53;
    add.s64 	%rd13, %rd13, 1;
    setp.ne.s64 	%p5, %rd13, 0;
    @%p5 bra 	$L__BB0_6;
$L__BB0_7:
    setp.neu.f32 	%p6, %f54, 0f42F6E979;
    @%p6 bra 	$L__BB0_9;
    add.f32 	%f39, %f53, 0f42F6E979;
    add.f32 	%f40, %f39, %f55;
    cvta.to.global.u64 	%rd11, %rd8;
    st.global.f32 	[%rd11], %f40;
$L__BB0_9:
    ret;
}
"""

BLOCKS = 1024
THREADS = 256
ITERS = 400000


def ck(res, what):
    if res != 0:
        raise RuntimeError("%s -> CUDA driver error %d" % (what, res & 0xFFFFFFFF))


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cu = ctypes.CDLL("nvcuda.dll")
    ck(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    name = ctypes.create_string_buffer(128)
    cu.cuDeviceGetName(name, 128, dev)
    print("[native] device:", name.value.decode())
    ctx = ctypes.c_void_p()
    ck(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
    mod = ctypes.c_void_p()
    ptx_buf = ctypes.create_string_buffer(PTX.encode("ascii") + b"\x00")
    ck(cu.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p)),
       "cuModuleLoadData")
    fn = ctypes.c_void_p()
    ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"burn"), "cuModuleGetFunction")
    d_out = ctypes.c_void_p()
    ck(cu.cuMemAlloc_v2(ctypes.byref(d_out), 4), "cuMemAlloc")

    arg0 = ctypes.c_void_p(d_out.value)
    iters = ctypes.c_ulonglong(ITERS)
    params = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.pointer(arg0), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(iters), ctypes.c_void_p),
    )

    def launch(n=1):
        for _ in range(n):
            ck(cu.cuLaunchKernel(fn, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, params, None),
               "cuLaunchKernel")
        ck(cu.cuCtxSynchronize(), "cuCtxSynchronize")

    launch(3)  # warmup
    flops_per_launch = BLOCKS * THREADS * ITERS * 3 * 2  # 3 FMA (2 flop each) per iter
    print("[native] stress %d blocks x %d threads, %d s ..." % (BLOCKS, THREADS, seconds))
    t0 = time.time()
    reps = 0
    while time.time() - t0 < seconds:
        launch(10)
        reps += 10
        el = time.time() - t0
        print("  t=%5.1fs  %.1f GFLOP/s (fp32)" % (el, reps * flops_per_launch / el / 1e9))
    el = time.time() - t0
    print("[native] done: %d launches, %.1f s, avg %.1f GFLOP/s"
          % (reps, el, reps * flops_per_launch / el / 1e9))


if __name__ == "__main__":
    main()
