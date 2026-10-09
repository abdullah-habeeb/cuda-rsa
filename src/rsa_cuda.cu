// =============================================================================
//  rsa_cuda.cu  -  CUDA Case Study: Cryptography (RSA Encryption / Decryption)
//
//  Scenario: a secure web server must decrypt thousands of RSA-2048 session
//  keys (one per incoming user connection) at the same time.  Every
//  decryption is an independent modular exponentiation  m = c^d mod n
//  -> "embarrassingly parallel":  ONE CUDA THREAD = ONE USER CONNECTION.
//
//  Modes
//    ./rsa_cuda demo   [N]   narrated end-to-end demo with readable messages
//    ./rsa_cuda verify [N]   correctness test suite (GPU vs CPU vs original)
//    ./rsa_cuda bench        performance experiments -> CSV files in results/
//    ./rsa_cuda info         GPU + kernel resource information
//    ./rsa_cuda all          demo + verify + bench   (default)
//  Options:  --out <dir>  (default "results")
// =============================================================================
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "rsa_key.h"
#include "bigint.cuh"

constexpr int NL = RSA_NLIMBS;   // 64 limbs x 32 bit = 2048-bit numbers
constexpr int HL = RSA_HLIMBS;   // 32 limbs = 1024-bit CRT halves
static_assert(NL == 2 * HL, "CRT needs n = p*q with equal-size primes");

using Key    = ExpParams<NL>;
using CrtKey = CrtParams<HL>;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n  -> %s\n", #call,         \
                    __FILE__, __LINE__, cudaGetErrorString(err_));                \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// -----------------------------------------------------------------------------
//  Keys (host copies)
// -----------------------------------------------------------------------------
static Key    g_pub;    // (n, e)  public key  -> encryption
static Key    g_priv;   // (n, d)  private key -> decryption
static CrtKey g_crt;    // (p, q, dp, dq, qinv) -> CRT decryption

// Constant-memory copies used by the GPU kernels (64 KB constant cache,
// broadcast to all threads of a warp that read the same address).
__constant__ Key    c_key;
__constant__ CrtKey c_crt;

static int bit_length(const uint32_t* x, int nl) {
    for (int i = nl - 1; i >= 0; --i)
        for (int b = 31; b >= 0; --b)
            if ((x[i] >> b) & 1u) return 32 * i + b + 1;
    return 0;
}

static void build_keys() {
    auto fill_ctx = [](auto& ctx, const uint32_t* n, const uint32_t* r2, uint32_t n0inv, int nl) {
        memcpy(ctx.n, n, nl * 4);
        memcpy(ctx.r2, r2, nl * 4);
        ctx.n0inv = n0inv;
        // sanity: n * n0inv == -1 (mod 2^32)
        if ((uint32_t)(n[0] * n0inv) != 0xFFFFFFFFu) {
            fprintf(stderr, "Key error: inconsistent Montgomery constant\n");
            exit(EXIT_FAILURE);
        }
    };
    fill_ctx(g_pub.m, RSA_N, RSA_R2N, RSA_N0INV, NL);
    memset(g_pub.exp, 0, sizeof(g_pub.exp));
    g_pub.exp[0] = RSA_E;
    g_pub.exp_bits = RSA_E_BITS;

    g_priv.m = g_pub.m;
    memcpy(g_priv.exp, RSA_D, sizeof(g_priv.exp));
    g_priv.exp_bits = RSA_D_BITS;

    fill_ctx(g_crt.P.m, RSA_P, RSA_R2P, RSA_P0INV, HL);
    memcpy(g_crt.P.exp, RSA_DP, sizeof(g_crt.P.exp));
    g_crt.P.exp_bits = RSA_DP_BITS;
    fill_ctx(g_crt.Q.m, RSA_Q, RSA_R2Q, RSA_Q0INV, HL);
    memcpy(g_crt.Q.exp, RSA_DQ, sizeof(g_crt.Q.exp));
    g_crt.Q.exp_bits = RSA_DQ_BITS;
    memcpy(g_crt.qinv_r, RSA_QINV_R, sizeof(g_crt.qinv_r));

    if (bit_length(g_pub.exp, NL) != g_pub.exp_bits || bit_length(g_priv.exp, NL) != g_priv.exp_bits ||
        bit_length(g_crt.P.exp, HL) != g_crt.P.exp_bits || bit_length(g_crt.Q.exp, HL) != g_crt.Q.exp_bits ||
        bit_length(RSA_N, NL) != RSA_KEY_BITS) {
        fprintf(stderr, "Key error: bit lengths in rsa_key.h are inconsistent\n");
        exit(EXIT_FAILURE);
    }
}

// =============================================================================
//  GPU KERNELS
//
//  Data layout ("structure of arrays" / interleaved):
//      limb l of message i is stored at  buf[l * stride + i]
//  so when thread i and thread i+1 read limb l they touch neighbouring
//  32-bit words -> every warp-wide load/store is fully COALESCED.
// =============================================================================
enum class KeyMem { Global, Shared, Constant };
enum class Op { Encrypt, Decrypt, DecryptCRT };

static const char* mem_name(KeyMem m) {
    return m == KeyMem::Global ? "global" : m == KeyMem::Shared ? "shared" : "constant";
}
static const char* op_name(Op o) {
    return o == Op::Encrypt ? "encrypt" : o == Op::Decrypt ? "decrypt" : "decrypt-CRT";
}

// Work done by ONE thread = ONE user connection
template <class K>
__device__ __forceinline__ void modexp_one_connection(uint32_t* __restrict__ out,
                                                      const uint32_t* __restrict__ in,
                                                      int idx, int stride, const K& key) {
    uint32_t x[NL], y[NL];
#pragma unroll
    for (int l = 0; l < NL; ++l) x[l] = in[(size_t)l * stride + idx];    // coalesced load
    mod_exp<NL>(y, x, key);                                              // y = x^exp mod n
#pragma unroll
    for (int l = 0; l < NL; ++l) out[(size_t)l * stride + idx] = y[l];   // coalesced store
}

// Plain modular-exponentiation kernel (encryption with e, or decryption with d).
// The template parameter selects WHERE the key (modulus, R^2, exponent) is read
// from, so the three levels of the memory hierarchy can be compared directly.
template <KeyMem MEM>
__global__ void rsa_modexp_kernel(uint32_t* __restrict__ out, const uint32_t* __restrict__ in,
                                  int first, int count, int stride, const Key* __restrict__ gkey) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;

    if constexpr (MEM == KeyMem::Shared) {
        // The whole block cooperatively stages the key in on-chip shared memory once.
        __shared__ Key s_key;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(gkey);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&s_key);
        for (int w = threadIdx.x; w < (int)(sizeof(Key) / 4); w += blockDim.x) dst[w] = src[w];
        __syncthreads();                                   // before any early exit!
        if (local < count) modexp_one_connection(out, in, first + local, stride, s_key);
    } else if constexpr (MEM == KeyMem::Global) {
        if (local < count) modexp_one_connection(out, in, first + local, stride, *gkey);
    } else {
        if (local < count) modexp_one_connection(out, in, first + local, stride, c_key);
    }
}

// CRT decryption kernel (key in constant memory) - what production servers use.
__global__ void rsa_crt_kernel(uint32_t* __restrict__ out, const uint32_t* __restrict__ in,
                               int first, int count, int stride) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;                            // boundary check
    const int idx = first + local;

    uint32_t c[NL], m[NL];
#pragma unroll
    for (int l = 0; l < NL; ++l) c[l] = in[(size_t)l * stride + idx];
    rsa_crt_decrypt<NL>(m, c, c_crt);
#pragma unroll
    for (int l = 0; l < NL; ++l) out[(size_t)l * stride + idx] = m[l];
}

// =============================================================================
//  HOST UTILITIES
// =============================================================================
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::string g_outdir = "results";
// Threads per kernel launch. On Windows/WSL (WDDM) the driver resets the GPU if
// one kernel runs longer than ~2 s (TDR watchdog), so a large batch is split
// into launches of ONE FULL WAVE each: exactly as many threads as the GPU can
// keep resident at once (from the occupancy calculator). 0 = automatic.
static int g_chunk = 0;
static int g_block = 64;

struct Rng {   // splitmix64 - small, fast, reproducible
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t u32() { return (uint32_t)(next() >> 32); }
};

// A batch of 'count' big numbers in interleaved layout
struct Batch {
    int count = 0;
    std::vector<uint32_t> w;
    explicit Batch(int c = 0) : count(c), w((size_t)NL * c, 0u) {}
    void get(int i, uint32_t* x) const { for (int l = 0; l < NL; ++l) x[l] = w[(size_t)l * count + i]; }
    void set(int i, const uint32_t* x) { for (int l = 0; l < NL; ++l) w[(size_t)l * count + i] = x[l]; }
    uint32_t* data() { return w.data(); }
    const uint32_t* data() const { return w.data(); }
};

// Uniformly random message m < n  (top limb strictly below n's top limb)
static void random_message(uint32_t* m, Rng& r) {
    for (int l = 0; l < NL; ++l) m[l] = r.u32();
    m[NL - 1] = r.u32() % RSA_N[NL - 1];
}

static Batch random_batch(int count, uint64_t seed) {
    Batch b(count);
    Rng r(seed);
    uint32_t m[NL];
    for (int i = 0; i < count; ++i) { random_message(m, r); b.set(i, m); }
    return b;
}

// text <-> big-endian integer (like PKCS#1 OS2IP / I2OSP, without padding)
static void text_to_bn(const std::string& s, uint32_t* x) {
    memset(x, 0, NL * 4);
    const int L = (int)s.size();
    for (int k = 0; k < L; ++k) {
        const int pos = L - 1 - k;                           // byte position from the LSB
        x[pos / 4] |= (uint32_t)(uint8_t)s[k] << (8 * (pos % 4));
    }
}
static std::string bn_to_text(const uint32_t* x) {
    std::string s;
    bool started = false;
    for (int pos = NL * 4 - 1; pos >= 0; --pos) {
        const uint8_t b = (uint8_t)(x[pos / 4] >> (8 * (pos % 4)));
        if (!started && b == 0) continue;
        started = true;
        s.push_back((b >= 32 && b < 127) ? (char)b : '?');
    }
    return s;
}
static std::string bn_hex(const uint32_t* x, int nl = NL) {
    std::string s;
    char buf[9];
    for (int i = nl - 1; i >= 0; --i) { snprintf(buf, sizeof buf, "%08x", x[i]); s += buf; }
    return s;
}

static int count_mismatches(const Batch& a, const Batch& b) {
    int bad = 0;
    uint32_t x[NL], y[NL];
    for (int i = 0; i < a.count; ++i) {
        a.get(i, x); b.get(i, y);
        if (memcmp(x, y, sizeof x) != 0) {
            if (getenv("RSA_DEBUG") && bad < 4) {
                int first_bad = -1;
                for (int l = 0; l < NL; ++l) if (x[l] != y[l]) { first_bad = l; break; }
                fprintf(stderr, "    mismatch at message %d (first differing limb %d)\n", i, first_bad);
            }
            ++bad;
        }
    }
    return bad;
}

static int max_cpu_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// =============================================================================
//  CPU BASELINE  (same bigint.cuh code, compiled for the host)
// =============================================================================
static double cpu_rsa_batch(Op op, const Batch& in, Batch& out, int threads) {
    const int count = in.count;
    const auto t0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
#endif
    for (int i = 0; i < count; ++i) {
        uint32_t x[NL], y[NL];
        in.get(i, x);
        if (op == Op::Encrypt)      mod_exp<NL>(y, x, g_pub);
        else if (op == Op::Decrypt) mod_exp<NL>(y, x, g_priv);
        else                        rsa_crt_decrypt<NL>(y, x, g_crt);
        out.set(i, y);
    }
    (void)threads;
    return ms_since(t0);
}

// =============================================================================
//  GPU PIPELINE - the standard 5 steps of every CUDA host program
// =============================================================================
struct GpuTimes {
    double alloc_ms = 0, h2d_ms = 0, kernel_ms = 0, d2h_ms = 0, free_ms = 0;
    double e2e() const { return h2d_ms + kernel_ms + d2h_ms; }   // data in -> results out
};

static void launch_rsa(Op op, KeyMem mem, uint32_t* d_out, const uint32_t* d_in,
                       int first, int n, int stride, const Key* d_key, int block) {
    const int grid = (n + block - 1) / block;   // ceil: may launch more threads than n
    if (op == Op::DecryptCRT) {
        rsa_crt_kernel<<<grid, block>>>(d_out, d_in, first, n, stride);
    } else if (mem == KeyMem::Global) {
        rsa_modexp_kernel<KeyMem::Global><<<grid, block>>>(d_out, d_in, first, n, stride, d_key);
    } else if (mem == KeyMem::Shared) {
        rsa_modexp_kernel<KeyMem::Shared><<<grid, block>>>(d_out, d_in, first, n, stride, d_key);
    } else {
        rsa_modexp_kernel<KeyMem::Constant><<<grid, block>>>(d_out, d_in, first, n, stride, d_key);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Number of threads the whole GPU can hold at once for this kernel + block size
static int resident_threads(Op op, KeyMem mem, int block) {
    int blocksPerSM = 0;
    if (op == Op::DecryptCRT)
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, rsa_crt_kernel, block, 0));
    else if (mem == KeyMem::Global)
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, rsa_modexp_kernel<KeyMem::Global>, block, 0));
    else if (mem == KeyMem::Shared)
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, rsa_modexp_kernel<KeyMem::Shared>, block, 0));
    else
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, rsa_modexp_kernel<KeyMem::Constant>, block, 0));
    int dev = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return std::max(1, blocksPerSM) * block * sms;
}

static GpuTimes gpu_rsa_batch(Op op, KeyMem mem, const Batch& in, Batch& out, int block,
                              bool narrate = false) {
    GpuTimes t;
    const int count = in.count;
    const size_t bytes = (size_t)NL * count * sizeof(uint32_t);
    const Key& key = (op == Op::Encrypt) ? g_pub : g_priv;
    const int chunk = g_chunk > 0 ? g_chunk : resident_threads(op, mem, block);

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    // ---- STEP 1: allocate device memory (host buffers already exist) --------
    auto c0 = Clock::now();
    uint32_t *d_in = nullptr, *d_out = nullptr;
    Key* d_key = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    if (op != Op::DecryptCRT && mem != KeyMem::Constant) CUDA_CHECK(cudaMalloc(&d_key, sizeof(Key)));
    CUDA_CHECK(cudaDeviceSynchronize());
    t.alloc_ms = ms_since(c0);
    if (narrate)
        printf("  [Step 1] cudaMalloc   : d_in, d_out = %zu bytes each (%d x 2048-bit numbers)\n", bytes, count);

    // ---- STEP 2: copy inputs (and the key) Host -> Device --------------------
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));
    if (op == Op::DecryptCRT)          CUDA_CHECK(cudaMemcpyToSymbol(c_crt, &g_crt, sizeof(CrtKey)));
    else if (mem == KeyMem::Constant)  CUDA_CHECK(cudaMemcpyToSymbol(c_key, &key, sizeof(Key)));
    else                               CUDA_CHECK(cudaMemcpy(d_key, &key, sizeof(Key), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    if (narrate)
        printf("  [Step 2] cudaMemcpy H->D: %d ciphertexts + key (%s memory)\n", count,
               op == Op::DecryptCRT ? "constant" : mem_name(mem));

    // ---- STEP 3: launch kernel  <<<grid, block>>>  (one thread per connection)
    for (int first = 0; first < count; first += chunk) {
        const int n = std::min(chunk, count - first);
        if (narrate)
            printf("  [Step 3] kernel %s<<<grid=%d, block=%d>>> [%s] -> %d threads for %d connections\n",
                   op == Op::DecryptCRT ? "rsa_crt_kernel" : "rsa_modexp_kernel",
                   (n + block - 1) / block, block, op_name(op), ((n + block - 1) / block) * block, n);
        launch_rsa(op, mem, d_out, d_in, first, n, count, d_key, block);
    }
    CUDA_CHECK(cudaEventRecord(e2));

    // ---- STEP 4: copy results Device -> Host ---------------------------------
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));
    if (narrate) printf("  [Step 4] cudaMemcpy D->H: %d results\n", count);

    float a, b, c;
    CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&b, e1, e2));
    CUDA_CHECK(cudaEventElapsedTime(&c, e2, e3));
    t.h2d_ms = a; t.kernel_ms = b; t.d2h_ms = c;

    // ---- STEP 5: free device memory -----------------------------------------
    c0 = Clock::now();
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    if (d_key) CUDA_CHECK(cudaFree(d_key));
    t.free_ms = ms_since(c0);
    if (narrate) printf("  [Step 5] cudaFree     : device buffers released\n");

    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2)); CUDA_CHECK(cudaEventDestroy(e3));
    return t;
}

// Median of several GPU runs (the first CUDA call of a process pays for context
// creation, so benchmarks always perform a warm-up first).
static GpuTimes gpu_median(Op op, KeyMem mem, const Batch& in, Batch& out, int block, int reps) {
    std::vector<GpuTimes> v;
    for (int r = 0; r < reps; ++r) v.push_back(gpu_rsa_batch(op, mem, in, out, block));
    std::sort(v.begin(), v.end(), [](const GpuTimes& x, const GpuTimes& y) { return x.kernel_ms < y.kernel_ms; });
    return v[v.size() / 2];
}

static void gpu_warmup() {
    Batch in = random_batch(64, 7), out(64);
    gpu_rsa_batch(Op::Encrypt, KeyMem::Constant, in, out, g_block);
}

// =============================================================================
//  MODE: info
// =============================================================================
template <class F>
static void kernel_line(FILE* f, const char* name, F kernel, int block) {
    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, kernel));
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, block, 0));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    const int active = blocksPerSM * block;
    fprintf(f, "  %-34s regs/thread=%3d  local(spill)=%4zu B  smem=%4zu B  maxThr/blk=%4d  "
               "block=%d -> %d blocks/SM = %d threads/SM (occupancy %.1f%%), %d resident threads on GPU\n",
            name, fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes, fa.maxThreadsPerBlock, block,
            blocksPerSM, active, 100.0 * active / p.maxThreadsPerMultiProcessor, active * p.multiProcessorCount);
}

static void print_info(FILE* f) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int rt = 0, drv = 0, clock_khz = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));   // works on CUDA 11, 12 and 13
    fprintf(f, "GPU            : %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    fprintf(f, "SMs            : %d, max %d threads/SM, %d regs/SM, warp size %d\n",
            p.multiProcessorCount, p.maxThreadsPerMultiProcessor, p.regsPerMultiprocessor, p.warpSize);
    fprintf(f, "Clock          : %d MHz, global memory %.0f MiB, L2 %d KiB, constant %zu KiB, shared/block %zu KiB\n",
            clock_khz / 1000, p.totalGlobalMem / 1048576.0, p.l2CacheSize / 1024, p.totalConstMem / 1024,
            p.sharedMemPerBlock / 1024);
    fprintf(f, "CUDA           : runtime %d.%d, driver supports %d.%d\n", rt / 1000, (rt % 1000) / 10, drv / 1000,
            (drv % 1000) / 10);
    fprintf(f, "CPU threads    : %d (OpenMP %s)\n", max_cpu_threads(),
#ifdef _OPENMP
            "enabled"
#else
            "disabled"
#endif
    );
    fprintf(f, "Kernel resources (block = %d):\n", g_block);
    kernel_line(f, "rsa_modexp_kernel<Global>", rsa_modexp_kernel<KeyMem::Global>, g_block);
    kernel_line(f, "rsa_modexp_kernel<Shared>", rsa_modexp_kernel<KeyMem::Shared>, g_block);
    kernel_line(f, "rsa_modexp_kernel<Constant>", rsa_modexp_kernel<KeyMem::Constant>, g_block);
    kernel_line(f, "rsa_crt_kernel", rsa_crt_kernel, g_block);
}

// =============================================================================
//  MODE: demo  - readable end-to-end walk-through
// =============================================================================
static int run_demo(int n) {
    static const char* users[] = {"alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi",
                                  "ivan", "judy", "mallory", "niaj", "olivia", "peggy", "rupert", "sybil"};
    printf("\n==================== DEMO: %d simultaneous HTTPS connections ====================\n", n);
    printf("Server RSA-%d public key: n = 0x%s...%s (%d bits), e = %u\n", RSA_KEY_BITS,
           bn_hex(RSA_N).substr(0, 16).c_str(), bn_hex(RSA_N).substr(NL * 8 - 8).c_str(),
           bit_length(RSA_N, NL), RSA_E);

    Batch plain(n), cipher(n), recovered(n), recovered_std(n);
    Rng r(2026);
    for (int i = 0; i < n; ++i) {
        char buf[160];
        snprintf(buf, sizeof buf, "conn#%04d user=%s session-key=%08x%08x%08x%08x", i + 1,
                 users[i % 16], r.u32(), r.u32(), r.u32(), r.u32());
        uint32_t x[NL];
        text_to_bn(buf, x);
        plain.set(i, x);
    }

    printf("\n(1) Clients encrypt their session keys with the server's PUBLIC key: c = m^e mod n\n");
    gpu_rsa_batch(Op::Encrypt, KeyMem::Constant, plain, cipher, g_block, true);

    printf("\n(2) Server decrypts ALL connections at once with its PRIVATE key (CRT): m = c^d mod n\n");
    GpuTimes t = gpu_rsa_batch(Op::DecryptCRT, KeyMem::Constant, cipher, recovered, g_block, true);
    printf("      GPU time: H2D %.3f ms | kernel %.3f ms | D2H %.3f ms\n", t.h2d_ms, t.kernel_ms, t.d2h_ms);

    printf("\n(3) Cross-check with plain (non-CRT) GPU decryption  m = c^d mod n\n");
    gpu_rsa_batch(Op::Decrypt, KeyMem::Constant, cipher, recovered_std, g_block);

    printf("\n%-5s | %-26s | %-66s | %s\n", "conn", "ciphertext (first 24 hex)", "decrypted plaintext", "check");
    printf("------+----------------------------+--------------------------------------------------------------------+------\n");
    int ok = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t m[NL], c[NL], d[NL], d2[NL];
        plain.get(i, m); cipher.get(i, c); recovered.get(i, d); recovered_std.get(i, d2);
        const bool good = !memcmp(m, d, sizeof m) && !memcmp(m, d2, sizeof m) && memcmp(m, c, sizeof m);
        ok += good;
        printf("%5d | %s.. | %-66s | %s\n", i + 1, bn_hex(c).substr(0, 24).c_str(), bn_to_text(d).c_str(),
               good ? "OK" : "FAIL");
    }

    // Tamper test: flip ONE bit of one ciphertext
    uint32_t c0[NL], m0[NL], dm[NL];
    cipher.get(0, c0);
    plain.get(0, m0);
    c0[5] ^= 1u << 7;
    rsa_crt_decrypt<NL>(dm, c0, g_crt);
    printf("\nTamper test: flipping 1 bit of ciphertext #1 decrypts to 0x%s...  (%s)\n",
           bn_hex(dm).substr(0, 32).c_str(),
           memcmp(dm, m0, sizeof dm) ? "garbage, as expected" : "UNEXPECTED");
    printf("\nDEMO RESULT: %d / %d connections decrypted correctly\n", ok, n);
    return ok == n ? 0 : 1;
}

// =============================================================================
//  MODE: verify  - correctness test suite
// =============================================================================
static bool report(const char* what, int bad, int total) {
    printf("  %-58s : %s (%d/%d match)\n", what, bad ? "FAIL" : "PASS", total - bad, total);
    return bad == 0;
}

static int run_verify(int n) {
    printf("\n==================== VERIFY: %d messages ====================\n", n);
    Batch M = random_batch(n, 12345);
    // edge cases: 0, 1, 2, n-1
    uint32_t x[NL];
    bn_set_word<NL>(x, 0); M.set(0, x);
    bn_set_word<NL>(x, 1); M.set(1, x);
    bn_set_word<NL>(x, 2); M.set(2, x);
    memcpy(x, RSA_N, sizeof x); x[0] -= 1; M.set(3, x);          // n is odd, so no borrow

    Batch C(n), Ccpu(n), D(n);
    bool all = true;
    const int T = max_cpu_threads();

    gpu_rsa_batch(Op::Encrypt, KeyMem::Constant, M, C, g_block);
    cpu_rsa_batch(Op::Encrypt, M, Ccpu, T);
    all &= report("GPU encryption == CPU encryption (c = m^e mod n)", count_mismatches(C, Ccpu), n);

    for (KeyMem mem : {KeyMem::Global, KeyMem::Shared, KeyMem::Constant}) {
        std::fill(D.w.begin(), D.w.end(), 0u);
        gpu_rsa_batch(Op::Decrypt, mem, C, D, g_block);
        char what[96];
        snprintf(what, sizeof what, "GPU decrypt, key in %s memory == original", mem_name(mem));
        all &= report(what, count_mismatches(D, M), n);
    }
    std::fill(D.w.begin(), D.w.end(), 0u);
    gpu_rsa_batch(Op::DecryptCRT, KeyMem::Constant, C, D, g_block);
    all &= report("GPU CRT decrypt == original", count_mismatches(D, M), n);

    std::fill(D.w.begin(), D.w.end(), 0u);
    cpu_rsa_batch(Op::DecryptCRT, C, D, T);
    all &= report("CPU CRT decrypt == original", count_mismatches(D, M), n);

    // Odd sizes exercise the "if (index < N)" boundary check (partial last block)
    for (int odd : {1, 31, 33, 65, 257, 999}) {
        if (odd > n) continue;
        Batch Ms(odd), Cs(odd), Ds(odd);
        for (int i = 0; i < odd; ++i) { M.get(i, x); Ms.set(i, x); }
        gpu_rsa_batch(Op::Encrypt, KeyMem::Constant, Ms, Cs, g_block);
        gpu_rsa_batch(Op::DecryptCRT, KeyMem::Constant, Cs, Ds, g_block);
        char what[96];
        snprintf(what, sizeof what, "Round trip with batch size %d (partial block)", odd);
        all &= report(what, count_mismatches(Ds, Ms), odd);
    }

    // Dump for the independent Python checker (tools/verify.py)
    std::filesystem::create_directories(g_outdir);
    const std::string path = g_outdir + "/verify_dump.txt";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { perror(path.c_str()); return 1; }
    fprintf(f, "n %s\ne %x\nd %s\n", bn_hex(RSA_N).c_str(), RSA_E, bn_hex(RSA_D).c_str());
    uint32_t c[NL];
    for (int i = 0; i < n; ++i) {
        M.get(i, x); C.get(i, c);
        fprintf(f, "%s %s\n", bn_hex(x).c_str(), bn_hex(c).c_str());
    }
    fclose(f);
    printf("  wrote %s (checked independently by tools/verify.py)\n", path.c_str());
    printf("VERIFY RESULT: %s\n", all ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all ? 0 : 1;
}

// =============================================================================
//  MODE: bench  - performance experiments
// =============================================================================
static FILE* open_csv(const char* name, const char* header) {
    std::filesystem::create_directories(g_outdir);
    const std::string path = g_outdir + "/" + name;
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { perror(path.c_str()); exit(EXIT_FAILURE); }
    fprintf(f, "%s\n", header);
    return f;
}

static void must_match(const Batch& a, const Batch& b, const char* ctx) {
    const int bad = count_mismatches(a, b);
    if (bad) { fprintf(stderr, "BENCH ABORTED: %d wrong results in %s\n", bad, ctx); exit(EXIT_FAILURE); }
}

static int run_bench() {
    printf("\n==================== BENCHMARKS ====================\n");
    const int T = max_cpu_threads();
    const int REPS = 3;
    gpu_warmup();

    {   // kernel resource usage
        std::filesystem::create_directories(g_outdir);
        FILE* f = fopen((g_outdir + "/device_info.txt").c_str(), "w");
        print_info(stdout);
        if (f) { print_info(f); fclose(f); }
    }

    // ---------------- Experiment 1: scaling with number of connections ------
    printf("\n[E1] Decryption time vs. number of simultaneous connections (block=%d)\n", g_block);
    printf("%6s | %11s %11s | %11s %11s | %11s %11s | %11s %11s\n", "conns", "GPU std", "GPU CRT",
           "CPU1 std", "CPU1 CRT", "CPU8 std", "CPU8 CRT", "GPUcrt e2e", "GPUstd e2e");
    FILE* f1 = open_csv("e1_scaling.csv",
                        "connections,gpu_std_kernel_ms,gpu_std_e2e_ms,gpu_crt_kernel_ms,gpu_crt_e2e_ms,"
                        "cpu1_std_ms,cpu1_crt_ms,cpuN_std_ms,cpuN_crt_ms,cpu_threads");
    const std::vector<int> sizes = {1, 8, 32, 64, 128, 256, 384, 512, 768, 1024, 1152, 1536, 2048};
    GpuTimes last_std, last_crt;
    for (int n : sizes) {
        Batch M = random_batch(n, 1000 + n), C(n), D(n);
        cpu_rsa_batch(Op::Encrypt, M, C, T);
        GpuTimes gs = gpu_median(Op::Decrypt, KeyMem::Constant, C, D, g_block, REPS);   must_match(D, M, "gpu std");
        GpuTimes gc = gpu_median(Op::DecryptCRT, KeyMem::Constant, C, D, g_block, REPS); must_match(D, M, "gpu crt");
        const double c1s = cpu_rsa_batch(Op::Decrypt, C, D, 1);     must_match(D, M, "cpu1 std");
        const double c1c = cpu_rsa_batch(Op::DecryptCRT, C, D, 1);  must_match(D, M, "cpu1 crt");
        const double cNs = cpu_rsa_batch(Op::Decrypt, C, D, T);     must_match(D, M, "cpuN std");
        const double cNc = cpu_rsa_batch(Op::DecryptCRT, C, D, T);  must_match(D, M, "cpuN crt");
        printf("%6d | %9.2fms %9.2fms | %9.1fms %9.1fms | %9.1fms %9.1fms | %9.2fms %9.2fms\n", n,
               gs.kernel_ms, gc.kernel_ms, c1s, c1c, cNs, cNc, gc.e2e(), gs.e2e());
        fprintf(f1, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, gs.kernel_ms, gs.e2e(), gc.kernel_ms,
                gc.e2e(), c1s, c1c, cNs, cNc, T);
        fflush(f1);
        last_std = gs; last_crt = gc;
    }
    fclose(f1);

    // ---------------- Experiment 2: time breakdown of the 5-step pipeline ---
    const int NB = sizes.back();
    FILE* f2 = open_csv("e2_breakdown.csv", "variant,connections,alloc_ms,h2d_ms,kernel_ms,d2h_ms,free_ms");
    fprintf(f2, "standard,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", NB, last_std.alloc_ms, last_std.h2d_ms,
            last_std.kernel_ms, last_std.d2h_ms, last_std.free_ms);
    fprintf(f2, "crt,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", NB, last_crt.alloc_ms, last_crt.h2d_ms, last_crt.kernel_ms,
            last_crt.d2h_ms, last_crt.free_ms);
    fclose(f2);
    printf("\n[E2] Pipeline breakdown @ %d connections (CRT): alloc %.3f | H2D %.3f | kernel %.2f | D2H %.3f | "
           "free %.3f ms\n", NB, last_crt.alloc_ms, last_crt.h2d_ms, last_crt.kernel_ms, last_crt.d2h_ms,
           last_crt.free_ms);

    // ---------------- Experiment 3: memory hierarchy for the key ------------
    const int NM = 1024;
    printf("\n[E3] Where should the key live?  (standard decryption, %d connections, block=%d)\n", NM, g_block);
    FILE* f3 = open_csv("e3_memory.csv", "key_memory,connections,kernel_ms,throughput_ops_per_s");
    {
        Batch M = random_batch(NM, 77), C(NM), D(NM);
        cpu_rsa_batch(Op::Encrypt, M, C, T);
        for (KeyMem mem : {KeyMem::Global, KeyMem::Shared, KeyMem::Constant}) {
            GpuTimes g = gpu_median(Op::Decrypt, mem, C, D, g_block, REPS);
            must_match(D, M, mem_name(mem));
            printf("  key in %-8s memory : kernel %9.2f ms  -> %8.1f decryptions/s\n", mem_name(mem), g.kernel_ms,
                   NM / (g.kernel_ms / 1000.0));
            fprintf(f3, "%s,%d,%.4f,%.2f\n", mem_name(mem), NM, g.kernel_ms, NM / (g.kernel_ms / 1000.0));
        }
    }
    fclose(f3);

    // ---------------- Experiment 4: block size ------------------------------
    printf("\n[E4] Effect of threads per block (%d connections)\n", NM);
    FILE* f4 = open_csv("e4_blocksize.csv", "block,connections,std_kernel_ms,crt_kernel_ms");
    {
        Batch M = random_batch(NM, 99), C(NM), D(NM);
        cpu_rsa_batch(Op::Encrypt, M, C, T);
        cudaFuncAttributes fa_s, fa_c;
        CUDA_CHECK(cudaFuncGetAttributes(&fa_s, rsa_modexp_kernel<KeyMem::Constant>));
        CUDA_CHECK(cudaFuncGetAttributes(&fa_c, rsa_crt_kernel));
        for (int b : {32, 64, 128, 256, 512}) {
            double s = -1, c = -1;
            if (b <= fa_s.maxThreadsPerBlock) {
                s = gpu_median(Op::Decrypt, KeyMem::Constant, C, D, b, REPS).kernel_ms;
                must_match(D, M, "block std");
            }
            if (b <= fa_c.maxThreadsPerBlock) {
                c = gpu_median(Op::DecryptCRT, KeyMem::Constant, C, D, b, REPS).kernel_ms;
                must_match(D, M, "block crt");
            }
            printf("  block %4d : standard %9.2f ms | CRT %9.2f ms%s\n", b, s, c,
                   (s < 0 || c < 0) ? "   (-1 = block too large for this kernel's register usage)" : "");
            fprintf(f4, "%d,%d,%.4f,%.4f\n", b, NM, s, c);
        }
    }
    fclose(f4);

    // ---------------- Experiment 5: encryption (public key, e = 65537) ------
    const int NE = 4096;
    printf("\n[E5] Encryption with e = 65537 (%d connections)\n", NE);
    FILE* f5 = open_csv("e5_encrypt.csv", "connections,gpu_kernel_ms,gpu_e2e_ms,cpu1_ms,cpuN_ms,cpu_threads");
    {
        Batch M = random_batch(NE, 5), C(NE), Cc(NE);
        GpuTimes g = gpu_median(Op::Encrypt, KeyMem::Constant, M, C, g_block, REPS);
        const double c1 = cpu_rsa_batch(Op::Encrypt, M, Cc, 1);
        must_match(C, Cc, "encrypt gpu vs cpu1");
        const double cN = cpu_rsa_batch(Op::Encrypt, M, Cc, T);
        must_match(C, Cc, "encrypt gpu vs cpuN");
        printf("  GPU kernel %.2f ms (e2e %.2f ms) | CPU 1 thread %.1f ms | CPU %d threads %.1f ms\n", g.kernel_ms,
               g.e2e(), c1, T, cN);
        fprintf(f5, "%d,%.4f,%.4f,%.4f,%.4f,%d\n", NE, g.kernel_ms, g.e2e(), c1, cN, T);
    }
    fclose(f5);
    printf("\nBENCH COMPLETE - CSV files written to %s/\n", g_outdir.c_str());
    return 0;
}

// =============================================================================
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered log output
    std::string mode = "all";
    int n = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) g_outdir = argv[++i];
        else if (a == "--chunk" && i + 1 < argc) g_chunk = std::max(1, atoi(argv[++i]));
        else if (a == "--block" && i + 1 < argc) g_block = std::max(32, atoi(argv[++i]));
        else if (a == "demo" || a == "verify" || a == "bench" || a == "info" || a == "all") mode = a;
        else if (atoi(a.c_str()) > 0) n = atoi(a.c_str());
        else { fprintf(stderr, "usage: %s [demo|verify|bench|info|all] [N] [--block B] [--chunk C] [--out DIR]\n", argv[0]); return 2; }
    }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { fprintf(stderr, "No CUDA device found\n"); return 1; }
    build_keys();

    int rc = 0;
    if (mode == "info")   { print_info(stdout); return 0; }
    if (mode == "demo"   || mode == "all") rc |= run_demo(n > 0 ? n : 8);
    if (mode == "verify" || mode == "all") rc |= run_verify(n > 0 ? n : 1000);
    if (mode == "bench"  || mode == "all") rc |= run_bench();
    return rc;
}
