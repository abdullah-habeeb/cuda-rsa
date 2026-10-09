// =============================================================================
//  bigint.cuh  -  Multi-precision (big integer) arithmetic for RSA
//
//  Every function here is  __host__ __device__ : the SAME source code is
//  compiled once for the CPU (sequential baseline) and once for the GPU
//  (CUDA kernels).  The CPU-vs-GPU comparison therefore measures the effect
//  of parallel hardware, not of two different algorithms.
//
//  Number representation
//    A big number is an array of NL 32-bit "limbs", little-endian
//    (limb 0 is the least significant).  RSA-2048 -> NL = 64 limbs.
//
//  Core algorithm: Montgomery multiplication (CIOS variant, Koc et al. 1996)
//    MontMul(a, b) = a * b * R^-1 mod n,   R = 2^(32*NL)
//    It replaces the expensive "divide by n" of a normal modular multiply by
//    cheap shifts, so modular exponentiation becomes ~3000 MontMuls.
//
//  Register-friendly convention used by mont_mul(r, a, b, ...):
//    * 'a' is only ever indexed with compile-time constants (fully unrolled
//      inner loops), so on the GPU it lives in REGISTERS.
//    * 'b' is indexed with the run-time outer-loop counter, so it lives in
//      (L1-cached) local memory.  Only NL reads of 'b' happen per MontMul.
// =============================================================================
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

// Loop-unrolling hints: honoured by nvcc for the device pass, ignored on host.
#if defined(__CUDA_ARCH__)
#define UNROLL_ALL _Pragma("unroll")
#define UNROLL_NONE _Pragma("unroll 1")
#else
#define UNROLL_ALL
#define UNROLL_NONE
#endif

// ---------------------------------------------------------------------------
//  Parameter blocks (plain structs so they can be placed in global, shared or
//  constant memory without any change to the arithmetic code).
// ---------------------------------------------------------------------------
template <int NL>
struct MontCtx {
    uint32_t n[NL];   // odd modulus
    uint32_t r2[NL];  // R^2 mod n   (converts a number into Montgomery form)
    uint32_t n0inv;   // -n^-1 mod 2^32
};

// One modular exponentiation "key": base^exp mod n
template <int NL>
struct ExpParams {
    MontCtx<NL> m;
    uint32_t exp[NL];
    int exp_bits;     // bit length of exp (index of highest set bit + 1)
};

// RSA private key in CRT form (Chinese Remainder Theorem), H = NL/2 limbs
template <int H>
struct CrtParams {
    ExpParams<H> P;      // modulus p, exponent dp = d mod (p-1)
    ExpParams<H> Q;      // modulus q, exponent dq = d mod (q-1)
    uint32_t qinv_r[H];  // (q^-1 mod p) * R mod p   (Montgomery form)
};

// ---------------------------------------------------------------------------
//  Small helpers
// ---------------------------------------------------------------------------
template <int NL>
HD void bn_copy(uint32_t* r, const uint32_t* a) {
    UNROLL_ALL
    for (int i = 0; i < NL; ++i) r[i] = a[i];
}

template <int NL>
HD void bn_set_word(uint32_t* r, uint32_t w) {
    UNROLL_ALL
    for (int i = 0; i < NL; ++i) r[i] = 0;
    r[0] = w;
}

// Branch-free: returns 1 if (hi:x) >= n else 0, where 'hi' is an extra top word.
template <int NL>
HD uint32_t bn_geq_with_top(const uint32_t* x, uint32_t hi, const uint32_t* n) {
    uint32_t br = 0;
    UNROLL_ALL
    for (int j = 0; j < NL; ++j) {
        const uint64_t s = (uint64_t)x[j] - n[j] - br;
        br = (uint32_t)(s >> 32) & 1u;          // 1 if the subtraction borrowed
    }
    return (hi >= br) ? 1u : 0u;                // (hi*2^(32NL) + x) - n >= 0 ?
}

// Branch-free: r = x - (n AND mask);  mask is 0 or 0xFFFFFFFF. Returns borrow.
template <int NL>
HD uint32_t bn_sub_masked(uint32_t* r, const uint32_t* x, const uint32_t* n, uint32_t mask) {
    uint32_t br = 0;
    UNROLL_ALL
    for (int j = 0; j < NL; ++j) {
        const uint64_t s = (uint64_t)x[j] - (n[j] & mask) - br;
        r[j] = (uint32_t)s;
        br = (uint32_t)(s >> 32) & 1u;
    }
    return br;
}

// If x >= n then x -= n  (requires x < 2n).  Constant-time, no branches.
template <int NL>
HD void bn_cond_sub(uint32_t* x, const uint32_t* n) {
    const uint32_t mask = 0u - bn_geq_with_top<NL>(x, 0u, n);
    bn_sub_masked<NL>(x, x, n, mask);
}

// r = (a + b) mod n   (requires a, b < n)
template <int NL>
HD void bn_mod_add(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* n) {
    uint32_t s[NL];
    uint32_t c = 0;
    UNROLL_ALL
    for (int j = 0; j < NL; ++j) {
        const uint64_t t = (uint64_t)a[j] + b[j] + c;
        s[j] = (uint32_t)t;
        c = (uint32_t)(t >> 32);
    }
    const uint32_t mask = 0u - bn_geq_with_top<NL>(s, c, n);
    bn_sub_masked<NL>(r, s, n, mask);
}

// r = (a - b) mod n   (requires a, b < n)
template <int NL>
HD void bn_mod_sub(uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* n) {
    uint32_t br = 0;
    UNROLL_ALL
    for (int j = 0; j < NL; ++j) {
        const uint64_t s = (uint64_t)a[j] - b[j] - br;
        r[j] = (uint32_t)s;
        br = (uint32_t)(s >> 32) & 1u;
    }
    const uint32_t mask = 0u - br;              // negative result -> add n back
    uint32_t c = 0;
    UNROLL_ALL
    for (int j = 0; j < NL; ++j) {
        const uint64_t s = (uint64_t)r[j] + (n[j] & mask) + c;
        r[j] = (uint32_t)s;
        c = (uint32_t)(s >> 32);
    }
}

// ---------------------------------------------------------------------------
//  Montgomery multiplication, CIOS (Coarsely Integrated Operand Scanning)
//
//    r = a * b * R^-1 mod n        with R = 2^(32*NL)
//
//  Pre-condition : a * b < R * n   (true if a,b < n, or if a < R and b < n)
//  Post-condition: 0 <= r < n      (fully reduced)
//  r may alias a (the result is written only at the very end).
// ---------------------------------------------------------------------------
template <int NL, class Ctx>
HD void mont_mul(uint32_t* r, const uint32_t* a, const uint32_t* b, const Ctx& ctx) {
    uint32_t t[NL + 2];
    UNROLL_ALL
    for (int j = 0; j < NL + 2; ++j) t[j] = 0;

    UNROLL_NONE
    for (int i = 0; i < NL; ++i) {
        // (1) t = t + a * b[i]
        const uint32_t bi = b[i];
        uint64_t C = 0;
        UNROLL_ALL
        for (int j = 0; j < NL; ++j) {
            const uint64_t s = (uint64_t)a[j] * bi + t[j] + C;   // max = 2^64 - 1, no overflow
            t[j] = (uint32_t)s;
            C = s >> 32;
        }
        uint64_t s = (uint64_t)t[NL] + C;
        t[NL] = (uint32_t)s;
        t[NL + 1] = (uint32_t)(s >> 32);

        // (2) m chosen so that t + m*n is divisible by 2^32;  then t = (t + m*n) / 2^32
        const uint32_t m = t[0] * ctx.n0inv;
        s = (uint64_t)m * ctx.n[0] + t[0];                      // low word is 0 by construction
        C = s >> 32;
        UNROLL_ALL
        for (int j = 1; j < NL; ++j) {
            s = (uint64_t)m * ctx.n[j] + t[j] + C;
            t[j - 1] = (uint32_t)s;
            C = s >> 32;
        }
        s = (uint64_t)t[NL] + C;
        t[NL - 1] = (uint32_t)s;
        t[NL] = t[NL + 1] + (uint32_t)(s >> 32);
    }

    // (3) t < 2n here: one conditional subtraction, done without branches
    const uint32_t mask = 0u - bn_geq_with_top<NL>(t, t[NL], ctx.n);
    bn_sub_masked<NL>(r, t, ctx.n, mask);
}

// ---------------------------------------------------------------------------
//  Modular exponentiation:  out = base ^ exp mod n
//  Left-to-right binary "square-and-multiply" in the Montgomery domain.
//  base may be any value < R (it is reduced by the first MontMul).
//
//  Because every connection on the server is decrypted with the SAME private
//  exponent d, all 32 threads of a warp take the same "if (bit)" decision at
//  the same time -> zero branch divergence on the GPU.
// ---------------------------------------------------------------------------
template <int NL, class P>
HD void mod_exp(uint32_t* out, const uint32_t* base, const P& prm) {
    uint32_t acc[NL];   // accumulator            (registers on the GPU)
    uint32_t xm[NL];    // base in Montgomery form (read as 'b' -> local memory)
    uint32_t tmp[NL];   // copy of acc used as 'b' operand when squaring

    if (prm.exp_bits <= 0) {             // x^0 = 1
        bn_set_word<NL>(out, 1u);
        return;
    }
    bn_copy<NL>(acc, base);
    mont_mul<NL>(xm, acc, prm.m.r2, prm.m);   // xm = base * R mod n
    bn_copy<NL>(acc, xm);                     // top bit of exp is always 1

    UNROLL_NONE
    for (int i = prm.exp_bits - 2; i >= 0; --i) {
        bn_copy<NL>(tmp, acc);
        mont_mul<NL>(acc, acc, tmp, prm.m);                       // square
        if ((prm.exp[i >> 5] >> (i & 31)) & 1u)
            mont_mul<NL>(acc, acc, xm, prm.m);                    // multiply
    }

    bn_set_word<NL>(tmp, 1u);
    mont_mul<NL>(out, acc, tmp, prm.m);       // leave Montgomery domain: acc * R^-1
}

// ---------------------------------------------------------------------------
//  x mod p for a 2H-limb x and an H-limb modulus p whose top bit is set.
//    x = xh * R + xl   ->   x mod p = (MontMul(xh, R^2) + (xl mod p)) mod p
// ---------------------------------------------------------------------------
template <int H>
HD void bn_reduce_double(uint32_t* r, const uint32_t* x, const MontCtx<H>& ctx) {
    uint32_t hi[H], lo[H];
    bn_copy<H>(hi, x + H);
    mont_mul<H>(hi, hi, ctx.r2, ctx);    // xh*R mod p   (xh < R, r2 < p: pre-condition holds)
    bn_copy<H>(lo, x);
    bn_cond_sub<H>(lo, ctx.n);           // xl < 2^(32H) <= 2p  ->  one subtraction suffices
    bn_mod_add<H>(r, hi, lo, ctx.n);
}

// out[2H] = h[H] * q[H] + add[H]   (schoolbook multiplication)
// Both loops are fully unrolled on the GPU: every index is a compile-time
// constant, so 'out' stays in registers.  (A version with a rolled outer loop,
// i.e. run-time indexed read-modify-write of out[i+j] in local memory, was
// compiled INCORRECTLY by NVCC 11.7 when inlined into the CRT kernel - see
// the report, section "Challenges".  The verification suite caught it.)
template <int H>
HD void bn_mul_add(uint32_t* out, const uint32_t* h, const uint32_t* q, const uint32_t* add) {
    UNROLL_ALL
    for (int k = 0; k < 2 * H; ++k) out[k] = (k < H) ? add[k] : 0u;
    UNROLL_ALL
    for (int i = 0; i < H; ++i) {
        const uint32_t hi = h[i];
        uint64_t C = 0;
        UNROLL_ALL
        for (int j = 0; j < H; ++j) {
            const uint64_t s = (uint64_t)hi * q[j] + out[i + j] + C;
            out[i + j] = (uint32_t)s;
            C = s >> 32;
        }
        out[i + H] = (uint32_t)C;        // this word is still 0 before row i
    }
}

// ---------------------------------------------------------------------------
//  RSA decryption with the Chinese Remainder Theorem (what real servers do):
//    m1 = c^dp mod p,  m2 = c^dq mod q,  h = qinv*(m1-m2) mod p,  m = m2 + h*q
//  Two half-size exponentiations ~ 3-4x less work than one full-size one.
// ---------------------------------------------------------------------------
template <int NL, class CP>
HD void rsa_crt_decrypt(uint32_t* m, const uint32_t* c, const CP& k) {
    constexpr int H = NL / 2;
    uint32_t cp[H], cq[H], m1[H], m2[H], h[H];

    bn_reduce_double<H>(cp, c, k.P.m);
    bn_reduce_double<H>(cq, c, k.Q.m);
    mod_exp<H>(m1, cp, k.P);
    mod_exp<H>(m2, cq, k.Q);
    bn_mod_sub<H>(h, m1, m2, k.P.m.n);        // m2 < q < p, so m2 is a valid residue mod p
    mont_mul<H>(h, h, k.qinv_r, k.P.m);       // h = (m1-m2) * qinv mod p
    bn_mul_add<H>(m, h, k.Q.m.n, m2);         // m = h*q + m2   (< n)
}
