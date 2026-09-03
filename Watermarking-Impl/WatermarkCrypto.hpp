#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <string>
#include <utility>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/*!
 *  \brief  Functions for watermark secure watermark generation (ChaCha20, Box-Muller transform and SHA-256)
 *  \author Dimitris Karatzas
 */
namespace WatermarkCrypto {

// convert a 64-bit int to a float strictly in the range (0, 1].
// only the top 24 bits are used, the result is always a multiple of 2^-24 in [2^-24, 1]
inline float toUniformFloat(const uint64_t x) { return (x >> 40) * 0x1.0p-24f + 0x1.0p-24f; }

// 2*pi for Box-Muller transform
inline constexpr float kTwoPi = 2.0f * std::numbers::pi_v<float>;

// convert 64-bit int to float strictly in the range (0, 1] and then convert it to Box-Muller normal distribution pair
inline std::pair<float, float> generateBoxMullerNormalPair(const uint64_t x1, const uint64_t x2) {
    const float radius = std::sqrt(-2.0f * std::log(toUniformFloat(x1)));
    const float theta = kTwoPi * toUniformFloat(x2);
    return {radius * std::cos(theta), radius * std::sin(theta)};
}

#if defined(__AVX2__)
namespace detail {

// AVX2 natural logarithm
inline __m256 logAvx2(const __m256 x) {
    // frexp: mantissa in [0.5, 1) plus the matching exponent
    const __m256i bits = _mm256_castps_si256(x);
    __m256 m = _mm256_or_ps(_mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x807FFFFF))), _mm256_castsi256_ps(_mm256_set1_epi32(0x3F000000)));
    __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7F800000)), 23), _mm256_set1_epi32(126)));

    // keep the mantissa near 1: if (m < sqrt(0.5)) { e -= 1; m = 2m - 1; } else { m -= 1; }
    const __m256 belowSqrtHalf = _mm256_cmp_ps(m, _mm256_set1_ps(0.707106781186547524f), _CMP_LT_OQ);
    e = _mm256_sub_ps(e, _mm256_and_ps(belowSqrtHalf, _mm256_set1_ps(1.0f)));
    m = _mm256_sub_ps(_mm256_add_ps(m, _mm256_and_ps(belowSqrtHalf, m)), _mm256_set1_ps(1.0f));

    const __m256 z = _mm256_mul_ps(m, m);
    __m256 y = _mm256_set1_ps(7.0376836292E-2f);
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(-1.1514610310E-1f));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(1.1676998740E-1f));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(-1.2420140846E-1f));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(1.4249322787E-1f));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(-1.6668057665E-1f));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(2.0000714765E-1f));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(-2.4999993993E-1f));
    y = _mm256_fmadd_ps(y, m, _mm256_set1_ps(3.3333331174E-1f));
    y = _mm256_mul_ps(_mm256_mul_ps(y, m), z);
    // recombine, using the hi/lo split of ln(2) for the exponent term
    y = _mm256_fmadd_ps(e, _mm256_set1_ps(-2.12194440e-4f), y);
    y = _mm256_fmadd_ps(z, _mm256_set1_ps(-0.5f), y);
    return _mm256_fmadd_ps(e, _mm256_set1_ps(0.693359375f), _mm256_add_ps(m, y));
}

// AVX2 sine and cosine, Cephes minimax polynomials
inline void sinCosAvx2(const __m256 x, __m256& sinOut, __m256& cosOut) {
    // reduce into an octant: j = ((int)(x * 4/pi) + 1) & ~1
    __m256 y = _mm256_mul_ps(x, _mm256_set1_ps(1.27323954473516f));
    const __m256i j = _mm256_and_si256(_mm256_add_epi32(_mm256_cvttps_epi32(y), _mm256_set1_epi32(1)), _mm256_set1_epi32(~1));
    y = _mm256_cvtepi32_ps(j);

    // extended precision modular arithmetic: r = ((x - y*DP1) - y*DP2) - y*DP3
    __m256 r = _mm256_fmadd_ps(y, _mm256_set1_ps(-0.78515625f), x);
    r = _mm256_fmadd_ps(y, _mm256_set1_ps(-2.4187564849853515625e-4f), r);
    r = _mm256_fmadd_ps(y, _mm256_set1_ps(-3.77489497744594108e-8f), r);

    // the octant bits decide which polynomial is sin and which is cos, and the sign of each
    const __m256i j2 = _mm256_and_si256(j, _mm256_set1_epi32(2));
    const __m256 swapSinSign = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(j, _mm256_set1_epi32(4)), 29));
    const __m256 polyMask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(j2, _mm256_setzero_si256()));
    const __m256 cosSign = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_andnot_si256(_mm256_sub_epi32(j, _mm256_set1_epi32(2)), _mm256_set1_epi32(4)), 29));

    // cosine polynomial of the reduced argument
    const __m256 z = _mm256_mul_ps(r, r);
    __m256 cosPoly = _mm256_set1_ps(2.443315711809948E-005f);
    cosPoly = _mm256_fmadd_ps(cosPoly, z, _mm256_set1_ps(-1.388731625493765E-003f));
    cosPoly = _mm256_fmadd_ps(cosPoly, z, _mm256_set1_ps(4.166664568298827E-002f));
    cosPoly = _mm256_mul_ps(_mm256_mul_ps(cosPoly, z), z);
    cosPoly = _mm256_fmadd_ps(z, _mm256_set1_ps(-0.5f), cosPoly);
    cosPoly = _mm256_add_ps(cosPoly, _mm256_set1_ps(1.0f));

    // sin polynomial of the reduced argument
    __m256 sinPoly = _mm256_set1_ps(-1.9515295891E-4f);
    sinPoly = _mm256_fmadd_ps(sinPoly, z, _mm256_set1_ps(8.3321608736E-3f));
    sinPoly = _mm256_fmadd_ps(sinPoly, z, _mm256_set1_ps(-1.6666654611E-1f));
    sinPoly = _mm256_fmadd_ps(_mm256_mul_ps(sinPoly, z), r, r);

    // pick the right polynomial per lane, then apply the octant signs
    sinOut = _mm256_xor_ps(_mm256_blendv_ps(cosPoly, sinPoly, polyMask), swapSinSign);
    cosOut = _mm256_xor_ps(_mm256_blendv_ps(sinPoly, cosPoly, polyMask), cosSign);
}

// take the top 24 bits of four uint64 lanes into the low 128 bits as four int32
inline __m128i unpackTop24(const __m256i v) {
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_srli_epi64(v, 40), _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)));
}

} // namespace detail

// vectorized Box-Muller over two ChaCha20 blocks (16 uint64 = 8 pairs), writing 16 floats.
inline void generateBoxMullerNormalBlockPair(const std::array<uint64_t, 8>& block0, const std::array<uint64_t, 8>& block1, float* dst) {
    // top 24 bits of every uint64, as int32, four per register
    const __m128i k0 = detail::unpackTop24(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block0.data())));
    const __m128i k1 = detail::unpackTop24(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block0.data() + 4)));
    const __m128i k2 = detail::unpackTop24(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block1.data())));
    const __m128i k3 = detail::unpackTop24(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block1.data() + 4)));

    // deinterleave into the x1 (even) and x2 (odd) member of every pair
    const __m128 even0 = _mm_shuffle_ps(_mm_castsi128_ps(k0), _mm_castsi128_ps(k1), _MM_SHUFFLE(2, 0, 2, 0));
    const __m128 odd0 = _mm_shuffle_ps(_mm_castsi128_ps(k0), _mm_castsi128_ps(k1), _MM_SHUFFLE(3, 1, 3, 1));
    const __m128 even1 = _mm_shuffle_ps(_mm_castsi128_ps(k2), _mm_castsi128_ps(k3), _MM_SHUFFLE(2, 0, 2, 0));
    const __m128 odd1 = _mm_shuffle_ps(_mm_castsi128_ps(k2), _mm_castsi128_ps(k3), _MM_SHUFFLE(3, 1, 3, 1));

    // exactly toUniformFloat: both the multiply and the add are exact for a 24-bit integer
    const __m256 scale = _mm256_set1_ps(0x1.0p-24f);
    const __m256 u1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_castps_si256(_mm256_set_m128(even1, even0))), scale, scale);
    const __m256 u2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_castps_si256(_mm256_set_m128(odd1, odd0))), scale, scale);

    const __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), detail::logAvx2(u1)));
    __m256 sinTheta, cosTheta;
    detail::sinCosAvx2(_mm256_mul_ps(_mm256_set1_ps(kTwoPi), u2), sinTheta, cosTheta);
    const __m256 z0 = _mm256_mul_ps(radius, cosTheta);
    const __m256 z1 = _mm256_mul_ps(radius, sinTheta);

    // interleave back into (z0, z1) pairs: block0's 8 floats, then block1's 8 floats
    const __m256 lo = _mm256_unpacklo_ps(z0, z1);
    const __m256 hi = _mm256_unpackhi_ps(z0, z1);
    _mm256_storeu_ps(dst, _mm256_permute2f128_ps(lo, hi, 0x20));
    _mm256_storeu_ps(dst + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}
#endif // __AVX2__

// clang-format off
// SHA-256 hash reference implementation
inline std::array<uint8_t, 32> sha256(const std::string& input) {
    // SHA-256 Constants
    static constexpr std::array<uint32_t, 64> K = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    // initial hash values
    std::array<uint32_t, 8> H = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    // pre-process (pad) the string
    const uint64_t bitlen = input.length() * 8;
    std::vector<uint8_t> data(input.begin(), input.end());
    data.push_back(0x80); // append 1 bit (0x80 in byte form)
    // pad with zeros until length in bits is 448 (mod 512)
    while ((data.size() * 8) % 512 != 448)
        data.push_back(0x00);
    // append original length in bits as a 64-bit big-endian integer
    for (int i = 7; i >= 0; --i)
        data.push_back(static_cast<uint8_t>((bitlen >> (i * 8)) & 0xFF));

    // process each 512-bit (64-byte) chunk
    for (size_t i = 0; i < data.size(); i += 64) {
        std::array<uint32_t, 64> W;
        for (int t = 0; t < 16; ++t)
            W[t] = (static_cast<uint32_t>(data[i + t * 4]) << 24) | (static_cast<uint32_t>(data[i + t * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(data[i + t * 4 + 2]) << 8) | static_cast<uint32_t>(data[i + t * 4 + 3]);
        for (int t = 16; t < 64; ++t) {
            uint32_t s0 = std::rotr(W[t - 15], 7) ^ std::rotr(W[t - 15], 18) ^ (W[t - 15] >> 3);
            uint32_t s1 = std::rotr(W[t - 2], 17) ^ std::rotr(W[t - 2], 19) ^ (W[t - 2] >> 10);
            W[t] = W[t - 16] + s0 + W[t - 7] + s1;
        }
        uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
        for (int t = 0; t < 64; ++t) {
            uint32_t S1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h + S1 + ch + K[t] + W[t];
            uint32_t S0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }
    // produce the final 32-byte hash
    std::array<uint8_t, 32> hash;
    for (int i = 0; i < 8; ++i) {
        hash[i * 4] = (H[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (H[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (H[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = (H[i] >> 0) & 0xFF;
    }
    return hash;
}
// clang-format on

// constructs the immutable base state (64 bytes) for ChaCha20 once (optimization)
inline std::array<uint32_t, 16> computeBaseState(const std::string& watermarkPassword) {
    // ChaCha20 constants (first 16 bytes)
    std::array<uint32_t, 16> state = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};
    // hash the password with SHA-256 and copy the 32 bytes into the "key slot" (indices 4-11)
    std::memcpy(&state[4], sha256(watermarkPassword).data(), 32);
    // block counter (indices 12-13) and nonce (indices 14-15) remain 0 for now (injected per block), last 16 bytes
    return state;
}

// clang-format off
// ChaCha20 QR helper function, performs ARX (Add, Rotate, XOR)
inline void QR(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = std::rotl(d, 16);
    c += d; b ^= c; b = std::rotl(b, 12);
    a += b; d ^= a; d = std::rotl(d, 8);
    c += d; b ^= c; b = std::rotl(b, 7);
}
// clang-format on

// 256-bit ChaCha20 block function, generates 64 bytes (eight 64-bit ints) of cryptographically secure noise based on a key string and a counter
// note: this implements the original DJB ChaCha20 specification (64-bit counter, 64-bit nonce)
inline std::array<uint64_t, 8> chacha20Block(const std::array<uint32_t, 16>& baseState, const uint64_t blockCounter) {
    std::array<uint32_t, 16> workingState;
    // copy the initial state
    std::memcpy(workingState.data(), baseState.data(), 64);

    // inject the block counter for this specific OpenMP thread
    const uint32_t c0 = static_cast<uint32_t>(blockCounter & 0xFFFFFFFF);
    const uint32_t c1 = static_cast<uint32_t>(blockCounter >> 32);
    workingState[12] = c0;
    workingState[13] = c1;

    // 20 rounds
    for (int i = 0; i < 10; i++) {
        QR(workingState[0], workingState[4], workingState[8], workingState[12]);
        QR(workingState[1], workingState[5], workingState[9], workingState[13]);
        QR(workingState[2], workingState[6], workingState[10], workingState[14]);
        QR(workingState[3], workingState[7], workingState[11], workingState[15]);
        QR(workingState[0], workingState[5], workingState[10], workingState[15]);
        QR(workingState[1], workingState[6], workingState[11], workingState[12]);
        QR(workingState[2], workingState[7], workingState[8], workingState[13]);
        QR(workingState[3], workingState[4], workingState[9], workingState[14]);
    }
    // ChaCha final addition
    for (int i = 0; i < 12; i++)
        workingState[i] += baseState[i];
    workingState[14] += baseState[14];
    workingState[15] += baseState[15];
    workingState[12] += c0;
    workingState[13] += c1;

    // output
    return std::bit_cast<std::array<uint64_t, 8>>(workingState);
}
} // namespace WatermarkCrypto
