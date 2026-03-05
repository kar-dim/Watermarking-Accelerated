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

/*!
 *  \brief  Functions for watermark secure watermark generation (ChaCha20, Box-Muller transform and SHA-256)
 *  \author Dimitris Karatzas
 */
namespace WatermarkCrypto {

// convert 64-bit int to float strictly in the range (0, 1]
// and then convert to Box-Muller polar pair
inline std::pair<float, float> generateBoxMullerPair(const uint64_t x1, const uint64_t x2) {
    const float u1 = (x1 >> 40) * 0x1.0p-24f + 0x1.0p-24f;
    const float u2 = (x2 >> 40) * 0x1.0p-24f + 0x1.0p-24f;
    return {std::sqrt(-2.0f * std::log(u1)), 2.0f * std::numbers::pi_v<float> * u2};
}

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
            W[t] = (data[i + t * 4] << 24) | (data[i + t * 4 + 1] << 16) | (data[i + t * 4 + 2] << 8) | (data[i + t * 4 + 3]);
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

// constructs the immutable base state for ChaCha20 once (optimization)
inline std::array<uint32_t, 16> computeBaseState(const std::string& watermarkPassword) {
    std::array<uint32_t, 16> state = {0};
    // ChaCha20 constants
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;
    // hash the password with SHA-256
    const std::array<uint8_t, 32> keyBytes = sha256(watermarkPassword);
    // copy the 32 bytes into the "key slot" (indices 4-11)
    std::memcpy(&state[4], keyBytes.data(), 32);
    // nonce (indices 14-15) remains 0 and block counter (indices 12-13) remains 0 for now (injected per block)
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

// 256-bit ChaCha20 block function, generates 64 bytes (eight 64-bit ints) of cryptographically secure noise based on a key string and a counter.
// note: this implements the original DJB ChaCha20 specification (64-bit counter, 64-bit nonce)
inline void chacha20Block(const std::array<uint32_t, 16>& baseState, const uint64_t blockCounter, uint64_t* out64) {
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
    std::memcpy(out64, workingState.data(), 64);
}
} // namespace WatermarkCrypto