#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <string>
#include <utility>

/*!
 *  \brief  Functions for watermark secure watermark generation (ChaCha20 and Box-Muller transform)
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

// constructs the immutable base state for ChaCha20 once (optimization)
inline std::array<uint32_t, 16> computeBaseState(const std::string& watermarkPassword) {
    std::array<uint32_t, 16> state = {0};
    // ChaCha20 constants
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;
    // XOR fold the password directly into bytes
    uint8_t keyBytes[32] = {0};
    for (size_t i = 0; i < watermarkPassword.length(); i++)
        keyBytes[i % 32] ^= static_cast<uint8_t>(watermarkPassword[i]);
    // copy the 32 bytes into the "key slot" (indices 4-11)
    std::memcpy(&state[4], keyBytes, 32);
    // nonce (indices 14-15) remains 0 and block counter (indices 12-13) remains 0 for now (injected per block)
    return state;
}

// clang-format off
// ChaCha20 QR fhelper unction, performs ARX (Add, Rotate, XOR)
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
    uint32_t workingState[16];
    // copy the initial state
    std::memcpy(workingState, baseState.data(), 64);

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
    std::memcpy(out64, workingState, 64);
}
} // namespace WatermarkCrypto