#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>

/*!
 *  \brief  Functions for watermark secure watermark generation (ChaCha20 and Box-Muller transform)
 *  \author Dimitris Karatzas
 */
namespace WatermarkCrypto {

// convert 64-bit int to float strictly in the range (0, 1]
inline float toFloat(uint64_t x) { return (x >> 40) * 0x1.0p-24f + 0x1.0p-24f; }

// generate secret key (256-bit): XOR the arbitrary length string into exactly 32 bytes
inline std::array<uint32_t, 8> computeKeyFromPassword(const std::string& watermarkPassword) {
    uint8_t keyBytes[32] = {0};
    for (size_t i = 0; i < watermarkPassword.length(); i++)
        keyBytes[i % 32] ^= static_cast<uint8_t>(watermarkPassword[i]);
    std::array<uint32_t, 8> finalBytes;
    std::memcpy(finalBytes.data(), keyBytes, 32);
    return finalBytes;
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

// 256-bit ChaCha20 block function, generates 64 bytes (eight 64-bit ints) of cryptographically secure noise based on a key string and a counter.
inline void chacha20Block(const std::array<uint32_t, 16>& baseState, const uint64_t blockCounter, uint64_t* out64) {
    uint32_t working_state[16];
    // copy the initial state
    std::memcpy(working_state, baseState.data(), 64);

    // inject the block counter for this specific OpenMP thread
    const uint32_t c0 = static_cast<uint32_t>(blockCounter & 0xFFFFFFFF);
    const uint32_t c1 = static_cast<uint32_t>(blockCounter >> 32);
    working_state[12] = c0;
    working_state[13] = c1;

    // ARX Quarter Round lambda
    auto QR = [&](uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
        a += b;
        d ^= a;
        d = std::rotl(d, 16);
        c += d;
        b ^= c;
        b = std::rotl(b, 12);
        a += b;
        d ^= a;
        d = std::rotl(d, 8);
        c += d;
        b ^= c;
        b = std::rotl(b, 7);
    };

    // 20 rounds
    for (int i = 0; i < 10; i++) {
        QR(working_state[0], working_state[4], working_state[8], working_state[12]);
        QR(working_state[1], working_state[5], working_state[9], working_state[13]);
        QR(working_state[2], working_state[6], working_state[10], working_state[14]);
        QR(working_state[3], working_state[7], working_state[11], working_state[15]);
        QR(working_state[0], working_state[5], working_state[10], working_state[15]);
        QR(working_state[1], working_state[6], working_state[11], working_state[12]);
        QR(working_state[2], working_state[7], working_state[8], working_state[13]);
        QR(working_state[3], working_state[4], working_state[9], working_state[14]);
    }
    // ChaCha final addition
    for (int i = 0; i < 12; i++)
        working_state[i] += baseState[i];
    working_state[14] += baseState[14];
    working_state[15] += baseState[15];
    working_state[12] += c0;
    working_state[13] += c1;

    // output
    std::memcpy(out64, working_state, 64);
}
} // namespace WatermarkCrypto