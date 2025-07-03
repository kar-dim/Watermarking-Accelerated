#pragma once
#include <string>
inline const std::string helpers = R"CLC(

#define PAD           (WINDOW_SIZE / 2)
#define SHAREDSIZE    (16 + 2 * PAD)

inline void fillBlock(
    __global const float* __restrict__ input,
    __local float* __restrict__ sharedMem,
    const int width,
    const int height)
{
    for (int i = get_local_id(1) * get_local_size(0) + get_local_id(0); i < SHAREDSIZE * SHAREDSIZE; i += get_local_size(0) * get_local_size(1))
    {
        const int tileRow = i / SHAREDSIZE;
        const int tileCol = i % SHAREDSIZE;
        const int globalX = clamp((int)(get_group_id(1) * get_local_size(1) + tileCol - PAD), 0, width - 1);
        const int globalY = clamp((int)(get_group_id(0) * get_local_size(0) + tileRow - PAD), 0, height - 1);
        sharedMem[tileRow * SHAREDSIZE + tileCol] = input[globalX * height + globalY];
    }
}

)CLC";