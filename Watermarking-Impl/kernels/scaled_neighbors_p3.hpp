#pragma once
#include <string>
inline const std::string scaled_neighbors_p3 = R"CLC(

__kernel void scaled_neighbors_p3(
    __global const float* __restrict__ input, 
    __global float* __restrict__ x_,
    __constant float* __restrict__ coeffs,
    const unsigned int width,
    const unsigned int height,
    __local float region[16 + 2][16 + 2]) //hold the 18 x 18 region for this 16 x 16 block
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);

    fillBlock(input, region, width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    //calculate the dot product of the coefficients and the neighborhood for this pixel
    if (x < width && y < height) 
    {
        const int centerCol = get_local_id(1) + 1;
        const int centerRow = get_local_id(0) + 1;
        float dot = 0.0f;
        dot += coeffs[0] * region[centerRow - 1][centerCol - 1];
        dot += coeffs[1] * region[centerRow - 1][centerCol];
        dot += coeffs[2] * region[centerRow - 1][centerCol + 1];
        dot += coeffs[3] * region[centerRow][centerCol - 1];
        dot += coeffs[4] * region[centerRow][centerCol + 1];
        dot += coeffs[5] * region[centerRow + 1][centerCol - 1];
        dot += coeffs[6] * region[centerRow + 1][centerCol];
        dot += coeffs[7] * region[centerRow + 1][centerCol + 1];
		x_[(x * height + y)] = dot;
    }
}
)CLC";