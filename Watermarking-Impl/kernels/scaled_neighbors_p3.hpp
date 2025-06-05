#pragma once
#include <string>
inline const std::string scaled_neighbors_p3 = R"CLC(

__kernel void scaled_neighbors_p3(
    __read_only image2d_t image, 
    __global float* __restrict__ x_,
    __constant float* __restrict__ coeffs,
    __local float region[16 + 2][16 + 2]) //hold the 18 x 18 region for this 16 x 16 block
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const int localId = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    //image2d is transposed, so we read the opposite dimensions
    const int width = get_image_height(image), height = get_image_width(image);

    //load cooperatively the 18 x 18 region for this 16 x 16 block
    for (int i = localId; i < 324; i += get_local_size(0) * get_local_size(1))
    {
        const int tileRow = i / 18;
        const int tileCol = i % 18;
        const int globalX =  get_group_id(1) * get_local_size(1) + tileCol - 1;
        const int globalY = get_group_id(0) * get_local_size(0) + tileRow - 1;
        region[tileRow][tileCol] = read_imagef(image, sampler, (int2)(globalY, globalX)).x;
    }
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