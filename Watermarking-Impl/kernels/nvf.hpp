#pragma once
#include <string>
inline const std::string nvf = R"CLC(

__kernel void nvf(__global const float* __restrict__ input, 
	__global float* __restrict__ nvf,
	const unsigned int width,
    const unsigned int height,
	__local float region[16 + 2 * (WINDOW_SIZE/2)][16 + 2 * (WINDOW_SIZE/2)])
{	
	const int pad = WINDOW_SIZE / 2;
	const int pSquared = WINDOW_SIZE * WINDOW_SIZE;
    const int sharedSize = 16 + (2 * pad);
	const int x = get_global_id(1);
    const int y = get_global_id(0);
    const int localId = get_local_id(1) * get_local_size(0) + get_local_id(0);

	fillBlock(input, region, width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (y >= height || x >= width)
        return;

    const int shX = get_local_id(1) + pad;
    const int shY = get_local_id(0) + pad;

	float sum = 0.0f, sumSq = 0.0f;
	for (int i = -pad; i <= pad; i++)
	{
		for (int j = -pad; j <= pad; j++)
		{
			float pixelValue = region[shY + i][shX + j];
			sum += pixelValue;
			sumSq += pixelValue * pixelValue;
		}
	}
	float mean = sum / pSquared;
	float variance = (sumSq / pSquared) - (mean * mean);
	nvf[(x * height) + y] = variance / (1 + variance);
}
)CLC";