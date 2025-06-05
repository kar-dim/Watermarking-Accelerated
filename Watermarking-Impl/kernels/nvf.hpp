#pragma once
#include <string>
inline const std::string nvf = R"CLC(

__kernel void nvf(__read_only image2d_t image, 
	__global float *nvf,
	__local float region[16 + 2 * (p/2)][16 + 2 * (p/2)])
{	
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	//image2d is transposed, so we read the opposite dimensions
	const int width = get_image_height(image), height = get_image_width(image);
	const int x = get_global_id(1), y = get_global_id(0);
    const int localId = get_local_id(1) * get_local_size(0) + get_local_id(0);
	//p is supplied at compile-time to allow the compiler to optimize more efficiently
	const int pSquared = p * p;
	const int pad = p / 2;
    const int sharedSize = 16 + (2 * pad);

    //load cooperatively the region for this 16 x 16 block
    for (int i = localId; i < sharedSize * sharedSize; i += get_local_size(0) * get_local_size(1))
    {
        const int tileRow = i / sharedSize;
        const int tileCol = i % sharedSize;
        const int globalX =  get_group_id(1) * get_local_size(1) + tileCol - pad;
        const int globalY = get_group_id(0) * get_local_size(0) + tileRow - pad;
        region[tileRow][tileCol] = read_imagef(image, sampler, (int2)(globalY, globalX)).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (y >= height || x >= width)
        return;

	// Local (shared memory) coordinates for center pixel
    const int centerCol = get_local_id(1) + pad;
    const int centerRow = get_local_id(0) + pad;

	float sum = 0.0f, sumSq = 0.0f;
	for (int i = -pad; i <= pad; i++)
	{
		for (int j = -pad; j <= pad; j++)
		{
			float pixelValue = region[centerRow + i][centerCol + j];
			sum += pixelValue;
			sumSq += pixelValue * pixelValue;
		}
	}
	float mean = sum / pSquared;
	float variance = (sumSq / pSquared) - (mean * mean);
	//calculate mask and write pixel value
	nvf[(x * height) + y] = variance / (1 + variance);
}
)CLC";