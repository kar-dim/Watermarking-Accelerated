#pragma once
#include <string>
inline const std::string kernels = R"CLC(

#define PAD               (WINDOW_SIZE / 2)
#define NEIGHB_SIZE       ((WINDOW_SIZE * WINDOW_SIZE) - 1)
#define N_PIXELS          (float) (WINDOW_SIZE * WINDOW_SIZE)
#define N_PIXELS_SQ       (N_PIXELS * N_PIXELS)
#define SHAREDSIZE        (16 + 2 * PAD)
#define SH_DIM_FAST       (32 + (2 * PAD))
#define SH_DIM_SLOW       (8 + (2 * PAD))

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline int2 getPackedCoords(int k) {
    const int r = (int)((sqrt(1.0f + 8.0f * k) - 1.0f) / 2.0f);
    const int c = k - (r * (r + 1)) / 2;
    return (int2)(r, c);
}

inline void fillBlock(
    const __global float* restrict input,
    __local float* restrict sharedMem,
    const int width,
    const int height) 
{
    const int baseGlobalX = (int)(get_group_id(1) * get_local_size(1)) - PAD; 
    const int baseGlobalY = (int)(get_group_id(0) * get_local_size(0)) - PAD;
    const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int totalThreads = get_local_size(0) * get_local_size(1); 
    const int totalElements = SH_DIM_FAST * SH_DIM_SLOW;
    
    for (int i = tid; i < totalElements; i += totalThreads) {
        const int r = i % SH_DIM_FAST;
        const int c = i / SH_DIM_FAST;
        const int globalX = clamp(baseGlobalX + c, 0, width - 1);
        const int globalY = clamp(baseGlobalY + r, 0, height - 1);
        sharedMem[i] = input[globalX * height + globalY];
    }
}

inline void fillBlockFused(
    const __global float* restrict inputA,
    const __global float* restrict inputB,
    __local float* restrict sharedMem,
    const int width,
    const int height) 
{
    const int baseGlobalX = (int)(get_group_id(1) * get_local_size(1)) - PAD; 
    const int baseGlobalY = (int)(get_group_id(0) * get_local_size(0)) - PAD;
    const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int totalThreads = get_local_size(0) * get_local_size(1); 
    const int totalElements = SH_DIM_FAST * SH_DIM_SLOW;
    
    for (int i = tid; i < totalElements; i += totalThreads) {
        const int r = i % SH_DIM_FAST;
        const int c = i / SH_DIM_FAST;
        const int globalX = clamp(baseGlobalX + c, 0, width - 1);
        const int globalY = clamp(baseGlobalY + r, 0, height - 1);
        const int idx = globalX * height + globalY;
        sharedMem[i] = inputA[idx] * inputB[idx];
    }
}

inline float compute_nvf_mask(
    __local float region[SH_DIM_SLOW][SH_DIM_FAST], 
    const int shSlow, const int shFast) 
{
    float sum = 0.0f, sumSq = 0.0f;

    #pragma unroll
    for (int i = -PAD; i <= PAD; i++) {
        #pragma unroll
        for (int j = -PAD; j <= PAD; j++) {
            const float pixelValue = region[shSlow + i][shFast + j];
            sum += pixelValue;
            sumSq += pixelValue * pixelValue;
        }
    }

    const float numerator = (N_PIXELS * sumSq) - (sum * sum);
    const float output = native_divide(numerator, N_PIXELS_SQ + numerator);
    return clamp(output, 0.0f, 1.0f);
}

__kernel void nvf(
    const __global float* restrict input, 
    __global float* restrict nvf, 
    const int width, 
    const int height) 
{    
    const int x = get_global_id(1);
    const int y = get_global_id(0);

    __local __attribute__((aligned(16))) float region[SH_DIM_SLOW][SH_DIM_FAST];

    fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (y >= height || x >= width)
        return;

    const int shSlow = get_local_id(1) + PAD;
    const int shFast = get_local_id(0) + PAD;
    nvf[(x * height) + y] = compute_nvf_mask(region, shSlow, shFast);
}

__kernel void nvf_u_and_partial_sumsq_fused(
    const __global float* restrict input,
    const __global float* restrict w,
    __global float* restrict u,
    __global float* restrict partials,
    const int width, const int height)
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    
    const int linearTid = get_local_id(1) * get_local_size(0) + get_local_id(0);

    __local __attribute__((aligned(16))) float region[SH_DIM_SLOW][SH_DIM_FAST];
    __local float sums[256];

    fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    float threadSumSq = 0.0f;

    if (x < width && y < height) {
        const int shSlow = get_local_id(1) + PAD;
        const int shFast = get_local_id(0) + PAD;
        const float maskVal = compute_nvf_mask(region, shSlow, shFast);
        const int idx = (x * height) + y;
        const float uVal = maskVal * w[idx];
        u[idx] = uVal;
        threadSumSq = uVal * uVal;
    }
    sums[linearTid] = threadSumSq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (linearTid < s)
            sums[linearTid] += sums[linearTid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (linearTid == 0)
        partials[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = sums[0];
}

__kernel void me_u_and_partial_sumsq_fused(
    __global const float* restrict errorSeq,
    __global const float* restrict w,
    __global float* restrict u,
    __global float* restrict partials,
    __global const float* restrict maxVal,
    const int N)
{
    const int tid = get_local_id(0);
    const int stride = get_global_size(0);
    const int gid = get_group_id(0);
    int idx = get_global_id(0);

    __local float sums[256];

    const float denom = (*maxVal) + 1.0e-6f;
    float localSumSq = 0.0f;

    while (idx < N) {
        const float maskVal = errorSeq[idx] / denom;
        const float uVal = maskVal * w[idx];
        u[idx] = uVal;
        localSumSq += uVal * uVal;
        
        idx += stride;
    }
    sums[tid] = localSumSq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s)
            sums[tid] += sums[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
        partials[gid] = sums[0];
}

//use pointer arithmetic for dot product to help compilers optimize address calculations fast
inline float error_sequence_coeffs_filter(__local float* centerPtr, __constant float* coeffs) {
#define P(slow, fast) centerPtr[(slow) * SH_DIM_FAST + (fast)]
    float dot = 0.0f;
    int k = 0;
#pragma unroll
    for (int i = -PAD; i <= PAD; i++) {
#pragma unroll
        for (int j = -PAD; j <= PAD; j++) {
            if (i == 0 && j == 0)
                continue;
            dot += coeffs[k] * P(j, i);
            k++;
        }
    }
    return P(0, 0) - dot;
#undef P
}

__kernel void error_sequence(
    __global const float* restrict input, 
    __global float* restrict x_,
    __constant float* restrict coeffs,
    const int width,
    const int height,
    const int calculateAbs,
    __constant int* restrict stopFlag) 
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);

    __local __attribute__((aligned(16))) float region[SH_DIM_SLOW][SH_DIM_FAST];
    __local float* centerPtr = &region[get_local_id(1) + PAD][get_local_id(0) + PAD];

    fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < width && y < height) {
        if (*stopFlag) {
            x_[x * height + y] = 0.0f;
            return;
        }
        const float output = error_sequence_coeffs_filter(centerPtr, coeffs);
        x_[x * height + y] = calculateAbs ? fabs(output) : output;
    }
}

__kernel void error_sequence_fused(
    __global const float* restrict inputA, 
    __global const float* restrict inputB,
    __global float* restrict x_,
    __constant float* restrict coeffs,
    const int width,
    const int height,
    __constant int* restrict stopFlag) 
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);

    __local __attribute__((aligned(16))) float region[SH_DIM_SLOW][SH_DIM_FAST];
    __local float* centerPtr = &region[get_local_id(1) + PAD][get_local_id(0) + PAD];
    
    fillBlockFused(inputA, inputB, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < width && y < height) {
        if (*stopFlag) { 
            x_[x * height + y] = 0.0f; 
            return; 
        }
        x_[x * height + y] = error_sequence_coeffs_filter(centerPtr, coeffs);
    }
}

__kernel void compute_u_and_partial_sumsq(
    __global const float* restrict mask,
    __global const float* restrict w,
    __global float* restrict u,
    __global float* restrict partials,
    const int N) 
{
    const int tid = get_local_id(0);
    const int stride = get_global_size(0);
    const int gid = get_group_id(0);
    int idx = get_global_id(0);
    
    __local float sums[256];

    float localSumSq = 0.0f;
    while (idx < N) {
        const float uVal = mask[idx] * w[idx];
        u[idx] = uVal;
        localSumSq += uVal * uVal;
        idx += stride;
    }
    sums[tid] = localSumSq;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s)
            sums[tid] += sums[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
        partials[gid] = sums[0];
}

__kernel void reduce_sumsq_partials(
    __global const float* restrict partials,
    __global float* restrict globalSumSq,
    const int numPartials)
{
    const int tid = get_local_id(0);
    __local float sums[256];

    float localSum = 0.0f;
    int idx = tid;
    while (idx < numPartials) {
        localSum += partials[idx];
        idx += 256;
    }
    sums[tid] = localSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s)
            sums[tid] += sums[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
        *globalSumSq = sums[0];
}

__kernel void apply_watermark_fused(
    __global const float* restrict input,
    __global const float* restrict u,
    __constant float* restrict sumSqPtr,
    __global unsigned char* restrict output,
    const float strengthNumerator,
    const int planeElements,
    const int numChannels) 
{
    const float uSumSquared = *sumSqPtr;
    float strength = (uSumSquared > 1e-12f) ? (strengthNumerator * rsqrt(uSumSquared)) : 0.0f;
    const int stride = get_global_size(0);
    int idx = get_global_id(0);
    while (idx < planeElements) {
        const float uStr = u[idx] * strength;
        for (int c = 0; c < numChannels; c++) {
            const int pixelIdx = idx + (c * planeElements);
            const float outputVal = clamp(input[pixelIdx] + uStr + 0.5f, 0.0f, 255.0f);
            output[pixelIdx] = (unsigned char)outputVal;
        }
        idx += stride;
    }
}

)CLC"
R"CLC(


// OpenCL upper packed Indexing:
// maps Lower diagonal coords (r, c) where r >= c to the corresponding upper diagonal packed index (c, r)
#define SOLVER_IDX(r, c) ((c * NEIGHB_SIZE) - (c * (c - 1)) / 2 + (r - c))

//compile time constants for me kernels (p=5,7 and 9, p=3 uses a different more optimal kernel)
#if WINDOW_SIZE == 5
#define N_ROWS 5
#define MAT_SIZE 300
#define ROW_STEP 1
#define COL_STEP 51
#define BUFFER_COLS 260
#define CENTER_IDX 12
#define ROW_CENTER 2
#define SHIFT_CENTER 2
#define TOTAL_TASKS 324
#define BOUNDARY 300

#elif WINDOW_SIZE == 7
#define N_ROWS 7
#define MAT_SIZE 1176
#define ROW_STEP 4
#define COL_STEP 36
#define BUFFER_COLS 262
#define CENTER_IDX 24
#define ROW_CENTER 3
#define SHIFT_CENTER 3
#define TOTAL_TASKS 1224
#define BOUNDARY 1176

#elif WINDOW_SIZE == 9
#define N_ROWS 9
#define MAT_SIZE 3240
#define ROW_STEP 4
#define COL_STEP 28
#define BUFFER_COLS 264
#define CENTER_IDX 40
#define ROW_CENTER 4
#define SHIFT_CENTER 4
#define TOTAL_TASKS 3320
#define BOUNDARY 3240
#endif

#if WINDOW_SIZE == 3
__kernel void me(__global const float* restrict input,
    __global float* restrict Rx,
    __global float* restrict rx,
    const int width,
    const int height)

{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int outputIndex = (y *  get_global_size(0)) + x;
    const int localId = get_local_id(0);
    const float halfScaleFactor = 0.00392156862f;

    __local __attribute__((aligned(16))) half RxLocal[256][40];
    __local half blockValues[3][258];
    __local float rxPartial[32][8];

    if (y >= height)
        return;

    const int totalPixels = 3 * 258;
    const int colStep = 85;
    const int rowStep = 1;
    const int baseGlobalCol = (get_group_id(0) * 256) - 1; 
    const int baseGlobalRow = get_group_id(1) - 1;
    int r = localId % 3;
    int c = localId / 3;
    int idx = localId;
    while (idx < totalPixels) {
        int gCol = clamp(baseGlobalCol + c, 0, (int)width - 1);
        int gRow = clamp(baseGlobalRow + r, 0, (int)height - 1);
        vstore_half(input[gCol * height + gRow] * halfScaleFactor, 0, &blockValues[r][c]);
        idx += 256;
        c += colStep;
        r += rowStep;
        if (r >= 3) {
            r -= 3;
            c += 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    half x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8;
    const bool isValid = (x < width);
    if (isValid) {
        const int localX = localId + 1;
        x_0 = blockValues[0][localX - 1];
        x_1 = blockValues[0][localX];
        x_2 = blockValues[0][localX + 1];
        x_3 = blockValues[1][localX - 1];
        x_4 = blockValues[1][localX];
        x_5 = blockValues[1][localX + 1];
        x_6 = blockValues[2][localX - 1];
        x_7 = blockValues[2][localX];
        x_8 = blockValues[2][localX + 1];
        __local half8* rowPtr = (__local half8*) &RxLocal[localId][0];
        *rowPtr = (half8)(x_0 * x_4, x_1 * x_4, x_2 * x_4, x_3 * x_4, x_5 * x_4, x_6 * x_4, x_7 * x_4, x_8 * x_4);
    }
    else
        vstore_half8((float8)(0.0f), 0, (__local half*)&RxLocal[localId][0]);
    barrier(CLK_LOCAL_MEM_FENCE);

    //OpenCL optimized rx summation
    const int col = localId & 7;
    const int rowStart = (localId >> 3) * 8; 
    half psum = 0.0h;
#pragma unroll
    for (int r = 0; r < 8; r++)
        psum += RxLocal[rowStart + r][col];
    rxPartial[localId >> 3][col] = (float)psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId < 8) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++)
            sum += rxPartial[i][localId];
        const int blockOffset = (y * get_num_groups(0) * 8) + (get_group_id(0) * 8);
        rx[blockOffset + localId] = sum;
    } //no barrier needed here

    if (isValid) {
        __local half8* rowPtr = (__local half8*) &RxLocal[localId][0];
        rowPtr[0] = (half8)(x_0*x_0, x_0*x_1, x_0*x_2, x_0*x_3, x_0*x_5, x_0*x_6, x_0*x_7, x_0*x_8);
        rowPtr[1] = (half8)(x_1*x_1, x_1*x_2, x_1*x_3, x_1*x_5, x_1*x_6, x_1*x_7, x_1*x_8, x_2*x_2);
        rowPtr[2] = (half8)(x_2*x_3, x_2*x_5, x_2*x_6, x_2*x_7, x_2*x_8, x_3*x_3, x_3*x_5, x_3*x_6);
        rowPtr[3] = (half8)(x_3*x_7, x_3*x_8, x_5*x_5, x_5*x_6, x_5*x_7, x_5*x_8, x_6*x_6, x_6*x_7);
        rowPtr[4] = (half8)(x_6*x_8, x_7*x_7, x_7*x_8, x_8*x_8, 0.0h,    0.0h,    0.0h,    0.0h);
    }
    else {
#pragma unroll
        for(int v=0; v<5; v++)
            vstore_half8((float8)(0.0f), 0, (__local half*)&RxLocal[localId][v*8]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //OpenCL optimized Rx summation
    //parallel partial summation with 252 threads active
    const int flatId = localId;
    if (flatId < 252) {
        const int col = flatId % 36;
        const int chunk = flatId / 36;
        const int rowsPerChunk = 37;
        const int startRow = chunk * rowsPerChunk;
        const int endRow = min(startRow + rowsPerChunk, 256);
        
        float pSum = 0.0f;
        for (int i = startRow; i < endRow; i++)
            pSum += (float)RxLocal[i][col];
        //smartly reuse rxPartial local memory here (for partial sum)
        ((__local float*)rxPartial)[flatId] = pSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //final summation by the first 36 threads
    if (flatId < 36) {
        float totalSum = 0.0f;
#pragma unroll
        for (int k = 0; k < 7; k++)
            totalSum += ((__local float*)rxPartial)[flatId + k * 36];
        const int blockOffset = (y * get_num_groups(0) * 36) + (get_group_id(0) * 36);
        Rx[blockOffset + flatId] = totalSum;
    }
}
#elif WINDOW_SIZE >= 5
__kernel void me(
__global const float* restrict input,
__global float* restrict Rx,
__global float* restrict rx,
const int width,
const int height) 
{
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int localId = get_local_id(0);
    const float halfScaleFactor = 0.00392156862f;

    __local half blockValues[N_ROWS][BUFFER_COLS]; 

    // load data
    const int loadLimit = N_ROWS * BUFFER_COLS; 
    const int radius = N_ROWS / 2;
    const int baseGlobalCol = (gx * 256) - radius; 
    const int baseGlobalRow = gy - radius;
    
    int r = localId % N_ROWS;
    int c = localId / N_ROWS;
    int idx = localId;
    
    while (idx < loadLimit) { 
        const int gCol = clamp(baseGlobalCol + c, 0, (int)width - 1);
        const int gRow = clamp(baseGlobalRow + r, 0, (int)height - 1);
        blockValues[r][c] = input[gCol * height + gRow] * halfScaleFactor;
        idx += 256;
        c += COL_STEP;
        r += ROW_STEP; 
        if (r >= N_ROWS) {
            r -= N_ROWS;
            c += 1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // vectorized compute of Rx and rx
    for (int k = localId; k < TOTAL_TASKS; k += 256) {
        float coeffSum = 0.0f;

        if (k < BOUNDARY) { 
            // Rx
            const int2 coords = getPackedCoords(k);
            const int r_idx = coords.x; 
            const int c_idx = coords.y; 
            const int cacheRow = (r_idx >= CENTER_IDX) ? r_idx + 1 : r_idx;
            const int cacheCol = (c_idx >= CENTER_IDX) ? c_idx + 1 : c_idx;
            const int rowA = cacheRow / N_ROWS;
            const int shiftA = cacheRow % N_ROWS;
            const int rowB = cacheCol / N_ROWS;
            const int shiftB = cacheCol % N_ROWS;

#pragma unroll
            for (int p = 0; p < 256; p += 4) {
                const float4 fa = vload_half4(0, (__local half*)&blockValues[rowA][p + shiftA]);
                const float4 fb = vload_half4(0, (__local half*)&blockValues[rowB][p + shiftB]);
                coeffSum += dot(fa, fb);
            }
            const int outputIndex = (gy * get_num_groups(0) * MAT_SIZE) + (gx * MAT_SIZE) + SOLVER_IDX(r_idx, c_idx);
            Rx[outputIndex] = coeffSum;
            
        } else {
            // rx
            const int nIdx = k - BOUNDARY;
            const int cacheIdx = (nIdx >= CENTER_IDX) ? nIdx + 1 : nIdx;
            const int rowN = cacheIdx / N_ROWS;
            const int shiftN = cacheIdx % N_ROWS;
#pragma unroll
            for (int p = 0; p < 256; p += 4) {
                const float4 fn = vload_half4(0, (__local half*)&blockValues[rowN][p + shiftN]);
                const float4 fc = vload_half4(0, (__local half*)&blockValues[ROW_CENTER][p + SHIFT_CENTER]);
                coeffSum += dot(fn, fc);
            }
            const int outputIndex = (gy * get_num_groups(0) * NEIGHB_SIZE) + (gx * NEIGHB_SIZE) + nIdx;
            rx[outputIndex] = coeffSum;
        }
    }
}
#undef N_ROWS
#undef MAT_SIZE
#undef COL_STEP
#undef BUFFER_COLS
#undef CENTER_IDX
#undef ROW_CENTER
#undef SHIFT_CENTER
#undef TOTAL_TASKS
#undef BOUNDARY
#endif

__kernel void reduce_abs_max_partials(
    __global const float* restrict errorSeq,
    __global float* restrict partialMax,
    const int N)
{
    const int tid = get_local_id(0);
    const int stride = get_global_size(0);
    int idx = get_global_id(0);

    __local float maxCache[256];

    float localMax = 0.0f;
    while (idx < N) {
        localMax = fmax(localMax, fabs(errorSeq[idx]));
        idx += stride;
    }
    maxCache[tid] = localMax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s)
            maxCache[tid] = fmax(maxCache[tid], maxCache[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
        partialMax[get_group_id(0)] = maxCache[0];
}

__kernel void reduce_abs_max_final(
    __global const float* restrict partialMax,
    __global float* restrict globalMax,
    const int numPartials)
{
    const int tid = get_local_id(0);

    __local float maxCache[256];

    float localMax = 0.0f;
    int idx = tid;
    while (idx < numPartials) {
        localMax = fmax(localMax, partialMax[idx]);
        idx += 256;
    }
    
    maxCache[tid] = localMax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s)
            maxCache[tid] = fmax(maxCache[tid], maxCache[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
        *globalMax = maxCache[0];
}

__kernel void compute_abs_normalized_mask(
    __global const float* restrict errorSeq,
    __global float* restrict mask,
    __constant float* restrict maxVal,
    const int N)
{
    const int stride = get_global_size(0);
    int idx = get_global_id(0);
    const float denom = *maxVal + 1.0e-6f;
    while (idx < N) {
        mask[idx] = fabs(errorSeq[idx]) / denom;
        idx += stride;
    }
}

__kernel void partial_reduce(
    __global const float* restrict input,
    __global float* restrict partials,
    const int stride,
    const int totalBlocks,
    const int blocksPerChunk)
{
    const int coeffIdx = get_global_id(0);
    const int chunkIdx = get_global_id(1);

    if (coeffIdx >= stride)
        return;

    const int startB = chunkIdx * blocksPerChunk;
    const int endB = min(startB + blocksPerChunk, totalBlocks);

    float sum = 0.0f;
    for (int b = startB; b < endB; ++b)
        sum += input[coeffIdx + b * stride];
    partials[coeffIdx * get_global_size(1) + chunkIdx] = sum;
}

__kernel void final_reduce(
    __global const float* restrict partials,
    __global float* restrict output,
    const int numChunks,
    const int stride)
{
    const int coeffIdx = get_group_id(0); 
    const int tid = get_local_id(0);
    const int wgSize = get_local_size(0);

    __local float sums[256]; 

    if (coeffIdx >= stride)
        return;

    float sum = 0.0f;
    for (int i = tid; i < numChunks; i += wgSize)
        sum += partials[coeffIdx * numChunks + i];
    sums[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = wgSize / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            sums[tid] += sums[tid + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
        output[coeffIdx] = sums[0];
}

)CLC"
R"CLC(

__kernel void partial_max_reduce(
    __global const float* restrict input,
    __global float* restrict partials,
    const int totalElements)
{
    const int tid = get_local_id(0);
    const int gid = get_global_id(0);
    const int stride = get_global_size(0);
    
    __local float scratch[256]; 

    float threadMax = 0.0f;
    for(int i = gid; i < totalElements; i += stride)
        threadMax = max(threadMax, input[i]);
    scratch[tid] = threadMax;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int offset = get_local_size(0) / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            scratch[tid] = max(scratch[tid], scratch[tid + offset]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
        partials[get_group_id(0)] = scratch[0];
}

__kernel void final_max_reduce(
    __global const float* restrict partials,
    __global float* restrict output,
    const int numPartials)
{
    const int tid = get_local_id(0);
    
    __local float scratch[256];

    float threadMax = 0.0f;
    for(int i = tid; i < numPartials; i += get_local_size(0))
        threadMax = max(threadMax, partials[i]);
    scratch[tid] = threadMax;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int offset = get_local_size(0) / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            scratch[tid] = max(scratch[tid], scratch[tid + offset]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (tid == 0)
        output[0] = scratch[0];
}

__kernel void calculate_error_sequence_and_partial_corr_fused(
    __global const float* restrict mask,
    __global const float* restrict w,
    __global const float* restrict e_u,
    __constant float* restrict coeffs,
    __global float* restrict partialDots,
    __global float* restrict partialNormU,
    __global float* restrict partialNormZ,
    const int width, const int height,
    const int calculateAbs,
    __constant int* restrict stopFlag)
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const int linearTid = get_local_id(1) * get_local_size(0) + get_local_id(0);

    __local __attribute__((aligned(16))) float region[SH_DIM_SLOW][SH_DIM_FAST];
    
    // Create the pointer directly to the center pixel of this thread's window
    __local float* centerPtr = &region[get_local_id(1) + PAD][get_local_id(0) + PAD];
    
    __local float dotCache[256];
    __local float normUCache[256];
    __local float normZCache[256];

    fillBlockFused(mask, w, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    float threadDot = 0.0f;
    float threadNormU = 0.0f;
    float threadNormZ = 0.0f;

    if (x < width && y < height && *stopFlag == 0) {
        const float errorSeq = error_sequence_coeffs_filter(centerPtr, coeffs);
        const float ez = calculateAbs ? fabs(errorSeq) : errorSeq;
        const float eu = e_u[(x * height) + y];
        threadDot = eu * ez;
        threadNormU = eu * eu;
        threadNormZ = ez * ez;
    }
    dotCache[linearTid] = threadDot;
    normUCache[linearTid] = threadNormU;
    normZCache[linearTid] = threadNormZ;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (linearTid < s) {
            dotCache[linearTid] += dotCache[linearTid + s];
            normUCache[linearTid] += normUCache[linearTid + s];
            normZCache[linearTid] += normZCache[linearTid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (linearTid == 0) {
        const int groupId1D = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        partialDots[groupId1D] = dotCache[0];
        partialNormU[groupId1D] = normUCache[0];
        partialNormZ[groupId1D] = normZCache[0];
    }
}

__kernel void calculate_final_correlation(
    __global const float* restrict partialDots,
    __global const float* restrict partialNormU,
    __global const float* restrict partialNormZ,
    __global float* restrict result,
    const int numBlocks) {

    const int tid = get_local_id(0);
    const int localSize = get_local_size(0);

    float localDot = 0.0f;
    float localU = 0.0f;
    float localZ = 0.0f;

    __local float sumDot[1024];
    __local float sumU[1024];
    __local float sumZ[1024];

    const size_t rawDots = (size_t)partialDots;
    const size_t rawU = (size_t)partialNormU;
    const size_t rawZ = (size_t)partialNormZ;
    if (((rawDots | rawU | rawZ) & 0xF) == 0) {
        __global const float4* vecDots = (__global const float4*)partialDots;
        __global const float4* vecU = (__global const float4*)partialNormU;
        __global const float4* vecZ = (__global const float4*)partialNormZ;

        const int vecLoopLimit = numBlocks >> 2; 
        for (int i = tid; i < vecLoopLimit; i += localSize) {
            const float4 d = vecDots[i];
            const float4 u = vecU[i];
            const float4 z = vecZ[i];
            localDot += d.x + d.y + d.z + d.w;
            localU += u.x + u.y + u.z + u.w;
            localZ += z.x + z.y + z.z + z.w;
        }
        for (int i = (vecLoopLimit << 2) + tid; i < numBlocks; i += localSize) {
            localDot += partialDots[i];
            localU += partialNormU[i];
            localZ += partialNormZ[i];
        }
    } 
    else {
        for (int i = tid; i < numBlocks; i += localSize) {
            localDot += partialDots[i];
            localU += partialNormU[i];
            localZ += partialNormZ[i];
        }
    }
    sumDot[tid] = localDot;
    sumU[tid] = localU;
    sumZ[tid] = localZ;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = localSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sumDot[tid] += sumDot[tid + s];
            sumU[tid] += sumU[tid + s];
            sumZ[tid] += sumZ[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        float final_dot = sumDot[0];
        float final_norm_u = sqrt(sumU[0]);
        float final_norm_z = sqrt(sumZ[0]);
        result[0] = (final_norm_u > 0.0f && final_norm_z > 0.0f) ? (final_dot / (final_norm_u * final_norm_z)) : 0.0f;
    }
}

// naive very low latency 1-thread solver
// define this kernel ONLY for p=3 and p=5
// do NOT define this for p=7 and p=9, opencl driver will hang when building!!
#if WINDOW_SIZE == 3 && WINDOW_SIZE == 5
__kernel void cholesky_solver(__global const float* restrict A, 
                                 __global const float* restrict B,
                                 __global float* restrict X,
                                 __global int* restrict stopFlag) {
    // packed array size: N * (N + 1) / 2
    #define PACKED_SIZE ((NEIGHB_SIZE * (NEIGHB_SIZE + 1)) / 2)
    
    if (get_local_id(0) > 0 || get_group_id(0) > 0)
        return;

    float __attribute__((aligned(16))) packed[PACKED_SIZE]; 
    float __attribute__((aligned(16))) localB[NEIGHB_SIZE];

    const bool isAligned = ((((size_t)A | (size_t)B | (size_t)X) & 0xF) == 0);
    if (isAligned) {
        __global const float4* vecA = (__global const float4*)A;
        __global const float4* vecB = (__global const float4*)B;
        const int vecLimitA = (PACKED_SIZE + 3) / 4;
        
#pragma unroll
        for (int k = 0; k < vecLimitA; k++) {
            const float4 v = vecA[k];
            packed[k * 4 + 0] = v.x;
            packed[k * 4 + 1] = v.y;
            packed[k * 4 + 2] = v.z;
            packed[k * 4 + 3] = v.w;
        }

        const int vecLimitB = (NEIGHB_SIZE + 3) / 4;
#pragma unroll
        for (int i = 0; i < vecLimitB; i++) {
            const float4 v = vecB[i];
            localB[i * 4 + 0] = v.x;
            localB[i * 4 + 1] = v.y;
            localB[i * 4 + 2] = v.z;
            localB[i * 4 + 3] = v.w;
        }
    } 
    else {
#pragma unroll
        for (int k = 0; k < PACKED_SIZE; k++)
            packed[k] = A[k];
#pragma unroll
        for (int i = 0; i < NEIGHB_SIZE; i++)
            localB[i] = B[i];
    }

    // Cholesky decomposition and solving
    for (int i = 0; i < NEIGHB_SIZE; i++) {
#pragma unroll
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < j; k++)
                sum += packed[SOLVER_IDX(i, k)] * packed[SOLVER_IDX(j, k)];

            if (i == j) {
                const float val = packed[SOLVER_IDX(i, i)] - sum;
                if (val <= 1e-12f) {
                    *stopFlag = 1;
                    goto exit; 
                }
                packed[SOLVER_IDX(i, i)] = sqrt(val);
            } else {
                packed[SOLVER_IDX(i, j)] = (packed[SOLVER_IDX(i, j)] - sum) / packed[SOLVER_IDX(j, j)];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NEIGHB_SIZE; i++) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < i; k++)
            sum += packed[SOLVER_IDX(i, k)] * localB[k];
        localB[i] = (localB[i] - sum) / packed[SOLVER_IDX(i, i)];
    }

#pragma unroll
    for (int i = NEIGHB_SIZE - 1; i >= 0; i--) {
        float sum = 0.0f;
#pragma unroll
        for (int k = i + 1; k < NEIGHB_SIZE; k++)
            sum += packed[SOLVER_IDX(k, i)] * localB[k];
        localB[i] = (localB[i] - sum) / packed[SOLVER_IDX(i, i)];
    }
    *stopFlag = 0;
exit:
    if (isAligned) {
        __global float4* vecX = (__global float4*)X;
        const int vecLimitB = NEIGHB_SIZE / 4;
#pragma unroll
        for (int i = 0; i < vecLimitB; i++) {
            float4 v;
            v.x = localB[i * 4 + 0];
            v.y = localB[i * 4 + 1];
            v.z = localB[i * 4 + 2];
            v.w = localB[i * 4 + 3];
            vecX[i] = v;
        }
        for (int i = vecLimitB * 4; i < NEIGHB_SIZE; i++)
            X[i] = localB[i];
    } else {
#pragma unroll
        for (int i = 0; i < NEIGHB_SIZE; i++)
            X[i] = localB[i];
    }
#undef PACKED_SIZE
}

#else
// parallel cholesky solver for p = 7 (N = 48) and p = 9 (N = 80), using one workgroup
__kernel void cholesky_solver(__global const float* restrict A, 
                              __global const float* restrict B,
                              __global float* restrict X,
                              __global int* restrict stopFlag) {

    __local float __attribute__((aligned(16))) sA[NEIGHB_SIZE][NEIGHB_SIZE + 1]; // +1 to avoid bank conflicts
    __local float __attribute__((aligned(16))) sB[NEIGHB_SIZE];

    const int tid = get_local_id(0);
    const int workers = get_local_size(0);

    // cooperative load of matrix A (Rx)
    // Rx is stored as col-major lower packed (SOLVER_IDX), we unpack it into sA[row][col]
    for (int c = 0; c < NEIGHB_SIZE; c++) {
        for (int r = c + tid; r < NEIGHB_SIZE; r += workers) {
            sA[r][c] = A[SOLVER_IDX(r, c)];
        }
    }

    // cooperative Load of Vector B (rx)
    for (int k = tid; k < NEIGHB_SIZE; k += workers)
        sB[k] = B[k];

    // initialize stopFlag (thread 0 only)
    if (tid == 0) 
        *stopFlag = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // cholesky decomposition (in-place on sA)
    for (int k = 0; k < NEIGHB_SIZE; k++) {
        // (thread 0 ) calculate the inverse diagonal to turn division into multiplication
        if (tid == 0) {
            const float diag = sA[k][k];
            if (diag <= 1e-12f) {
                *stopFlag = 1; 
                sA[k][k] = NAN; // signal!
            } else {
                sA[k][k] = 1 / sqrt(diag); // Store 1.0 / L_kk
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // check for abort
        const float invDiag = sA[k][k];
        if (isnan(invDiag))
            return;

        // scale column k (parallel)
        // L_ik = L_ik * (1/L_kk) for i > k
        for (int i = k + 1 + tid; i < NEIGHB_SIZE; i += workers)
            sA[i][k] *= invDiag;
        barrier(CLK_LOCAL_MEM_FENCE);

        // update trailing matrix (in parallel), parallelize i (row)
        for (int j = k + 1; j < NEIGHB_SIZE; j++) {
            const float L_jk = sA[j][k]; // Read L_jk once
            for (int i = j + tid; i < NEIGHB_SIZE; i += workers)
                sA[i][j] -= sA[i][k] * L_jk;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // forward Substitution (L * y = b)
    for (int k = 0; k < NEIGHB_SIZE; k++) {
        if (tid == 0)
            sB[k] *= sA[k][k]; // sA[k][k] is actually 1/L_kk stored from earlier
        barrier(CLK_LOCAL_MEM_FENCE);

        // broadcast y_k
        const float y_k = sB[k];

        // parallel update: b[i] -= L_ik * y_k
        for (int i = k + 1 + tid; i < NEIGHB_SIZE; i += workers)
            sB[i] -= sA[i][k] * y_k;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // backward Substitution (L^T * x = y)
    for (int k = NEIGHB_SIZE - 1; k >= 0; k--) {
        if (tid == 0)
            sB[k] *= sA[k][k]; 
        barrier(CLK_LOCAL_MEM_FENCE);

        // broadcast x_k
        const float x_k = sB[k];

        // parallel update: y[i] -= L_ki * x_k
        // L^T, row i is column i of L, we need L_ki, (we stored Lower), we access sA[k][i]
        for (int i = tid; i < k; i += workers)
            sB[i] -= sA[k][i] * x_k;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // store
    for (int i = tid; i < NEIGHB_SIZE; i += workers) {
        X[i] = sB[i];
    }
}
#endif

#undef SOLVER_IDX

)CLC";