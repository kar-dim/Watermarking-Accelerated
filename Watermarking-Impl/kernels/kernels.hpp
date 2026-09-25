#pragma once
#include <string>
// OpenCL kernels for ME prediction system, prediction errors, and watermark embedding
inline const std::string kernels = R"CLC(

#define PAD                 (WINDOW_SIZE / 2)
#define NEIGHB_SIZE         ((WINDOW_SIZE * WINDOW_SIZE) - 1)
#define WINDOW_CENTER       ((WINDOW_SIZE * WINDOW_SIZE) / 2)
#define N_PIXELS            (float) (WINDOW_SIZE * WINDOW_SIZE)
#define N_PIXELS_SQ         (N_PIXELS * N_PIXELS)
// Work-group dimensions for NVF kernels
#define NVF_ROWS            (WG_SIZE / 32)
#define SH_DIM_FAST         (32 + (2 * PAD))
#define SH_DIM_SLOW         (NVF_ROWS + (2 * PAD))

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define WM_INLINE static inline __attribute__((always_inline))

#if !defined(WM_DISABLE_SUBGROUPS)
#if defined(WM_INTEL_SUBGROUPS)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define WM_USE_SUBGROUPS 1
#elif defined(WM_KHR_SUBGROUPS)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define WM_USE_SUBGROUPS 1
#elif defined(__opencl_c_subgroups)
#define WM_USE_SUBGROUPS 1
#endif
#endif

// Memory fence helpers for cross-workgroup synchronization
#if defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 200 && (__OPENCL_C_VERSION__ < 300 || (defined(__opencl_c_atomic_scope_device) && defined(__opencl_c_atomic_order_acq_rel)))
#define DEVICE_RELEASE_FENCE() atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device)
#define DEVICE_ACQUIRE_FENCE() atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device)
#else
#define DEVICE_RELEASE_FENCE() mem_fence(CLK_GLOBAL_MEM_FENCE)
#define DEVICE_ACQUIRE_FENCE() mem_fence(CLK_GLOBAL_MEM_FENCE)
#endif

// Work-group reduction macros for sum and maximum
#if defined(WM_WORK_GROUP_REDUCTIONS)
#define REDUCTION_SCRATCH_STRIDE 1
#define REDUCE_SUM(TID, START_OFFSET, ARR, VALUE)                    \
    do {                                                             \
        const float _groupValue = work_group_reduce_add(VALUE);      \
        if ((TID) == 0)                                              \
            (ARR)[0] = _groupValue;                                  \
    } while (0)

#define REDUCE_MAX(TID, START_OFFSET, ARR, VALUE)                    \
    do {                                                             \
        const float _groupValue = work_group_reduce_max(VALUE);      \
        if ((TID) == 0)                                              \
            (ARR)[0] = _groupValue;                                  \
    } while (0)

#define REDUCE_SUM_3(TID, START_OFFSET, ARR1, ARR2, ARR3, VALUE1, VALUE2, VALUE3) \
    do {                                                                        \
        const float _groupValue1 = work_group_reduce_add(VALUE1);                \
        const float _groupValue2 = work_group_reduce_add(VALUE2);                \
        const float _groupValue3 = work_group_reduce_add(VALUE3);                \
        if ((TID) == 0) {                                                        \
            (ARR1)[0] = _groupValue1;                                            \
            (ARR2)[0] = _groupValue2;                                            \
            (ARR3)[0] = _groupValue3;                                            \
        }                                                                        \
    } while (0)
#elif defined(WM_USE_SUBGROUPS)
#define REDUCTION_SCRATCH_STRIDE get_num_sub_groups()
#define REDUCE_SUM(TID, START_OFFSET, ARR, VALUE)                               \
    do {                                                                        \
        const float _subgroupValue = sub_group_reduce_add(VALUE);               \
        if (get_sub_group_local_id() == 0)                                      \
            (ARR)[get_sub_group_id()] = _subgroupValue;                         \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
        if ((TID) == 0) {                                                       \
            float _groupValue = 0.0f;                                           \
            for (uint _group = 0; _group < get_num_sub_groups(); ++_group)      \
                _groupValue += (ARR)[_group];                                   \
            (ARR)[0] = _groupValue;                                             \
        }                                                                       \
    } while (0)

#define REDUCE_MAX(TID, START_OFFSET, ARR, VALUE)                               \
    do {                                                                        \
        const float _subgroupValue = sub_group_reduce_max(VALUE);               \
        if (get_sub_group_local_id() == 0)                                      \
            (ARR)[get_sub_group_id()] = _subgroupValue;                         \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
        if ((TID) == 0) {                                                       \
            float _groupValue = 0.0f;                                           \
            for (uint _group = 0; _group < get_num_sub_groups(); ++_group)      \
                _groupValue = fmax(_groupValue, (ARR)[_group]);                 \
            (ARR)[0] = _groupValue;                                             \
        }                                                                       \
    } while (0)

#define REDUCE_SUM_3(TID, START_OFFSET, ARR1, ARR2, ARR3, VALUE1, VALUE2, VALUE3) \
    do {                                                                        \
        const float _subgroupValue1 = sub_group_reduce_add(VALUE1);             \
        const float _subgroupValue2 = sub_group_reduce_add(VALUE2);             \
        const float _subgroupValue3 = sub_group_reduce_add(VALUE3);             \
        if (get_sub_group_local_id() == 0) {                                    \
            const uint _subgroup = get_sub_group_id();                          \
            (ARR1)[_subgroup] = _subgroupValue1;                                \
            (ARR2)[_subgroup] = _subgroupValue2;                                \
            (ARR3)[_subgroup] = _subgroupValue3;                                \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
        if ((TID) == 0) {                                                       \
            float _groupValue1 = 0.0f;                                          \
            float _groupValue2 = 0.0f;                                          \
            float _groupValue3 = 0.0f;                                          \
            for (uint _group = 0; _group < get_num_sub_groups(); ++_group) {    \
                _groupValue1 += (ARR1)[_group];                                 \
                _groupValue2 += (ARR2)[_group];                                 \
                _groupValue3 += (ARR3)[_group];                                 \
            }                                                                   \
            (ARR1)[0] = _groupValue1;                                           \
            (ARR2)[0] = _groupValue2;                                           \
            (ARR3)[0] = _groupValue3;                                           \
        }                                                                       \
    } while (0)
#else
#define REDUCTION_SCRATCH_STRIDE \
    (get_local_size(0) * get_local_size(1) * get_local_size(2))
#define REDUCE_SUM(TID, START_OFFSET, ARR, VALUE)                       \
    do {                                                                \
        (ARR)[(TID)] = (VALUE);                                         \
        barrier(CLK_LOCAL_MEM_FENCE);                                   \
        for (int _s = (START_OFFSET); _s > 0; _s >>= 1) {               \
            if ((TID) < _s) ARR[(TID)] += ARR[(TID) + _s];              \
            barrier(CLK_LOCAL_MEM_FENCE);                               \
        }                                                               \
    } while (0)

#define REDUCE_MAX(TID, START_OFFSET, ARR, VALUE)                       \
    do {                                                                \
        (ARR)[(TID)] = (VALUE);                                         \
        barrier(CLK_LOCAL_MEM_FENCE);                                   \
        for (int _s = (START_OFFSET); _s > 0; _s >>= 1) {               \
            if ((TID) < _s) ARR[(TID)] = fmax(ARR[(TID)], ARR[(TID) + _s]); \
            barrier(CLK_LOCAL_MEM_FENCE);                               \
        }                                                               \
    } while (0)

#define REDUCE_SUM_3(TID, START_OFFSET, ARR1, ARR2, ARR3, VALUE1, VALUE2, VALUE3) \
    do {                                                                \
        (ARR1)[(TID)] = (VALUE1);                                       \
        (ARR2)[(TID)] = (VALUE2);                                       \
        (ARR3)[(TID)] = (VALUE3);                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                   \
        for (int _s = (START_OFFSET); _s > 0; _s >>= 1) {               \
            if ((TID) < _s) {                                           \
                ARR1[(TID)] += ARR1[(TID) + _s];                        \
                ARR2[(TID)] += ARR2[(TID) + _s];                        \
                ARR3[(TID)] += ARR3[(TID) + _s];                        \
            }                                                           \
            barrier(CLK_LOCAL_MEM_FENCE);                               \
        }                                                               \
    } while (0)
#endif

WM_INLINE ulong toScaledUlong(const float value) { return (ulong)(value * 1000000000.0f); }
WM_INLINE float toUnscaledFloat(const ulong value) { return (float)(value) * 1.0e-9f; }
#define ME_MASK_PRESCALE (1.0f / 255.0f)

)CLC"
                                   R"CLC(

// Computes NVF mask for tests with one output per work-item
#define FILL_BLOCK_IMPL(LOAD_EXPR)                                             \
    const int baseGlobalX = (int)(get_group_id(1) * get_local_size(1)) - PAD;  \
    const int baseGlobalY = (int)(get_group_id(0) * get_local_size(0)) - PAD;  \
    const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);     \
    const int totalThreads = get_local_size(0) * get_local_size(1);            \
    const int totalElements = SH_DIM_FAST * SH_DIM_SLOW;                       \
                                                                               \
    for (int i = tid; i < totalElements; i += totalThreads) {                  \
        const int r = i % SH_DIM_FAST;                                         \
        const int c = i / SH_DIM_FAST;                                         \
        const int globalX = clamp(baseGlobalX + c, 0, width - 1);              \
        const int globalY = clamp(baseGlobalY + r, 0, height - 1);             \
        const int idx = globalX * height + globalY;                            \
        sharedMem[i] = (LOAD_EXPR);                                            \
    }

WM_INLINE void fillBlock(const __global float* restrict input, __local float* restrict sharedMem, const int width, const int height) {
    FILL_BLOCK_IMPL(input[idx])
}

WM_INLINE float compute_nvf_mask(__local float region[SH_DIM_SLOW][SH_DIM_FAST], const int shSlow, const int shFast) {
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


__kernel void nvf(const __global float* restrict input, __global float* restrict nvf, const int width, const int height) {
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    __local __attribute__((aligned(16))) float region[SH_DIM_SLOW][SH_DIM_FAST];
    fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (y >= height || x >= width)
        return;
    nvf[(x * height) + y] = compute_nvf_mask(region, get_local_id(1) + PAD, get_local_id(0) + PAD);
}

WM_INLINE float roundToHalf(const float value) {
    ushort bits;
    vstore_half_rte(value, 0, (__private half*)&bits);
    return vload_half(0, (__private const half*)&bits);
}

__kernel void nvf_u_and_sumsq_fused(const __global float* restrict input, const __global half* restrict w, __global half* restrict u, volatile __global ulong* restrict globalSumSq,
    const int width, const int height, __local float* restrict reductionScratch)
{
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const int linearTid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    __local __attribute__((aligned(16))) float region[SH_DIM_SLOW][SH_DIM_FAST];
    fillBlock(input, &region[0][0], width, height);
    barrier(CLK_LOCAL_MEM_FENCE);

    float threadSumSq = 0.0f;
    if (x < width && y < height) {
        const float maskVal = compute_nvf_mask(region, get_local_id(1) + PAD, get_local_id(0) + PAD);
        const int idx = (x * height) + y;
        const float uVal = roundToHalf(maskVal * vload_half(idx, w));
        vstore_half_rte(uVal, idx, u);
        threadSumSq = uVal * uVal;
    }
    REDUCE_SUM(linearTid, (get_local_size(0) * get_local_size(1)) / 2, reductionScratch, threadSumSq);
    if (linearTid == 0)
        atom_add(globalSumSq, toScaledUlong(reductionScratch[0]));
}

// Prediction error tile where each work-item processes 4 rows by 2 columns
// Reuses loaded window columns across outputs while keeping coefficients in registers
#define ET_ROWS             4
#define ET_COLS             2
#define ET_TILE_FAST        (32 * ET_ROWS)
#define ET_TILE_SLOW        ((WG_SIZE / 32) * ET_COLS)
// Per-item column window padded to float4 alignment
#define ET_WINDOW_FAST      (((ET_ROWS + (2 * PAD) + 3) / 4) * 4)
// Local memory tile covering padded windows across all work-items
#define ET_SH_FAST          (ET_TILE_FAST - ET_ROWS + ET_WINDOW_FAST)
#define ET_SH_SLOW          (ET_TILE_SLOW + (2 * PAD))
#define ET_SH_SIZE          (ET_SH_SLOW * ET_SH_FAST)
#define ET_FILL_ITERATIONS  ((ET_SH_SIZE + WG_SIZE - 1) / WG_SIZE)
#define ET_OUTPUTS          (ET_ROWS * ET_COLS)

// Fills local memory tile with image region using clamped edges
// Loads all values into registers before writing to local memory
WM_INLINE void fillErrorTile(const __global float* restrict inputA, const __global half* restrict inputB, const int mode, __local float* restrict region, const int tileSlow0,
    const int tileFast0, const int width, const int height)
{
    const int lid = get_local_id(0);
    float values[ET_FILL_ITERATIONS];
    #pragma unroll
    for (int it = 0; it < ET_FILL_ITERATIONS; it++) {
        const int i = lid + (it * WG_SIZE);
        if (i < ET_SH_SIZE) {
            const int r = i % ET_SH_FAST;
            const int c = i / ET_SH_FAST;
            const int idx = (clamp(tileSlow0 - PAD + c, 0, width - 1) * height) + clamp(tileFast0 - PAD + r, 0, height - 1);
            float value = inputA[idx];
            if (mode == 2)
                value = fabs(value);
            if (mode != 0)
                value *= vload_half(idx, inputB);
            values[it] = value;
        }
    }
    #pragma unroll
    for (int it = 0; it < ET_FILL_ITERATIONS; it++) {
        const int i = lid + (it * WG_SIZE);
        if (i < ET_SH_SIZE)
            region[i] = values[it];
    }
}

// Loads prediction coefficients into registers using float4 loads
WM_INLINE void loadCoefficients(const __global float* restrict coeffs, float* restrict coef) {
    #pragma unroll
    for (int v = 0; v < NEIGHB_SIZE / 4; v++) {
        const float4 c = vload4(v, coeffs);
        coef[(4 * v) + 0] = c.x;
        coef[(4 * v) + 1] = c.y;
        coef[(4 * v) + 2] = c.z;
        coef[(4 * v) + 3] = c.w;
    }
}

// Computes prediction errors for all outputs assigned to this work-item
WM_INLINE void predictionErrorTile(__local const float* restrict region, const float* restrict coef, const int slow0, const int fast0, float* restrict error) {
    float dot[ET_OUTPUTS];
    float pixel[ET_OUTPUTS];
    #pragma unroll
    for (int o = 0; o < ET_OUTPUTS; o++)
        dot[o] = 0.0f;
    #pragma unroll
    for (int wc = 0; wc < WINDOW_SIZE + ET_COLS - 1; wc++) {
        float window[ET_WINDOW_FAST];
        __local const float4* src = (__local const float4*)(region + ((slow0 + wc) * ET_SH_FAST) + fast0);
        #pragma unroll
        for (int v = 0; v < ET_WINDOW_FAST / 4; v++) {
            const float4 value = src[v];
            window[(4 * v) + 0] = value.x;
            window[(4 * v) + 1] = value.y;
            window[(4 * v) + 2] = value.z;
            window[(4 * v) + 3] = value.w;
        }
        #pragma unroll
        for (int c = 0; c < ET_COLS; c++) {
            const int i = wc - c;
            if (i < 0 || i >= WINDOW_SIZE)
                continue;
            #pragma unroll
            for (int j = 0; j < WINDOW_SIZE; j++) {
                const int k = (i * WINDOW_SIZE) + j;
                if (k == WINDOW_CENTER) {
                    #pragma unroll
                    for (int r = 0; r < ET_ROWS; r++)
                        pixel[(c * ET_ROWS) + r] = window[r + j];
                    continue;
                }
                #pragma unroll
                for (int r = 0; r < ET_ROWS; r++)
                    dot[(c * ET_ROWS) + r] = fma(coef[k - (k > WINDOW_CENTER)], window[r + j], dot[(c * ET_ROWS) + r]);
            }
        }
    }
    #pragma unroll
    for (int o = 0; o < ET_OUTPUTS; o++)
        error[o] = pixel[o] - dot[o];
}

// Loads 4 consecutive rows using float4 when aligned or scalar reads otherwise
WM_INLINE float4 loadRows4(const __global float* restrict input, const int x, const int y0, const int height) {
    const int idx = (x * height) + y0;
    if ((height & 3) == 0)
        return y0 < height ? vload4(0, input + idx) : (float4)(0.0f);
    return (float4)(y0 < height ? input[idx] : 0.0f, y0 + 1 < height ? input[idx + 1] : 0.0f, y0 + 2 < height ? input[idx + 2] : 0.0f, y0 + 3 < height ? input[idx + 3] : 0.0f);
}
WM_INLINE void storeRows4(__global float* restrict output, const int x, const int y0, const int height, const float4 value) {
    const int idx = (x * height) + y0;
    if ((height & 3) == 0) {
        if (y0 < height)
            vstore4(value, 0, output + idx);
        return;
    }
    if (y0 < height) output[idx] = value.x;
    if (y0 + 1 < height) output[idx + 1] = value.y;
    if (y0 + 2 < height) output[idx + 2] = value.z;
    if (y0 + 3 < height) output[idx + 3] = value.w;
}
WM_INLINE float4 loadRows4Half(const __global half* restrict input, const int x, const int y0, const int height) {
    const int idx = (x * height) + y0;
    if ((height & 3) == 0)
        return y0 < height ? vload_half4(0, input + idx) : (float4)(0.0f);
    return (float4)(y0 < height ? vload_half(idx, input) : 0.0f, y0 + 1 < height ? vload_half(idx + 1, input) : 0.0f, y0 + 2 < height ? vload_half(idx + 2, input) : 0.0f,
        y0 + 3 < height ? vload_half(idx + 3, input) : 0.0f);
}
WM_INLINE void storeRows4Half(__global half* restrict output, const int x, const int y0, const int height, const float4 value) {
    const int idx = (x * height) + y0;
    if ((height & 3) == 0) {
        if (y0 < height)
            vstore_half4_rte(value, 0, output + idx);
        return;
    }
    if (y0 < height) vstore_half_rte(value.x, idx, output);
    if (y0 + 1 < height) vstore_half_rte(value.y, idx + 1, output);
    if (y0 + 2 < height) vstore_half_rte(value.z, idx + 2, output);
    if (y0 + 3 < height) vstore_half_rte(value.w, idx + 3, output);
}

// Base coordinates for work-group and work-item outputs
#define ET_TILE_SLOW0 ((int)get_group_id(1) * ET_TILE_SLOW)
#define ET_TILE_FAST0 ((int)get_group_id(0) * ET_TILE_FAST)
#define ET_SLOW0 (((int)get_local_id(0) / 32) * ET_COLS)
#define ET_FAST0 (((int)get_local_id(0) % 32) * ET_ROWS)

)CLC"
                                   R"CLC(

// Computes prediction error across the image for detection
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1))) void calculate_error_sequence(__global const float* restrict input, __global float* restrict errorOut,
    __global const float* restrict coeffs, const int width, const int height, __global const int* restrict stopFlag)
{
    __local float4 regionStorage[ET_SH_SIZE / 4];
    __local float* region = (__local float*)regionStorage;
    float coef[NEIGHB_SIZE];
    fillErrorTile(input, 0, 0, region, ET_TILE_SLOW0, ET_TILE_FAST0, width, height);
    loadCoefficients(coeffs, coef);
    barrier(CLK_LOCAL_MEM_FENCE);

    float error[ET_OUTPUTS];
    #pragma unroll
    for (int o = 0; o < ET_OUTPUTS; o++)
        error[o] = 0.0f;
    if (!(*stopFlag))
        predictionErrorTile(region, coef, ET_SLOW0, ET_FAST0, error);
    const int x0 = ET_TILE_SLOW0 + ET_SLOW0;
    const int y0 = ET_TILE_FAST0 + ET_FAST0;
    #pragma unroll
    for (int c = 0; c < ET_COLS; c++)
        if (x0 + c < width)
            storeRows4(errorOut, x0 + c, y0, height, vload4(c, error));
}

// Computes prediction error, creates u = (|e| / 255) * w, and tracks energy sums
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1))) void me_error_sequence_u_sumsq_fused(__global const float* restrict input, __global const half* restrict w,
    __global half* restrict u, __global const float* restrict coeffs, volatile __global ulong* restrict globalSumSqMax, const int width, const int height,
    __global const int* restrict stopFlag, __local float* restrict reductionScratch)
{
    __local float4 regionStorage[ET_SH_SIZE / 4];
    __local float* region = (__local float*)regionStorage;
    float coef[NEIGHB_SIZE];
    fillErrorTile(input, 0, 0, region, ET_TILE_SLOW0, ET_TILE_FAST0, width, height);
    loadCoefficients(coeffs, coef);
    barrier(CLK_LOCAL_MEM_FENCE);

    float error[ET_OUTPUTS];
    #pragma unroll
    for (int o = 0; o < ET_OUTPUTS; o++)
        error[o] = 0.0f;
    if (!(*stopFlag)) // Skip computation if system solve failed
        predictionErrorTile(region, coef, ET_SLOW0, ET_FAST0, error);
    // Outputs outside image bounds contribute zero
    const int x0 = ET_TILE_SLOW0 + ET_SLOW0;
    const int y0 = ET_TILE_FAST0 + ET_FAST0;
    float threadSumSq = 0.0f;
    float threadMax = 0.0f;
    #pragma unroll
    for (int c = 0; c < ET_COLS; c++) {
        if (x0 + c >= width)
            continue;
        const float4 wv = loadRows4Half(w, x0 + c, y0, height);
        const float wr[4] = {wv.x, wv.y, wv.z, wv.w};
        float uv[4];
        #pragma unroll
        for (int r = 0; r < ET_ROWS; r++) {
            const float absError = y0 + r < height ? fabs(error[(c * ET_ROWS) + r]) : 0.0f;
            uv[r] = roundToHalf(absError * ME_MASK_PRESCALE * wr[r]);
            threadSumSq += uv[r] * uv[r];
            threadMax = fmax(threadMax, absError);
        }
        storeRows4Half(u, x0 + c, y0, height, (float4)(uv[0], uv[1], uv[2], uv[3]));
    }
    const int lid = get_local_id(0);
    REDUCE_SUM(lid, WG_SIZE / 2, reductionScratch, threadSumSq);
    const float groupSumSq = reductionScratch[0]; // Valid on first work-item only
    barrier(CLK_LOCAL_MEM_FENCE);
    REDUCE_MAX(lid, WG_SIZE / 2, reductionScratch, threadMax);
    if (lid == 0) {
        atom_add(globalSumSqMax, toScaledUlong(groupSumSq));
        // Update global maximum with 32-bit atomic max on float bit patterns
        atomic_max((volatile __global uint*)(globalSumSqMax + 1), as_uint(reductionScratch[0]));
    }
}

// Computes partial correlation sums and writes final result from the last finished work-group
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1))) void calculate_error_sequence_and_partial_corr_fused(__global const float* restrict mask,
    __global const half* restrict w, __global const float* restrict e_u, __global const float* restrict coeffs, __global float* restrict partialDots,
    __global float* restrict partialNormU, __global float* restrict partialNormZ, volatile __global uint* restrict groupCounter, __global float* restrict correlation,
    const int width, const int height, __global const int* restrict stopFlag, const int absMask, __local float* restrict reductionScratch)
{
    __local float4 regionStorage[ET_SH_SIZE / 4];
    __local float* region = (__local float*)regionStorage;
    __local int isLastGroup;
    float coef[NEIGHB_SIZE];
    fillErrorTile(mask, w, absMask ? 2 : 1, region, ET_TILE_SLOW0, ET_TILE_FAST0, width, height);
    loadCoefficients(coeffs, coef);
    barrier(CLK_LOCAL_MEM_FENCE);

    float threadDot = 0.0f;
    float threadNormU = 0.0f;
    float threadNormZ = 0.0f;
    if (!(*stopFlag)) {
        float ez[ET_OUTPUTS];
        predictionErrorTile(region, coef, ET_SLOW0, ET_FAST0, ez);
        const int x0 = ET_TILE_SLOW0 + ET_SLOW0;
        const int y0 = ET_TILE_FAST0 + ET_FAST0;
        #pragma unroll
        for (int c = 0; c < ET_COLS; c++) {
            if (x0 + c >= width)
                continue;
            // e_u reads zero outside image bounds
            const float4 euv = loadRows4(e_u, x0 + c, y0, height);
            const float eu[4] = {euv.x, euv.y, euv.z, euv.w};
            #pragma unroll
            for (int r = 0; r < ET_ROWS; r++) {
                const float z = y0 + r < height ? ez[(c * ET_ROWS) + r] : 0.0f;
                threadDot += eu[r] * z;
                threadNormU += eu[r] * eu[r];
                threadNormZ += z * z;
            }
        }
    }

    const int lid = get_local_id(0);
    const int numGroups = get_num_groups(0) * get_num_groups(1);
    __local float* dotCache = reductionScratch;
    __local float* normUCache = reductionScratch + REDUCTION_SCRATCH_STRIDE;
    __local float* normZCache = reductionScratch + (2 * REDUCTION_SCRATCH_STRIDE);
    REDUCE_SUM_3(lid, WG_SIZE / 2, dotCache, normUCache, normZCache, threadDot, threadNormU, threadNormZ);
    if (lid == 0) {
        const int groupId = (get_group_id(1) * get_num_groups(0)) + get_group_id(0);
        partialDots[groupId] = dotCache[0];
        partialNormU[groupId] = normUCache[0];
        partialNormZ[groupId] = normZCache[0];
        // Ensure group results are visible before atomic counter increment
        DEVICE_RELEASE_FENCE();
        isLastGroup = atomic_inc(groupCounter) == (uint)(numGroups - 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!isLastGroup)
        return;

    // Final work-group sums all group outputs and writes normalized correlation
    DEVICE_ACQUIRE_FENCE();
    volatile __global const float* dots = partialDots;
    volatile __global const float* normsU = partialNormU;
    volatile __global const float* normsZ = partialNormZ;
    float totalDot = 0.0f;
    float totalNormU = 0.0f;
    float totalNormZ = 0.0f;
    for (int i = lid; i < numGroups; i += WG_SIZE) {
        totalDot += dots[i];
        totalNormU += normsU[i];
        totalNormZ += normsZ[i];
    }
    REDUCE_SUM_3(lid, WG_SIZE / 2, dotCache, normUCache, normZCache, totalDot, totalNormU, totalNormZ);
    if (lid == 0) {
        const float normU = sqrt(normUCache[0]);
        const float normZ = sqrt(normZCache[0]);
        *correlation = (normU > 1e-12f && normZ > 1e-12f) ? (dotCache[0] / (normU * normZ)) : 0.0f;
        *groupCounter = 0; // Reset counter for next launch
    }
}

)CLC"
                                   R"CLC(

// Computes embedding strength and checks whether the image is flat
WM_INLINE float embedStrength(__global const ulong* restrict sumSqPtr, const int hasMaxAbs, const float strengthNumerator) {
    const float uSumSquared = toUnscaledFloat(sumSqPtr[0]);
    float normalizedSumSquared = uSumSquared;
    if (hasMaxAbs) {
        const float normFactor = 1.0f / ((as_float((uint)sumSqPtr[1]) + 1.0e-6f) * ME_MASK_PRESCALE);
        normalizedSumSquared *= normFactor * normFactor;
    }
    return (normalizedSumSquared > 1e-3f) ? (strengthNumerator * rsqrt(uSumSquared)) : 0.0f;
}

// Rounds and clamps float values to 8-bit pixels
WM_INLINE uchar toPixel(const float value) { return convert_uchar(clamp(value + 0.5f, 0.0f, 255.0f)); }
WM_INLINE uchar4 toPixels4(const float4 values) { return convert_uchar4(clamp(values + 0.5f, 0.0f, 255.0f)); }

// Adds watermark to RGB channels: output = input + strength * u
__kernel void apply_watermark_rgb(__global const uchar* restrict input, __global const half* restrict u, __global const ulong* restrict sumSqPtr, __global uchar* restrict output,
    const float strengthNumerator, const int planeElements, const int hasMaxAbs)
{
    const float strength = embedStrength(sumSqPtr, hasMaxAbs, strengthNumerator);
    const int stride = get_global_size(0);
    const int planeVectors = planeElements % 4 == 0 ? planeElements / 4 : 0;
    for (int v = get_global_id(0); v < planeVectors; v += stride) {
        const float4 us = vload_half4(v, u) * strength;
        for (int c = 0; c < 3; c++) {
            const int vectorIndex = v + c * planeVectors;
            vstore4(toPixels4(convert_float4(vload4(vectorIndex, input)) + us), vectorIndex, output);
        }
    }
    for (int i = planeVectors * 4 + get_global_id(0); i < planeElements; i += stride) {
        const float us = vload_half(i, u) * strength;
        for (int c = 0; c < 3; c++)
            output[i + c * planeElements] = toPixel(convert_float(input[i + c * planeElements]) + us);
    }
}

// Adds watermark to grayscale image: output = luma + strength * u
__kernel void apply_watermark_gray(__global const float* restrict input, __global const half* restrict u, __global const ulong* restrict sumSqPtr, __global uchar* restrict output,
    const float strengthNumerator, const int planeElements, const int hasMaxAbs)
{
    const float strength = embedStrength(sumSqPtr, hasMaxAbs, strengthNumerator);
    const int stride = get_global_size(0);
    const int planeVectors = planeElements % 4 == 0 ? planeElements / 4 : 0;
    for (int v = get_global_id(0); v < planeVectors; v += stride)
        vstore4(toPixels4(vload4(v, input) + vload_half4(v, u) * strength), v, output);
    for (int i = planeVectors * 4 + get_global_id(0); i < planeElements; i += stride)
        output[i] = toPixel(input[i] + vload_half(i, u) * strength);
}

// Grayscale embedding transposing column-major input to row-major output via local memory
#define TR_TILE 16
#define TR_ROWS (WG_SIZE / TR_TILE)
__kernel __attribute__((reqd_work_group_size(TR_TILE, TR_ROWS, 1))) void apply_watermark_row_major(__global const float* restrict input, __global const half* restrict u,
    __global const ulong* restrict sumSqPtr, __global uchar* restrict output, const float strengthNumerator, const int width, const int height, const int hasMaxAbs)
{
    __local uchar tile[TR_TILE][TR_TILE + 4];
    const float strength = embedStrength(sumSqPtr, hasMaxAbs, strengthNumerator);
    const int row = get_group_id(0) * TR_TILE + get_local_id(0);
    for (int i = get_local_id(1); i < TR_TILE; i += TR_ROWS) {
        const int col = get_group_id(1) * TR_TILE + i;
        if (row < height && col < width) {
            const int idx = col * height + row;
            tile[i][get_local_id(0)] = toPixel(input[idx] + vload_half(idx, u) * strength);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int outCol = get_group_id(1) * TR_TILE + get_local_id(0);
    for (int i = get_local_id(1); i < TR_TILE; i += TR_ROWS) {
        const int outRow = get_group_id(0) * TR_TILE + i;
        if (outRow < height && outCol < width)
            output[outRow * width + outCol] = tile[get_local_id(0)][i];
    }
}

)CLC"
                                   R"CLC(

// Autocorrelation shift layout for fast prediction system assembly
// Sums unique spatial shifts over the inner area, then adds border lines and corners
#define MAX_SHIFT             (WINDOW_SIZE - 1)
#define SHIFT_SPAN            ((2 * MAX_SHIFT) + 1)
// Keep only unique shift directions (dc > 0 or dc == 0 with dr >= 0)
#define NUM_SHIFTS            (((SHIFT_SPAN * SHIFT_SPAN) + 1) / 2)
// Border rows and columns outside the inner image area
#define BORDER_SIZE           (4 * PAD)
// Layout of fixed point sums for inner area, border rows, and border columns
#define BORDER_ROWS_OFFSET     NUM_SHIFTS
#define BORDER_COLS_OFFSET     (NUM_SHIFTS + (NUM_SHIFTS * BORDER_SIZE))
#define SHIFT_SUMS_SIZE    (NUM_SHIFTS * (1 + (2 * BORDER_SIZE)))
// Scale products by 1 / 255^2 so fixed point sums do not overflow
#define PRODUCT_SCALE       (1.0f / (255.0f * 255.0f))
#define SHIFT_INDEX(dr, dc)   ((dc) == 0 ? (dr) : (MAX_SHIFT + 1) + (((dc) - 1) * SHIFT_SPAN) + ((dr) + MAX_SHIFT))
// First shift index for column offset dc
#define FIRST_SHIFT(dc)       ((dc) == 0 ? 0 : SHIFT_INDEX(-MAX_SHIFT, dc))
#define SHIFT_END(dc)         ((dc) > MAX_SHIFT ? NUM_SHIFTS : FIRST_SHIFT(dc))
// Number of column shift groups to keep register usage in check
#define SHIFT_GROUPS     (WINDOW_SIZE >= 9 ? 3 : (WINDOW_SIZE >= 7 ? 2 : 1))
// Number of neighboring inner columns processed per task
#define INTERIOR_COLS       (WINDOW_SIZE >= 7 ? 2 : 1)
// Number of rows processed per task
#define INTERIOR_RUN        8
#define HALO                8
#define GROUP_FIRST_DC(g)   (((g) * (MAX_SHIFT + 1)) / SHIFT_GROUPS)
#define GROUP_SHIFTS(g)       (SHIFT_END(GROUP_FIRST_DC((g) + 1)) - FIRST_SHIFT(GROUP_FIRST_DC(g)))
// Maximum number of shifts assigned to any single group
#define MAX_OF(a, b)      ((a) > (b) ? (a) : (b))
#define GROUP_SHIFTS_OR0(g)   ((g) < SHIFT_GROUPS ? GROUP_SHIFTS(g) : 0)
#define MAX_GROUP_SHIFTS      MAX_OF(MAX_OF(MAX_OF(GROUP_SHIFTS_OR0(0), GROUP_SHIFTS_OR0(1)), MAX_OF(GROUP_SHIFTS_OR0(2), GROUP_SHIFTS_OR0(3))), \
                                MAX_OF(MAX_OF(GROUP_SHIFTS_OR0(4), GROUP_SHIFTS_OR0(5)), MAX_OF(GROUP_SHIFTS_OR0(6), MAX_OF(GROUP_SHIFTS_OR0(7), GROUP_SHIFTS_OR0(8)))))
// Copy border rows to row-major format for p >= 7 to speed up memory access
#define COPY_BORDER_ROWS      (WINDOW_SIZE >= 7)
#define BORDER_COPY_ROWS          (6 * PAD)
// Number of shifts reduced per pass
#define RED_CHUNK           16
#define RED_SLICES          (WG_SIZE / RED_CHUNK)

// Maps a border index to the unclamped row or column coordinate
WM_INLINE int borderToCoord(const int border, const int size) { return border < 2 * PAD ? border - PAD : size - (3 * PAD) + border; }
WM_INLINE int borderCopyRow(const int row, const int height) { return row < BORDER_COPY_ROWS / 2 ? row : row - height + BORDER_COPY_ROWS; }

// Reads an image pixel with edge clamping
WM_INLINE float clampedPixel(const __global float* restrict input, const int r, const int c, const int width, const int height) {
    return input[((size_t)clamp(c, 0, width - 1) * height) + clamp(r, 0, height - 1)];
}

// Reduces work-item sums in local memory and updates global output with atomic adds
WM_INLINE void reduceShiftSums(const float* restrict sums, const int num, __global ulong* restrict output, const int stride, __local float* restrict scratch) {
    const int lid = get_local_id(0);
    __local float* partial = scratch + (RED_CHUNK * WG_SIZE);
    #pragma unroll
    for (int base = 0; base < MAX_GROUP_SHIFTS; base += RED_CHUNK) {
        if (base < num) {
            #pragma unroll
            for (int i = 0; i < RED_CHUNK; i++)
                scratch[(i * WG_SIZE) + lid] = base + i < num ? sums[base + i] : 0.0f;
            barrier(CLK_LOCAL_MEM_FENCE);
            const int shift = lid / RED_SLICES;
            const int slice = lid % RED_SLICES;
            float value = 0.0f;
            #pragma unroll
            for (int s = 0; s < WG_SIZE / RED_SLICES; s++)
                value += scratch[(shift * WG_SIZE) + (s * RED_SLICES) + slice];
            partial[(shift * RED_SLICES) + slice] = value;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < RED_CHUNK && base + lid < num) {
                float total = 0.0f;
                for (int s = 0; s < RED_SLICES; s++)
                    total += partial[(lid * RED_SLICES) + s];
                atom_add(output + ((base + lid) * stride), toScaledUlong(total));
            }
        }
    }
}

// Accumulates shift sums for INTERIOR_RUN rows across neighboring columns
WM_INLINE void sumColumnRun(const __global float* restrict input, const int c, const int r0, const int width, const int height, float* restrict sums, const int dc0, const int dc1,
    const int cols, const int validCols)
{
    const bool alignedColumns = (height & 3) == 0;
    const bool interiorRun = alignedColumns && r0 >= PAD && r0 + INTERIOR_RUN <= height - PAD;
    // Scale task values while zeroing pixels outside inner image rows
    float a[2 * INTERIOR_RUN];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        if (j >= cols)
            continue;
        const __global float* colA = input + ((size_t)clamp(c + j, 0, width - 1) * height);
        const float scale = j < validCols ? PRODUCT_SCALE : 0.0f;
        if (interiorRun) {
            #pragma unroll
            for (int v = 0; v < INTERIOR_RUN; v += 4) {
                const float4 f = vload4(0, colA + r0 + v);
                a[(j * INTERIOR_RUN) + v + 0] = f.x * scale;
                a[(j * INTERIOR_RUN) + v + 1] = f.y * scale;
                a[(j * INTERIOR_RUN) + v + 2] = f.z * scale;
                a[(j * INTERIOR_RUN) + v + 3] = f.w * scale;
            }
        } else {
            #pragma unroll
            for (int t = 0; t < INTERIOR_RUN; t++) {
                const int r = r0 + t;
                a[(j * INTERIOR_RUN) + t] = (r >= PAD && r < height - PAD) ? colA[r] * scale : 0.0f;
            }
        }
    }

    const bool fastWindow = alignedColumns && r0 - HALO >= 0 && r0 + INTERIOR_RUN + HALO <= height;
    #pragma unroll
    for (int k = dc0; k < dc1 + cols - 1; k++) {
        const __global float* colB = input + ((size_t)clamp(c + k, 0, width - 1) * height);
        float w[INTERIOR_RUN + (2 * HALO)];
        if (fastWindow) {
            #pragma unroll
            for (int v = 0; v < INTERIOR_RUN + (2 * HALO); v += 4) {
                const float4 f = vload4(0, colB + r0 - HALO + v);
                w[v + 0] = f.x;
                w[v + 1] = f.y;
                w[v + 2] = f.z;
                w[v + 3] = f.w;
            }
        } else {
            #pragma unroll
            for (int v = 0; v < INTERIOR_RUN + (2 * HALO); v++)
                w[v] = colB[clamp(r0 - HALO + v, 0, height - 1)];
        }
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int dc = k - j;
            if (j >= cols || dc < dc0 || dc >= dc1)
                continue;
            #pragma unroll
            for (int dr = -MAX_SHIFT; dr <= MAX_SHIFT; dr++) {
                if (dc == 0 && dr < 0)
                    continue; // Skip opposite shift already covered
                float sum = 0.0f;
                #pragma unroll
                for (int t = 0; t < INTERIOR_RUN; t++)
                    sum = fma(a[(j * INTERIOR_RUN) + t], w[t + HALO + dr], sum);
                sums[SHIFT_INDEX(dr, dc) - FIRST_SHIFT(dc0)] += sum;
            }
        }
    }
}

)CLC"
                                   R"CLC(

// Returns pointer to image row in either row-major copy or column-major source
WM_INLINE const __global float* borderRowPointer(const __global float* restrict source, const int imageRow, const int width, const int height) {
    const int clamped = clamp(imageRow, 0, height - 1);
#if COPY_BORDER_ROWS
    return source + ((size_t)borderCopyRow(clamped, height) * width);
#else
    return source + clamped;
#endif
}

// Accumulates shift sums along a border row for columns in range
WM_INLINE void sumRowRun(const __global float* restrict source, const int r, const int c0, const int width, const int height, float* restrict sums, const int dc0, const int dc1) {
    const size_t columnStride = COPY_BORDER_ROWS ? 1 : height;
    const __global float* rowA = borderRowPointer(source, r, width, height);
    float a[INTERIOR_RUN];
    #pragma unroll
    for (int t = 0; t < INTERIOR_RUN; t++) {
        const int c = c0 + t;
        a[t] = (c >= PAD && c < width - PAD) ? rowA[c * columnStride] * PRODUCT_SCALE : 0.0f;
    }
    #pragma unroll
    for (int dr = -MAX_SHIFT; dr <= MAX_SHIFT; dr++) {
        const __global float* rowB = borderRowPointer(source, r + dr, width, height);
        float w[INTERIOR_RUN + MAX_SHIFT];
        #pragma unroll
        for (int v = 0; v < INTERIOR_RUN + (dc1 - 1 - dc0); v++)
            w[v] = rowB[clamp(c0 + dc0 + v, 0, width - 1) * columnStride];
        #pragma unroll
        for (int dc = dc0; dc < dc1; dc++) {
            if (dc == 0 && dr < 0)
                continue; // Skip opposite shift already covered
            float sum = 0.0f;
            #pragma unroll
            for (int t = 0; t < INTERIOR_RUN; t++)
                sum = fma(a[t], w[t + dc - dc0], sum);
            sums[SHIFT_INDEX(dr, dc) - FIRST_SHIFT(dc0)] += sum;
        }
    }
}

// Copies top and bottom border rows into a contiguous row-major buffer
__kernel void me_copy_border_rows(__global const float* restrict input, __global float* restrict borderCopy, const int width, const int height) {
    const int total = BORDER_COPY_ROWS * width;
    for (int i = get_global_id(0); i < total; i += get_global_size(0)) {
        const int row = i % BORDER_COPY_ROWS;
        const int col = i / BORDER_COPY_ROWS;
        const int imageRow = row < BORDER_COPY_ROWS / 2 ? row : height - BORDER_COPY_ROWS + row;
        borderCopy[((size_t)row * width) + col] = input[((size_t)col * height) + imageRow];
    }
}

// Dispatches inner tasks or border lines for one column shift group
WM_INLINE void shiftSumsGroup(__global const float* restrict input, __global const float* restrict borderRowsCopy, __global ulong* restrict shiftSums, const int width, const int height,
    const bool isBorder, const int groupBlock, const int interiorBlocks, const int borderBlocksPerLine, __local float* restrict scratch, const int dc0, const int dc1)
{
    const int shift0 = FIRST_SHIFT(dc0);
    const int num = SHIFT_END(dc1) - shift0;
    const int lid = get_local_id(0);
    float sums[MAX_GROUP_SHIFTS];
    #pragma unroll
    for (int i = 0; i < MAX_GROUP_SHIFTS; i++)
        sums[i] = 0.0f;
    if (!isBorder) {
        const int runsPerColumn = (height + INTERIOR_RUN - 1) / INTERIOR_RUN;
        const int interiorColumns = width - (2 * PAD);
        const int totalTasks = runsPerColumn * ((interiorColumns + INTERIOR_COLS - 1) / INTERIOR_COLS);
        const int stride = (interiorBlocks / SHIFT_GROUPS) * WG_SIZE;
        for (int task = (groupBlock * WG_SIZE) + lid; task < totalTasks; task += stride) {
            const int c = INTERIOR_COLS * (task / runsPerColumn);
            sumColumnRun(input, PAD + c, (task % runsPerColumn) * INTERIOR_RUN, width, height, sums, dc0, dc1, INTERIOR_COLS, min(INTERIOR_COLS, interiorColumns - c));
        }
        reduceShiftSums(sums, num, shiftSums + shift0, 1, scratch);
    } else {
        const int line = groupBlock / borderBlocksPerLine;
        const int lineBlock = groupBlock % borderBlocksPerLine;
        const bool isBorderRow = line < BORDER_SIZE;
        const int border = isBorderRow ? line : line - BORDER_SIZE;
        const int fixedCoord = borderToCoord(border, isBorderRow ? height : width);
        const int runs = ((isBorderRow ? width : height) + INTERIOR_RUN - 1) / INTERIOR_RUN;
        for (int run = (lineBlock * WG_SIZE) + lid; run < runs; run += borderBlocksPerLine * WG_SIZE) {
            if (isBorderRow)
                sumRowRun(COPY_BORDER_ROWS ? borderRowsCopy : input, fixedCoord, run * INTERIOR_RUN, width, height, sums, dc0, dc1);
            else
                sumColumnRun(input, fixedCoord, run * INTERIOR_RUN, width, height, sums, dc0, dc1, 1, 1);
        }
        __global ulong* output = shiftSums + (isBorderRow ? BORDER_ROWS_OFFSET : BORDER_COLS_OFFSET) + (shift0 * BORDER_SIZE) + border;
        reduceShiftSums(sums, num, output, BORDER_SIZE, scratch);
    }
}

// Single-launch kernel computing all inner and border shift sums
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1))) void me_shift_sums(__global const float* restrict input, __global const float* restrict borderRowsCopy,
    __global ulong* restrict shiftSums, const int width, const int height, const int interiorBlocks, const int borderBlocksPerLine)
{
    __local float scratch[(RED_CHUNK * WG_SIZE) + (RED_CHUNK * RED_SLICES)];
    const int blockId = get_group_id(0);
    const bool isBorder = blockId >= interiorBlocks;
    const int localBlock = isBorder ? blockId - interiorBlocks : blockId;
    const int group = localBlock % SHIFT_GROUPS;
    const int groupBlock = localBlock / SHIFT_GROUPS;
    // Branch per column shift group so ranges are compile time constants
#define SHIFT_GROUP(G)                                                                                                                             \
    if (group == (G)) {                                                                                                                          \
        shiftSumsGroup(input, borderRowsCopy, shiftSums, width, height, isBorder, groupBlock, interiorBlocks, borderBlocksPerLine, scratch, GROUP_FIRST_DC(G), \
            GROUP_FIRST_DC((G) + 1));                                                                                                            \
        return;                                                                                                                                  \
    }
    SHIFT_GROUP(0)
#if SHIFT_GROUPS > 1
    SHIFT_GROUP(1)
#endif
#if SHIFT_GROUPS > 2
    SHIFT_GROUP(2)
#endif
#if SHIFT_GROUPS > 3
    SHIFT_GROUP(3)
#endif
#if SHIFT_GROUPS > 4
    SHIFT_GROUP(4)
#endif
#if SHIFT_GROUPS > 5
    SHIFT_GROUP(5)
#endif
#if SHIFT_GROUPS > 6
    SHIFT_GROUP(6)
#endif
#if SHIFT_GROUPS > 7
    SHIFT_GROUP(7)
#endif
#if SHIFT_GROUPS > 8
    SHIFT_GROUP(8)
#endif
#undef SHIFT_GROUP
}

)CLC"
                                   R"CLC(

// Converts a packed lower triangular index to row and column coordinates
WM_INLINE int2 packedToRowCol(const int k) {
    int r = (int)(0.5f * (sqrt(1.0f + 8.0f * k) - 1.0f));
    if ((r * (r + 1)) / 2 > k)
        r--;
    if (((r + 1) * (r + 2)) / 2 <= k)
        r++;
    return (int2)(r, k - (r * (r + 1)) / 2);
}

// Converts solver variable index to row-major coefficient index
WM_INLINE int coefficientIndex(const int k) {
    const int kPixel = k + (k >= WINDOW_CENTER);
    const int r = kPixel / WINDOW_SIZE;
    const int originalPixel = (kPixel * WINDOW_SIZE) - (r * NEIGHB_SIZE);
    return originalPixel - (originalPixel > WINDOW_CENTER);
}

// Builds one entry of the solver system from inner shift sums, borders, and corners
WM_INLINE ulong buildSystemEntry(__global const float* restrict input, __global const ulong* restrict shiftSums, const int entry, const int width, const int height) {
    const int rxOffset = (NEIGHB_SIZE * (NEIGHB_SIZE + 1)) / 2;
    int wa, wb;
    if (entry < rxOffset) {
        const int2 coords = packedToRowCol(entry);
        const int ea = coefficientIndex(coords.x);
        const int eb = coefficientIndex(coords.y);
        wa = ea + (ea >= WINDOW_CENTER);
        wb = eb + (eb >= WINDOW_CENTER);
    } else {
        const int ea = coefficientIndex(entry - rxOffset);
        wa = ea + (ea >= WINDOW_CENTER);
        wb = WINDOW_CENTER;
    }
    // Map window coordinates to shift offset, swapping direction if needed
    int baseRow = (wa % WINDOW_SIZE) - PAD, baseCol = (wa / WINDOW_SIZE) - PAD;
    int dr = (wb % WINDOW_SIZE) - (wa % WINDOW_SIZE), dc = (wb / WINDOW_SIZE) - (wa / WINDOW_SIZE);
    if (dc < 0 || (dc == 0 && dr < 0)) {
        baseRow += dr;
        baseCol += dc;
        dr = -dr;
        dc = -dc;
    }
    const int shift = SHIFT_INDEX(dr, dc);
    // Accumulate border sums across the contiguous window range
    const int rowLo = baseRow + PAD;
    const int colLo = baseCol + PAD;
    ulong fixedSum = shiftSums[shift];
    #pragma unroll
    for (int i = 0; i < 2 * PAD; i++)
        fixedSum += shiftSums[BORDER_ROWS_OFFSET + (shift * BORDER_SIZE) + rowLo + i] + shiftSums[BORDER_COLS_OFFSET + (shift * BORDER_SIZE) + colLo + i];
    float corners = 0.0f;
    #pragma unroll
    for (int i = 0; i < 2 * PAD; i++) {
        const int r = borderToCoord(rowLo + i, height);
        #pragma unroll
        for (int j = 0; j < 2 * PAD; j++) {
            const int c = borderToCoord(colLo + j, width);
            corners += clampedPixel(input, r, c, width, height) * clampedPixel(input, r + dr, c + dc, width, height);
        }
    }
    return fixedSum + toScaledUlong(corners * PRODUCT_SCALE);
}

// Assembles the solver matrix entries in parallel across work-items
__kernel void me_build_system(__global const float* restrict input, __global const ulong* restrict shiftSums, __global ulong* restrict system, const int width, const int height) {
    const int entry = get_global_id(0);
    if (entry < ((NEIGHB_SIZE * (NEIGHB_SIZE + 1)) / 2) + NEIGHB_SIZE)
        system[entry] = buildSystemEntry(input, shiftSums, entry, width, height);
}

)CLC"
                                   R"CLC(

// Emulated double precision using float2 (x = hi, y = lo) for roughly 48 bits of precision
#pragma OPENCL FP_CONTRACT OFF

// Exact sum of two floats using Dekker addition
WM_INLINE float2 ffTwoSum(const float a, const float b) {
    const float s = a + b;
    const float v = s - a;
    return (float2)(s, (a - (s - v)) + (b - v));
}
// Fast exact sum when magnitude of a is greater than or equal to b
WM_INLINE float2 ffQuickTwoSum(const float a, const float b) {
    const float s = a + b;
    return (float2)(s, b - (s - a));
}
// Exact product of two floats using FMA
WM_INLINE float2 ffTwoProd(const float a, const float b) {
    const float p = a * b;
    return (float2)(p, fma(a, b, -p));
}
// Exact conversion from ulong to float-float
WM_INLINE float2 ffFromUlong(const ulong value) {
    const float h = convert_float_rte(value);
    return ffQuickTwoSum(h, (float)((long)value - convert_long(h)));
}
WM_INLINE float ffToFloat(const float2 a) { return a.x + a.y; }
WM_INLINE float2 ffMul(const float2 a, const float2 b) {
    float2 p = ffTwoProd(a.x, b.x);
    p.y += (a.x * b.y) + (a.y * b.x);
    return ffQuickTwoSum(p.x, p.y);
}
// Computes c - a * b with fast error compensation for Cholesky updates
WM_INLINE float2 ffFnma(const float2 a, const float2 b, const float2 c) {
    const float ph = a.x * b.x;
    const float pl = fma(a.x, b.y, fma(a.y, b.x, fma(a.x, b.x, -ph)));
    const float2 s = ffTwoSum(c.x, -ph);
    return ffQuickTwoSum(s.x, s.y + (c.y - pl));
}
// Fast reciprocal square root refined with one float-float Newton step
WM_INLINE float2 ffRsqrt(const float2 v) {
    const float h = rsqrt(v.x);
    const float2 vh2 = ffMul(v, ffTwoProd(h, h));
    const float r = (1.0f - vh2.x) - vh2.y;
    return ffQuickTwoSum(h, 0.5f * h * r);
}

// Blocked Cholesky solver in float-float arithmetic using panel updates
// Panels factor 8 columns at a time, followed by a trailing submatrix update
#define NB                  8
#define PACKED_SIZE         ((NEIGHB_SIZE * (NEIGHB_SIZE + 1)) / 2)
#define SYSTEM_SIZE         (PACKED_SIZE + NEIGHB_SIZE)
// Dense panel buffer for trailing submatrix update
#define PANEL_STRIDE        ((2 * NB) + 4)
#define PANEL_SIZE          ((NEIGHB_SIZE - NB) * PANEL_STRIDE)
#define ROWS_PER_ITEM       ((NEIGHB_SIZE + WG_SIZE - 1) / WG_SIZE)

WM_INLINE int packedIndex(const int r, const int c) { return ((r * (r + 1)) / 2) + c; }

// Updates trailing submatrix A22 using outer product of panel L21
WM_INLINE void choleskyTrailingUpdate(__local float2* restrict sA, __local const float* restrict sPanel, const int k0) {
    const int first = k0 + NB;
    const int rows = NEIGHB_SIZE - first;
    const int trailing = (rows * (rows + 1)) / 2;
    for (int e = get_local_id(0); e < trailing; e += WG_SIZE) {
        const int2 coords = packedToRowCol(e);
        __local const float4* rowI = (__local const float4*)(sPanel + (coords.x * PANEL_STRIDE));
        __local const float4* rowJ = (__local const float4*)(sPanel + (coords.y * PANEL_STRIDE));
        float pi[2 * NB], pj[2 * NB];
        #pragma unroll
        for (int v = 0; v < (2 * NB) / 4; v++) {
            vstore4(rowI[v], v, pi);
            vstore4(rowJ[v], v, pj);
        }
        const int index = packedIndex(first + coords.x, first + coords.y);
        // Accumulate low parts first and normalize once at the end
        const float2 acc = sA[index];
        float hi = acc.x, lo = acc.y;
        #pragma unroll
        for (int m = 0; m < NB; m++) {
            const float ph = pi[m] * pj[m];
            const float pl = fma(pi[m], pj[NB + m], fma(pi[NB + m], pj[m], fma(pi[m], pj[m], -ph)));
            const float2 sum = ffTwoSum(hi, -ph);
            hi = sum.x;
            lo += sum.y - pl;
        }
        sA[index] = ffQuickTwoSum(hi, lo);
    }
}

// Solves Rx * x = rx using blocked Cholesky decomposition in local memory
__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1))) void me_solve_system(__global const ulong* restrict system, __global ulong* restrict shiftSums,
    __global ulong* restrict embedSums, __global float* restrict X, __global int* restrict stopFlag)
{
    __local float2 sSystem[SYSTEM_SIZE];
    __local float2* sA = sSystem;
    __local float2* sB = sSystem + PACKED_SIZE;
    __local float2 sInv[NEIGHB_SIZE];
    __local float4 sPanelStorage[PANEL_SIZE > 0 ? PANEL_SIZE / 4 : 1];
    __local float* sPanel = (__local float*)sPanelStorage;
    __local float2 sL[NB * NB];
    __local float2 sPivot[NB];
    __local float2 sY[NB];
    __local float2 sX[NEIGHB_SIZE];
    const int lid = get_local_id(0);
    for (int i = lid; i < SYSTEM_SIZE; i += WG_SIZE)
        sSystem[i] = ffFromUlong(system[i]);
    for (int i = lid; i < SHIFT_SUMS_SIZE; i += WG_SIZE)
        shiftSums[i] = 0;
    if (lid < 2)
        embedSums[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k0 = 0; k0 < NEIGHB_SIZE; k0 += NB) {
        float2 a[ROWS_PER_ITEM][NB];
        float2 b[ROWS_PER_ITEM];
        #pragma unroll
        for (int m = 0; m < ROWS_PER_ITEM; m++) {
            const int row = k0 + lid + (m * WG_SIZE);
            const bool valid = row < NEIGHB_SIZE;
            b[m] = valid ? sB[row] : (float2)(0.0f);
            #pragma unroll
            for (int j = 0; j < NB; j++)
                a[m][j] = valid && (row >= k0 + NB || j <= row - k0) ? sA[packedIndex(row, k0 + j)] : (float2)(0.0f);
        }
        #pragma unroll
        for (int k = 0; k < NB; k++) {
            const float2 pivot = k == 0 ? sA[packedIndex(k0, k0)] : sPivot[k];
            if (!(pivot.x > 1e-3f)) { // Exit early if matrix is not positive definite
                if (lid == 0)
                    *stopFlag = 1;
                return;
            }
            const float2 inv = ffRsqrt(pivot);
            float2 l[ROWS_PER_ITEM];
            #pragma unroll
            for (int m = 0; m < ROWS_PER_ITEM; m++) {
                const int row = k0 + lid + (m * WG_SIZE);
                // Scale panel entries by inverse pivot square root
                l[m] = ffMul(a[m][k], inv);
                if (row == k0 + k) {
                    const float2 y = ffMul(b[m], inv);
                    sY[k] = y;
                    b[m] = y;
                }
                if (row > k0 + k && row < k0 + NB)
                    sL[(k * NB) + (row - k0)] = l[m];
                if (k + 1 < NB && row == k0 + k + 1)
                    sPivot[k + 1] = ffFnma(l[m], l[m], a[m][k + 1]);
                a[m][k] = l[m];
            }
            if (lid == 0)
                sInv[k0 + k] = inv;
            barrier(CLK_LOCAL_MEM_FENCE);
            const float2 yk = sY[k];
            #pragma unroll
            for (int m = 0; m < ROWS_PER_ITEM; m++) {
                const int row = k0 + lid + (m * WG_SIZE);
                if (row > k0 + k && row < NEIGHB_SIZE) {
                    #pragma unroll
                    for (int j = k + 1; j < NB; j++)
                        a[m][j] = ffFnma(l[m], sL[(k * NB) + j], a[m][j]);
                    b[m] = ffFnma(l[m], yk, b[m]);
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < ROWS_PER_ITEM; m++) {
            const int row = k0 + lid + (m * WG_SIZE);
            if (row < NEIGHB_SIZE) {
                #pragma unroll
                for (int j = 0; j < NB; j++)
                    if (row >= k0 + NB || j <= row - k0)
                        sA[packedIndex(row, k0 + j)] = a[m][j];
                sB[row] = b[m];
                if (row >= k0 + NB) {
                    __local float* panelRow = sPanel + ((row - k0 - NB) * PANEL_STRIDE);
                    #pragma unroll
                    for (int j = 0; j < NB; j++) {
                        panelRow[j] = a[m][j].x;
                        panelRow[NB + j] = a[m][j].y;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k0 + NB < NEIGHB_SIZE) {
            choleskyTrailingUpdate(sA, sPanel, k0);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Backward substitution L^T * x = y across work-items
    float2 y[ROWS_PER_ITEM];
    #pragma unroll
    for (int m = 0; m < ROWS_PER_ITEM; m++) {
        const int i = lid + (m * WG_SIZE);
        y[m] = i < NEIGHB_SIZE ? sB[i] : (float2)(0.0f);
        if (i == NEIGHB_SIZE - 1)
            sX[i] = ffMul(y[m], sInv[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = NEIGHB_SIZE - 1; k > 0; k--) {
        const float2 xk = sX[k];
        #pragma unroll
        for (int m = 0; m < ROWS_PER_ITEM; m++) {
            const int i = lid + (m * WG_SIZE);
            if (i < k) {
                y[m] = ffFnma(sA[packedIndex(k, i)], xk, y[m]);
                if (i == k - 1)
                    sX[i] = ffMul(y[m], sInv[i]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = lid; i < NEIGHB_SIZE; i += WG_SIZE)
        X[coefficientIndex(i)] = ffToFloat(sX[i]);
    if (lid == 0)
        *stopFlag = 0;
}

)CLC";
