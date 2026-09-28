#pragma once
// the host code needs at least AVX2. The AVX-512 code paths (#if defined(__AVX512F__)) are used when the
// build enables AVX-512 (msbuild -p:WatermarkingSimd=AVX512), an AVX-512 build also defines __AVX2__
#if !defined(__AVX2__)
#error "AVX2 is required: build with AVX2 (default) or AVX-512 enabled"
#endif
#include <immintrin.h>
