#pragma once
// Types and enums used across the watermarking implementation
// helps to avoid circular dependencies and keeps the code organized
enum class MaskMethod { NVF, ME };
enum class VideoMode { EMBED, DETECT };