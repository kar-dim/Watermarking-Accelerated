#pragma once
#include <string>
#include <WatermarkBase.hpp>

struct MaskDiskConfig {
    MASK_TYPE strategy;
    std::string label;
    std::string outputFile;
};