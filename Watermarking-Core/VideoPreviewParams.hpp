#pragma once

/*!
 * \brief Shared CUDA and OpenCL video preview parameters
 * \author Dimitris Karatzas
 */

// keep the field order identical to Params in the OpenCL preview kernel
struct VideoPreviewParams {
    int width, height, outWidth, outHeight;
    int yPitch, uPitch, vPitch;
    int shiftX, shiftY, interleaved, gray;
    float yOffset, yScale, uvScale, redV, greenU, greenV, blueU;
};
static_assert(sizeof(VideoPreviewParams) == 72);
