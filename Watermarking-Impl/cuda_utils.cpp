#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <utility>

namespace cuda_utils 
{
    //Helper method to calculate kernel grid size from given 2D dimensions and blockSize
    dim3 gridSizeCalculate(const dim3 blockSize, const int rows, const int cols, const bool rowsFirst)
    {
        return rowsFirst ? 
            dim3((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y) :
            dim3((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    }

    //Simple wrapper of cudaMallocArray to reduce boilerplate
    cudaArray* cudaMallocArray(const std::size_t cols, const std::size_t rows)
    {
        cudaArray* cuArray;
        auto cudaChannelDescriptor = cudaCreateChannelDesc<float>();
        cudaMallocArray(&cuArray, &cudaChannelDescriptor, cols, rows);
        return cuArray;
    }

    //Creates a cudaResourceDesc from a specified cudaArray
    cudaResourceDesc createResourceDescriptor(cudaArray* cuArray)
    {
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        return resDesc;
    }

    //creates a cudaTextureDesc with these properties:
    //Border addressing mode, point filtering mode, element read mode, non-normalized coords
    cudaTextureDesc createTextureDescriptor()
    {
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        return texDesc;
    }

    //creates a texture object from the given cuda resource and cuda texture descriptors
    cudaTextureObject_t createTextureObject(const cudaResourceDesc& pResDesc, const cudaTextureDesc& pTexDesc)
    {
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &pResDesc, &pTexDesc, NULL);
        return texObj;
    }

    //get a cudaDeviceProp handle to query for various device information
    cudaDeviceProp getDeviceProperties()
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        return properties;
    }

    //create the cudaArray and the textureObject binded to this array.
    std::pair<cudaTextureObject_t, cudaArray*> createTextureData(const unsigned int rows, const unsigned int cols)
    {
        cudaArray* cuArray = cuda_utils::cudaMallocArray(cols, rows);
        cudaResourceDesc resDesc = cuda_utils::createResourceDescriptor(cuArray);
        cudaTextureDesc texDesc = cuda_utils::createTextureDescriptor();
        cudaTextureObject_t texObj = cuda_utils::createTextureObject(resDesc, texDesc);
        return std::make_pair(texObj, cuArray);
    }

    //copy Device data to Device Array
    void copyDataToCudaArray(const float* data, const unsigned int rows, const unsigned int cols, cudaArray* cuArray) 
    {
        cudaMemcpy2DToArray(cuArray, 0, 0, data, cols * sizeof(float), cols * sizeof(float), rows, cudaMemcpyDeviceToDevice);
    }
}