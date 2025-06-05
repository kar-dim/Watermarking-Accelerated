#include "opencl_utils.hpp"
#include "CL/opencl.hpp"

namespace cl_utils 
{
    KernelBuilder::KernelBuilder(const cl::Program& program, const char* name)
    {
        kernel = cl::Kernel(program, name);
        argsCounter = 0;
    }

    cl::Kernel KernelBuilder::build() const 
    {
        return kernel;
    }

    void copyBufferToImage(const cl::CommandQueue& queue, const cl::Image2D& image2d, const cl_mem* imageBuff, const long long rows, const long long cols)
    {
        const size_t orig[] = { 0,0,0 };
        const size_t des[] = { static_cast<size_t>(cols), static_cast<size_t>(rows), 1 };
        clEnqueueCopyBufferToImage(queue(), *imageBuff, image2d(), 0, orig, des, NULL, NULL, NULL);
    }
}