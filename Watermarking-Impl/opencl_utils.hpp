#pragma once
#include "opencl_init.h"

namespace cl_utils 
{
    class KernelBuilder 
    {
    private:
        cl::Kernel kernel;
        int argsCounter;
    public:
        KernelBuilder(const cl::Program& program, const char* name);

        /*! \brief setArg overload taking a POD type */
        template <typename... T>
        KernelBuilder& args(const T&... values)
        {
            (kernel.setArg<T>(argsCounter++, values), ...);
            return *this;
        }

        /*! \brief build the cl::Kernel object */
        cl::Kernel build() const;
    };

    //helper method to copy an OpenCL buffer into an OpenCL Image (fast copy that happens in the device)
    void copyBufferToImage(const cl::CommandQueue& queue, const cl::Image2D& image2d, const cl_mem* imageBuff, const long long rows, const  long long cols);
}