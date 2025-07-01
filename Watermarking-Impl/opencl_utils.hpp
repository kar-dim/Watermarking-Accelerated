#pragma once
#include "opencl_init.h"
#include <vector>

/*!
 *  \brief  Helper utility functions related to OpenCL.
 *  \author Dimitris Karatzas
 */
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

    //helper method to build opencl kernels from source
    bool buildKernels(std::vector<cl::Program>& programs, const int p);
}