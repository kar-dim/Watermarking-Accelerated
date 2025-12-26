#pragma once
#include "opencl_init.h"
#include <stdexcept>
#include <string>

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
    cl::Program buildKernels(const int p);

	//wrap a cl_mem pointer into a cl::Buffer
    inline cl::Buffer wrap(const cl_mem* mem) { return cl::Buffer(*mem, true); }

	//helper method to execute an OpenCL kernel and throw detailed error on failure
    template<typename Func>
    void executeKernel(const Func& kernelFunc, const std::string& context)
    {
        try {
            kernelFunc();
        }
        catch (const cl::Error& ex) {
            throw std::runtime_error("OpenCL Error in " + context + ": " + std::string(ex.what()) + " Error code: " + std::to_string(ex.err()) + "\n");
        }
    }
}