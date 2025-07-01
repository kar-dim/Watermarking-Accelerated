#include "kernels/me_p3.hpp"
#include "kernels/nvf.hpp"
#include "kernels/scaled_neighbors_p3.hpp"
#include "opencl_utils.hpp"
#include <af/opencl.h>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::string;

namespace cl_utils 
{
    KernelBuilder::KernelBuilder(const cl::Program& program, const char* name) 
		: kernel(program, name), argsCounter(0)
    { }

    cl::Kernel KernelBuilder::build() const { return kernel; }

    bool buildKernels(std::vector<cl::Program>& programs, const int p)
    {
		cl::Context context(afcl::getContext(false));
		cl::Device device(afcl::getDeviceId(), false);
		//compile opencl kernels
		try {
			auto buildProgram = [&context, &device](auto& program, const string& kernelName, const string& buildOptions)
				{
					program = cl::Program(context, kernelName);
					program.build(device, buildOptions.c_str());
				};
			buildProgram(programs[0], nvf, "-cl-mad-enable -Dp=" + std::to_string(p));
			buildProgram(programs[1], me_p3, "-cl-mad-enable");
			buildProgram(programs[2], scaled_neighbors_p3, "-cl-mad-enable");
		}
		catch (const cl::Error& e) {
			cout << "Could not build a kernel, Reason: " << e.what() << "\n\n";
			for (const cl::Program& program : programs)
			{
				if (program.get() != NULL && program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_BUILD_SUCCESS)
					cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
			}
			return false;
		}
		catch (const std::exception& ex) {
			cout << ex.what() << "\n";
			return false;
		}
		return true;
    }
}