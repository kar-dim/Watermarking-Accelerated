#pragma once

#include "AvUtil.hpp"
#include "opencl_init.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <Windows.h>

extern "C" {
#include "libavutil/dict.h"
}

namespace video_utils {
// physical identifiers, not enumeration indices, optional fields accommodate older drivers
struct VideoDeviceIdentity {
    std::optional<std::array<uint8_t, 16>> uuid;
    std::optional<std::array<uint8_t, 8>> luid;
    std::optional<uint32_t> nodeMask;
    std::optional<uint32_t> domain, bus, slot, function;
};

inline bool sameVideoDevice(const VideoDeviceIdentity& selected, const VideoDeviceIdentity& candidate) {
    if (selected.uuid && candidate.uuid)
        return selected.uuid == candidate.uuid;
    if (selected.luid && candidate.luid)
        return selected.luid == candidate.luid && (!selected.nodeMask || !candidate.nodeMask || (*selected.nodeMask & *candidate.nodeMask) != 0);
    if (!selected.bus || !selected.slot || !candidate.bus || !candidate.slot || selected.bus != candidate.bus || selected.slot != candidate.slot)
        return false;
    return (!selected.domain || !candidate.domain || selected.domain == candidate.domain) && (!selected.function || !candidate.function || selected.function == candidate.function);
}

inline size_t matchingVideoDevice(const VideoDeviceIdentity& selected, const std::span<const VideoDeviceIdentity> candidates) {
    std::optional<size_t> found;
    for (size_t index = 0; index < candidates.size(); ++index)
        if (sameVideoDevice(selected, candidates[index])) {
            if (found)
                throw std::runtime_error("The selected OpenCL GPU matches multiple CUDA adapters; NVENC device selection is ambiguous");
            found = index;
        }
    if (!found)
        throw std::runtime_error("Cannot match the selected OpenCL GPU to a CUDA adapter for host-staged NVENC; use D3D11 sharing or CPU encoding");
    return *found;
}

template <typename T>
std::optional<T> videoDeviceInfo(const cl_device_id device, const cl_device_info property) {
    T value{};
    return clGetDeviceInfo(device, property, sizeof(value), &value, nullptr) == CL_SUCCESS ? std::optional<T>{value} : std::nullopt;
}

inline VideoDeviceIdentity openclVideoDeviceIdentity(const cl_device_id device) {
    VideoDeviceIdentity identity;
    identity.uuid = videoDeviceInfo<std::array<uint8_t, 16>>(device, CL_DEVICE_UUID_KHR);
    if (videoDeviceInfo<cl_bool>(device, CL_DEVICE_LUID_VALID_KHR).value_or(CL_FALSE)) {
        identity.luid = videoDeviceInfo<std::array<uint8_t, 8>>(device, CL_DEVICE_LUID_KHR);
        identity.nodeMask = videoDeviceInfo<cl_uint>(device, CL_DEVICE_NODE_MASK_KHR);
    }
    if (const auto pci = videoDeviceInfo<cl_device_pci_bus_info_khr>(device, CL_DEVICE_PCI_BUS_INFO_KHR)) {
        identity.domain = pci->pci_domain;
        identity.bus = pci->pci_bus;
        identity.slot = pci->pci_device;
        identity.function = pci->pci_function;
    } else {
        // cl_nv_device_attribute_query, legacy PCI identity queries predate KHR extension
        identity.bus = videoDeviceInfo<cl_uint>(device, 0x4008);
        identity.slot = videoDeviceInfo<cl_uint>(device, 0x4009);
        identity.domain = videoDeviceInfo<cl_uint>(device, 0x400A);
    }
    return identity;
}

// host-staged fallback for NVIDIA, Intel/AMD builds need neither CUDA libraries nor SDK headers
inline int nvidiaEncoderDevice(const cl_device_id selectedDevice) {
    const auto close = [](HMODULE module) {
        if (module)
            FreeLibrary(module);
    };
    std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(close)> library(LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32), close);
    if (!library)
        throw std::runtime_error("Cannot load the NVIDIA driver to identify the selected OpenCL GPU for NVENC");
    using Init = int(WINAPI*)(unsigned);
    using Count = int(WINAPI*)(int*);
    using Device = int(WINAPI*)(int*, int);
    using Uuid = int(WINAPI*)(void*, int);
    using Luid = int(WINAPI*)(char*, unsigned*, int);
    using Pci = int(WINAPI*)(char*, int, int);
    const auto init = reinterpret_cast<Init>(GetProcAddress(library.get(), "cuInit"));
    const auto count = reinterpret_cast<Count>(GetProcAddress(library.get(), "cuDeviceGetCount"));
    const auto device = reinterpret_cast<Device>(GetProcAddress(library.get(), "cuDeviceGet"));
    const auto uuid = reinterpret_cast<Uuid>(GetProcAddress(library.get(), "cuDeviceGetUuid"));
    const auto luid = reinterpret_cast<Luid>(GetProcAddress(library.get(), "cuDeviceGetLuid"));
    const auto pci = reinterpret_cast<Pci>(GetProcAddress(library.get(), "cuDeviceGetPCIBusId"));
    int devices = 0;
    if (!init || !count || !device || init(0) != 0 || count(&devices) != 0 || devices <= 0)
        throw std::runtime_error("Cannot enumerate NVIDIA adapters for the selected OpenCL GPU");
    std::vector<VideoDeviceIdentity> identities(static_cast<size_t>(devices));
    for (int ordinal = 0; ordinal < devices; ++ordinal) {
        int handle = 0;
        if (device(&handle, ordinal) != 0)
            continue;
        auto& identity = identities[ordinal];
        std::array<uint8_t, 16> id{};
        if (uuid && uuid(id.data(), handle) == 0)
            identity.uuid = id;
        std::array<uint8_t, 8> winId{};
        unsigned mask = 0;
        if (luid && luid(reinterpret_cast<char*>(winId.data()), &mask, handle) == 0) {
            identity.luid = winId;
            identity.nodeMask = mask;
        }
        char address[64]{};
        unsigned domain = 0, bus = 0, slot = 0, function = 0;
        if (pci && pci(address, sizeof(address), handle) == 0 && std::sscanf(address, "%x:%x:%x.%x", &domain, &bus, &slot, &function) == 4) {
            identity.domain = domain;
            identity.bus = bus;
            identity.slot = slot;
            identity.function = function;
        }
    }
    return static_cast<int>(matchingVideoDevice(openclVideoDeviceIdentity(selectedDevice), identities));
}
inline int bindHostNvencDevice(const cl_device_id selectedDevice, OptionDict& options) {
    const int ordinal = nvidiaEncoderDevice(selectedDevice);
    const std::string selected = std::to_string(ordinal);
    const auto* requested = av_dict_get(options.get(), "gpu", nullptr, 0);
    if (requested && requested->value != selected && std::string_view(requested->value) != "-1" && std::string_view(requested->value) != "any")
        throw std::runtime_error("NVENC -gpu conflicts with the GPU selected for OpenCL processing");
    checkAv(av_dict_set(options.ptr(), "gpu", selected.c_str(), 0), "Bind NVENC to the selected OpenCL GPU");
    return ordinal;
}
} // namespace video_utils
