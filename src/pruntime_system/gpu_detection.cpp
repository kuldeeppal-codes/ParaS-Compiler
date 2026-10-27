/**
 * Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
 *
 * This file is part of the ParaS Compiler, a component of the ParaS Ecosystem.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/gpu_detection.hpp"
#include "utilities/internal_utils.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cctype>

namespace paras_extension {

static inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static inline std::string strip_label_prefix(std::string s) {

    auto pos = s.find(':');
    if (pos != std::string::npos)
        s = s.substr(pos + 1);
    trim_inplace(s);
    return s;
}

static unsigned int get_compute_units_nvidia(const std::string& raw_name) {
    std::string n = raw_name;
    std::transform(n.begin(), n.end(), n.begin(), ::toupper);

    if (n.find("GH200") != std::string::npos)
        return 132;
    if (n.find("H200") != std::string::npos)
        return 132;
    if (n.find("H100") != std::string::npos)
        return 132;
    if (n.find("H800") != std::string::npos)
        return 132;

    if (n.find("A100") != std::string::npos)
        return 108;
    if (n.find("A40") != std::string::npos)
        return 84;
    if (n.find("A10G") != std::string::npos)
        return 72;
    if (n.find("A10") != std::string::npos)
        return 72;

    if (n.find("V100") != std::string::npos)
        return 80;

    if (n.find("T4") != std::string::npos)
        return 40;

    if (n.find("RTX 4090") != std::string::npos)
        return 128;
    if (n.find("RTX 4080") != std::string::npos)
        return 76;

    return 0;
}

static inline std::string detect_cuda_version_from_nvidia_smi() {
    std::vector<std::string> lines;
    if (!run_cmd_lines_cached("nvidia-smi 2>/dev/null | head -n 5", lines))
        return {};
    for (auto& ln : lines) {
        auto pos = ln.find("CUDA Version:");
        if (pos == std::string::npos)
            continue;
        std::string s = ln.substr(pos + std::string("CUDA Version:").size());
        trim_inplace(s);
        auto sp = s.find(' ');
        if (sp != std::string::npos)
            s = s.substr(0, sp);
        trim_inplace(s);
        return s;
    }
    return {};
}

static inline std::vector<std::string> split_csv_4(const std::string& ln) {
    std::vector<std::string> tok;
    size_t start = 0;
    while (true) {
        size_t end = ln.find(',', start);
        if (end == std::string::npos) {
            tok.push_back(ln.substr(start));
            break;
        }
        tok.push_back(ln.substr(start, end - start));
        start = end + 1;
    }
    if (tok.size() > 4) {
        std::string name = tok[0];
        for (size_t i = 1; i + 3 < tok.size(); ++i)
            name += "," + tok[i];
        std::vector<std::string> out;
        out.reserve(4);
        out.push_back(std::move(name));
        out.push_back(tok[tok.size() - 3]);
        out.push_back(tok[tok.size() - 2]);
        out.push_back(tok[tok.size() - 1]);
        return out;
    }
    while (tok.size() < 4)
        tok.push_back("");
    return tok;
}

static inline bool is_uuid_token(const std::string& token) {
    return token.rfind("GPU-", 0) == 0 || token.rfind("MIG-", 0) == 0;
}

static inline long uuid_index(const std::string& token, const std::vector<std::string>& uuids) {
    const std::string want = to_lower(token);
    long found = -1;
    for (std::size_t i = 0; i < uuids.size(); ++i) {
        if (to_lower(uuids[i]).rfind(want, 0) != 0)
            continue;
        if (found >= 0)
            return -1;
        found = static_cast<long>(i);
    }
    return found;
}

static inline std::vector<std::size_t>
get_visible_indices(const char* envName, const std::size_t physicalDeviceCount,
                    const std::vector<std::string>& uuids = {}) {
    std::vector<std::size_t> indices;

    const char* env = std::getenv(envName);

    if (env == nullptr) {
        for (std::size_t i = 0; i < physicalDeviceCount; ++i) {
            indices.push_back(i);
        }
        return indices;
    }

    std::string visibleDevices(env);
    trim_inplace(visibleDevices);

    if (visibleDevices.empty() || visibleDevices == "-1") {
        return indices;
    }

    std::vector<std::string> tokens;
    std::size_t start = 0;
    while (start <= visibleDevices.size()) {
        const std::size_t end = visibleDevices.find(',', start);
        std::string token = visibleDevices.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        trim_inplace(token);
        if (!token.empty())
            tokens.push_back(token);
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    const bool uuidsKnown = uuids.size() == physicalDeviceCount && physicalDeviceCount > 0;
    const bool allUuids = !tokens.empty() &&
                          std::all_of(tokens.begin(), tokens.end(), is_uuid_token);
    if (allUuids && !uuidsKnown) {
        for (std::size_t i = 0; i < tokens.size() && i < physicalDeviceCount; ++i)
            indices.push_back(i);
        return indices;
    }

    for (const std::string& token : tokens) {
        long physicalId = -1;
        if (is_uuid_token(token)) {
            if (uuidsKnown)
                physicalId = uuid_index(token, uuids);
        } else {
            char* parseEnd = nullptr;
            const long id = std::strtol(token.c_str(), &parseEnd, 10);
            if (parseEnd != token.c_str() && *parseEnd == '\0')
                physicalId = id;
        }

        if (physicalId >= 0 && static_cast<std::size_t>(physicalId) < physicalDeviceCount) {
            const std::size_t index = static_cast<std::size_t>(physicalId);
            if (std::find(indices.begin(), indices.end(), index) == indices.end()) {
                indices.push_back(index);
            }
        }
    }

    return indices;
}

static inline std::vector<std::string> nvidia_gpu_uuids() {
    std::vector<std::string> lines, uuids;
    if (!run_cmd_lines_cached("nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null", lines))
        return uuids;
    for (std::string ln : lines) {
        trim_inplace(ln);
        if (!ln.empty())
            uuids.push_back(ln);
    }
    return uuids;
}

static inline std::vector<std::string> amd_gpu_uuids() {
    std::vector<std::string> lines, uuids;
    if (!run_cmd_lines_cached("rocm-smi --showuniqueid 2>/dev/null", lines))
        return uuids;
    for (const std::string& ln : lines) {
        const auto p = ln.find("GPU[");
        const auto q = ln.find("Unique ID:");
        if (p == std::string::npos || q == std::string::npos)
            continue;
        const std::size_t idx = static_cast<std::size_t>(std::atoi(ln.c_str() + p + 4));
        std::string id = ln.substr(q + std::string("Unique ID:").size());
        trim_inplace(id);
        if (id.rfind("0x", 0) == 0 || id.rfind("0X", 0) == 0)
            id = id.substr(2);
        if (id.empty() || id.size() > 16 ||
            !std::all_of(id.begin(), id.end(), [](unsigned char c) { return std::isxdigit(c) != 0; }))
            continue;
        if (uuids.size() <= idx)
            uuids.resize(idx + 1);
        uuids[idx] = "GPU-" + std::string(16 - id.size(), '0') + to_lower(id);
    }
    if (std::any_of(uuids.begin(), uuids.end(), [](const std::string& u) { return u.empty(); }))
        uuids.clear();
    return uuids;
}

static inline std::vector<std::size_t>
get_visible_nvidia_indices(const std::size_t physicalDeviceCount) {
    return get_visible_indices("CUDA_VISIBLE_DEVICES", physicalDeviceCount, nvidia_gpu_uuids());
}

static inline std::vector<std::size_t>
get_visible_amd_indices(const std::size_t physicalDeviceCount) {
    const auto uuids = amd_gpu_uuids();
    const auto rocr = get_visible_indices("ROCR_VISIBLE_DEVICES", physicalDeviceCount, uuids);
    const char* hipEnv = std::getenv("HIP_VISIBLE_DEVICES") ? "HIP_VISIBLE_DEVICES"
                                                            : "CUDA_VISIBLE_DEVICES";
    std::vector<std::string> rocrUuids;
    if (uuids.size() == physicalDeviceCount)
        for (const std::size_t i : rocr)
            rocrUuids.push_back(uuids[i]);
    const auto hip = get_visible_indices(hipEnv, rocr.size(), rocrUuids);

    std::vector<std::size_t> indices;
    indices.reserve(hip.size());
    for (const std::size_t i : hip) {
        indices.push_back(rocr[i]);
    }
    return indices;
}

std::vector<sycl::device> detect_nvidia_gpus() {
    std::vector<sycl::device> out;

#ifdef __linux__
    std::vector<std::string> lines;
    if (!run_cmd_lines_cached("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap "
                              "--format=csv,noheader,nounits 2>/dev/null",
                              lines) ||
        lines.empty()) {
        return out;
    }

    static const std::string cuda_ver = detect_cuda_version_from_nvidia_smi();
    const auto visibleIndices = get_visible_nvidia_indices(lines.size());

    for (std::size_t logicalId = 0; logicalId < visibleIndices.size(); ++logicalId) {
        const std::size_t physicalId = visibleIndices[logicalId];
        const auto& ln = lines[physicalId];
        auto tokens = split_csv_4(ln);

        std::string name = tokens[0].empty() ? "NVIDIA GPU" : tokens[0];
        std::string mem = tokens[1].empty() ? "0" : tokens[1];
        std::string drv = tokens[2].empty() ? "Unknown" : tokens[2];
        std::string cap = tokens[3].empty() ? "unknown" : tokens[3];

        trim_inplace(name);
        trim_inplace(mem);
        trim_inplace(drv);
        trim_inplace(cap);

        const unsigned long mem_mib = parse_ul_or0(mem);
        const std::uint64_t mem_bytes = static_cast<std::uint64_t>(mem_mib) * 1024ULL * 1024ULL;
        const unsigned int computeUnits = get_compute_units_nvidia(name);

        std::string versionString = "Compute Capability " + cap;
        if (!cuda_ver.empty()) {
            versionString += " | CUDA Version " + cuda_ver;
        }

        out.emplace_back(::paras_extension::device_ctor_tag{}, name, "NVIDIA", drv, versionString,
                         computeUnits, 1024, mem_bytes, sycl::info::local_mem_type::local, false,
                         true, false, static_cast<int>(logicalId), true);
    }
#endif

    return out;
}

struct amd_gpu_info {
    int idx = -1;
    std::string name;
    std::string gfx;
    std::uint64_t vram_total_bytes = 0;
};

static inline std::vector<amd_gpu_info> parse_rocm_smi_product_and_vram() {
    std::vector<amd_gpu_info> gpus;

    auto get = [&](int idx) -> amd_gpu_info& {
        for (auto& g : gpus)
            if (g.idx == idx)
                return g;
        gpus.push_back({});
        gpus.back().idx = idx;
        return gpus.back();
    };

    {
        std::vector<std::string> lines;
        run_cmd_lines_cached("rocm-smi --showproductname 2>/dev/null", lines);

        for (auto& ln : lines) {
            auto p = ln.find("GPU[");
            if (p == std::string::npos)
                continue;
            auto q = ln.find("]", p);
            if (q == std::string::npos)
                continue;

            int idx = std::atoi(ln.substr(p + 4, q - (p + 4)).c_str());
            auto& g = get(idx);

            if (ln.find("Card Series:") != std::string::npos) {
                auto k = ln.find("Card Series:");
                std::string s = ln.substr(k + std::string("Card Series:").size());
                trim_inplace(s);
                g.name = s;
            } else if (ln.find("GFX Version:") != std::string::npos) {
                auto k = ln.find("GFX Version:");
                std::string s = ln.substr(k + std::string("GFX Version:").size());
                trim_inplace(s);
                g.gfx = s;
            }
        }
    }

    {
        std::vector<std::string> lines;
        run_cmd_lines_cached("rocm-smi --showmeminfo vram 2>/dev/null", lines);

        for (auto& ln : lines) {
            auto p = ln.find("GPU[");
            if (p == std::string::npos)
                continue;
            auto q = ln.find("]", p);
            if (q == std::string::npos)
                continue;

            int idx = std::atoi(ln.substr(p + 4, q - (p + 4)).c_str());

            auto key = std::string("VRAM Total Memory (B):");
            auto k = ln.find(key);
            if (k == std::string::npos)
                continue;

            std::string s = ln.substr(k + key.size());
            trim_inplace(s);

            std::uint64_t bytes = (std::uint64_t)parse_ul_or0(s);
            auto& g = get(idx);
            g.vram_total_bytes = bytes;
        }
    }

    std::sort(gpus.begin(), gpus.end(),
              [](const amd_gpu_info& a, const amd_gpu_info& b) { return a.idx < b.idx; });
    return gpus;
}

static inline std::string rocm_driver_version_clean() {
    std::vector<std::string> lines;
    if (!run_cmd_lines_cached("rocm-smi --showdriverversion 2>/dev/null", lines))
        return {};
    for (auto& ln : lines) {
        auto p = ln.find("Driver version:");
        if (p == std::string::npos)
            continue;
        std::string s = ln.substr(p + std::string("Driver version:").size());
        trim_inplace(s);
        return s;
    }
    return {};
}

static inline std::string rocm_release_version() {

    std::string v;
    if (const char* rp = std::getenv("ROCM_PATH"))
        v = run_cmd_first_line_cached(
            ("cat '" + std::string(rp) + "/.info/version' 2>/dev/null").c_str());
    if (v.empty())
        v = run_cmd_first_line_cached("cat /opt/rocm/.info/version 2>/dev/null");
    trim_inplace(v);
    return v;
}

static inline std::uint32_t amd_cu_fallback(const std::string& name, const std::string& gfx) {
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(), ::toupper);

    if (n.find("MI300X") != std::string::npos)
        return 304;
    if (n.find("MI300A") != std::string::npos)
        return 228;

    if (gfx == "gfx942")
        return 304;
    return 0;
}

std::vector<sycl::device> detect_amd_gpus() {
    std::vector<sycl::device> out;

#ifdef __linux__
    auto gpus = parse_rocm_smi_product_and_vram();
    if (gpus.empty())
        return out;

    const std::string drv = rocm_driver_version_clean();
    const std::string rel = rocm_release_version();

    const auto visibleIndices = get_visible_amd_indices(gpus.size());

    for (std::size_t logicalId = 0; logicalId < visibleIndices.size(); ++logicalId) {
        const auto& g = gpus[visibleIndices[logicalId]];
        std::string name = g.name.empty() ? "AMD GPU" : g.name;

        std::string version = "ROCm";
        if (!rel.empty())
            version += " " + rel;
        if (!g.gfx.empty())
            version += " | " + g.gfx;

        std::uint32_t cus = 0;
        if (cus == 0)
            cus = amd_cu_fallback(name, g.gfx);

        std::uint64_t mem_bytes = g.vram_total_bytes;

        out.emplace_back(::paras_extension::device_ctor_tag{}, name, "AMD",
                         drv.empty() ? "unknown" : drv, version, cus, 1024, mem_bytes,
                         sycl::info::local_mem_type::local, false, true, false,
                         static_cast<int>(logicalId), false);
    }
#endif

    return out;
}

std::vector<sycl::device> detect_all_gpus() {
    std::vector<sycl::device> out;
    auto n = detect_nvidia_gpus();
    out.insert(out.end(), n.begin(), n.end());

    auto a = detect_amd_gpus();
    out.insert(out.end(), a.begin(), a.end());

    return out;
}

} // namespace paras_extension
