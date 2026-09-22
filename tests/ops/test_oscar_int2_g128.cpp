#include "ops/kv_cache/oscar_int2_g128.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kMagic[] = "OSCAR21\0";
constexpr std::uint32_t kVersion = 1;
constexpr int kD = ninfer::ops::kOscarInt2G128HeadDim;

struct Reader {
    explicit Reader(const std::string& path) : stream(path, std::ios::binary) {
        if (!stream) { throw std::runtime_error("cannot open fixture: " + path); }
    }
    template <typename T>
    T read() {
        T value{};
        stream.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!stream) { throw std::runtime_error("truncated fixture"); }
        return value;
    }
    std::string string() {
        const std::uint32_t length = read<std::uint32_t>();
        if (length > 1U << 20U) { throw std::runtime_error("fixture string is unreasonable"); }
        std::string value(length, '\0');
        stream.read(value.data(), static_cast<std::streamsize>(length));
        if (!stream) { throw std::runtime_error("truncated fixture string"); }
        return value;
    }
    void bytes(void* destination, std::size_t count) {
        stream.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
        if (!stream) { throw std::runtime_error("truncated fixture payload"); }
    }
    std::ifstream stream;
};

template <typename T>
void write_json_string(std::ostream& out, const T& value) {
    out << '"' << value << '"';
}

float max_abs_difference(const float* left, const float* right, int count) {
    float result = 0.0F;
    for (int index = 0; index < count; ++index) {
        result = std::max(result, std::fabs(left[index] - right[index]));
    }
    return result;
}

float relative_l2(const float* left, const float* right, int count) {
    double difference = 0.0;
    double reference  = 0.0;
    for (int index = 0; index < count; ++index) {
        const double a = left[index];
        const double b = right[index];
        difference += (a - b) * (a - b);
        reference += a * a;
    }
    return static_cast<float>(std::sqrt(difference) /
                              std::max(std::sqrt(reference), 1.0e-30));
}

int run(const std::string& fixture_path, const std::string& report_path) {
    Reader reader(fixture_path);
    std::array<char, sizeof(kMagic) - 1> magic{};
    reader.bytes(magic.data(), magic.size());
    if (std::memcmp(magic.data(), kMagic, magic.size()) != 0 ||
        reader.read<std::uint32_t>() != kVersion) {
        throw std::runtime_error("unsupported OSCAR D2.1 fixture header");
    }
    const std::uint32_t case_count = reader.read<std::uint32_t>();
    if (case_count == 0 || case_count > 1000) {
        throw std::runtime_error("invalid OSCAR D2.1 fixture case count");
    }

    std::ofstream report(report_path, std::ios::trunc);
    if (!report) { throw std::runtime_error("cannot open parity report: " + report_path); }
    report << "{\n  \"schema\": \"oscar-d2-1-cpp-parity-v1\",\n"
           << "  \"passed\": true,\n  \"cases\": [\n";

    std::uint64_t total_rows = 0;
    float max_scale_error = 0.0F;
    float max_decoded_error = 0.0F;
    float max_clipped_error = 0.0F;
    float max_decoded_relative_l2 = 0.0F;
    bool first_case = true;

    for (std::uint32_t case_index = 0; case_index < case_count; ++case_index) {
        const std::string name        = reader.string();
        const std::string source_kind = reader.string();
        const std::int32_t source_layer = reader.read<std::int32_t>();
        const std::uint32_t rows      = reader.read<std::uint32_t>();
        const float clip_ratio        = reader.read<float>();
        if (rows == 0 || rows > 10000) { throw std::runtime_error("invalid fixture row count"); }

        float case_scale_error = 0.0F;
        float case_decoded_error = 0.0F;
        float case_clipped_error = 0.0F;
        float case_decoded_relative_l2 = 0.0F;
        for (std::uint32_t row = 0; row < rows; ++row) {
            std::array<float, kD> input{};
            std::array<float, kD> expected_clipped{};
            std::array<float, 4> expected_scales_zeros{};
            std::array<std::uint8_t, kD> expected_symbols{};
            std::array<std::uint8_t, 64> expected_packed{};
            std::array<float, kD> expected_decoded{};
            reader.bytes(input.data(), sizeof(input));
            reader.bytes(expected_clipped.data(), sizeof(expected_clipped));
            reader.bytes(expected_scales_zeros.data(), sizeof(expected_scales_zeros));
            reader.bytes(expected_symbols.data(), expected_symbols.size());
            reader.bytes(expected_packed.data(), expected_packed.size());
            reader.bytes(expected_decoded.data(), sizeof(expected_decoded));

            const auto encoded = ninfer::ops::oscar_int2_g128_encode(
                input.data(), kD, clip_ratio);
            std::array<float, kD> decoded{};
            ninfer::ops::oscar_int2_g128_decode(encoded, decoded.data(), kD);
            const float clipped_error = max_abs_difference(
                encoded.clipped.data(), expected_clipped.data(), kD);
            const float scale_error = max_abs_difference(
                encoded.scales_zeros.data(), expected_scales_zeros.data(), 4);
            const float decoded_error = max_abs_difference(
                decoded.data(), expected_decoded.data(), kD);
            const float decoded_rel = relative_l2(
                decoded.data(), expected_decoded.data(), kD);
            case_clipped_error = std::max(case_clipped_error, clipped_error);
            case_scale_error = std::max(case_scale_error, scale_error);
            case_decoded_error = std::max(case_decoded_error, decoded_error);
            case_decoded_relative_l2 = std::max(case_decoded_relative_l2, decoded_rel);
            max_clipped_error = std::max(max_clipped_error, clipped_error);
            max_scale_error = std::max(max_scale_error, scale_error);
            max_decoded_error = std::max(max_decoded_error, decoded_error);
            max_decoded_relative_l2 = std::max(max_decoded_relative_l2, decoded_rel);

            if (clipped_error != 0.0F) {
                throw std::runtime_error("clipped mismatch at case " + name + " row " +
                                         std::to_string(row));
            }
            if (std::memcmp(encoded.symbols.data(), expected_symbols.data(), kD) != 0) {
                throw std::runtime_error("symbol mismatch at case " + name + " row " +
                                         std::to_string(row));
            }
            if (std::memcmp(encoded.packed.data(), expected_packed.data(), 64) != 0) {
                throw std::runtime_error("packed-byte mismatch at case " + name + " row " +
                                         std::to_string(row));
            }
            if (scale_error > 2.0e-6F || decoded_error > 2.0e-6F || decoded_rel > 2.0e-6F) {
                throw std::runtime_error("floating parity mismatch at case " + name + " row " +
                                         std::to_string(row));
            }
            for (float value : decoded) {
                if (!std::isfinite(value)) { throw std::runtime_error("non-finite decode"); }
            }
        }
        total_rows += rows;
        if (!first_case) { report << ",\n"; }
        first_case = false;
        report << "    {\"name\": ";
        write_json_string(report, name);
        report << ", \"source_kind\": ";
        write_json_string(report, source_kind);
        report << ", \"source_layer\": " << source_layer << ", \"rows\": " << rows
               << ", \"clip_ratio\": " << std::setprecision(9) << clip_ratio
               << ", \"max_clipped_abs\": " << case_clipped_error
               << ", \"max_scale_abs\": " << case_scale_error
               << ", \"max_decoded_abs\": " << case_decoded_error
               << ", \"max_decoded_relative_l2\": " << case_decoded_relative_l2 << "}";
    }
    report << "\n  ],\n  \"total_rows\": " << total_rows
           << ",\n  \"max_clipped_abs\": " << max_clipped_error
           << ",\n  \"max_scale_abs\": " << max_scale_error
           << ",\n  \"max_decoded_abs\": " << max_decoded_error
           << ",\n  \"max_decoded_relative_l2\": " << max_decoded_relative_l2 << "\n}\n";
    report.close();
    std::cout << "OscarInt2G128 parity: PASS cases=" << case_count
              << " rows=" << total_rows << " max_scale_abs=" << std::scientific
              << max_scale_error << " max_decoded_abs=" << max_decoded_error
              << " max_decoded_rel_l2=" << max_decoded_relative_l2 << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 || std::string(argv[1]) != "--fixture" ||
            std::string(argv[3]) != "--report") {
            std::cerr << "usage: ninfer_oscar_int2_g128_test --fixture <golden.bin> "
                         "--report <report.json>\n";
            return 2;
        }
        return run(argv[2], argv[4]);
    } catch (const std::exception& error) {
        std::cerr << "OscarInt2G128 parity: FAIL: " << error.what() << "\n";
        return 1;
    }
}
