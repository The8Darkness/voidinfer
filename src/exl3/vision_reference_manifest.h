#pragma once

#include "exl3/vision_model.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::exl3 {

inline bool exl3_vision_sha256_text(std::string_view hash) noexcept {
    if(hash.size()!=64)return false;
    for(char value:hash)if(!((value>='0'&&value<='9') ||
        (value>='a'&&value<='f') || (value>='A'&&value<='F')))return false;
    return true;
}

enum class Exl3VisionReferenceScalar : std::uint8_t { fp16,fp32 };

struct Exl3VisionReferenceArtifact {
    int boundary=0;
    std::size_t logical_rows=0,logical_columns=0,stored_elements=0;
    Exl3VisionReferenceScalar scalar=Exl3VisionReferenceScalar::fp32;
    std::filesystem::path native_file,reference_file;
    std::string native_sha256,reference_sha256;

    [[nodiscard]] std::size_t scalar_bytes() const noexcept {
        return scalar==Exl3VisionReferenceScalar::fp16?2:4;
    }
    [[nodiscard]] std::uint64_t stored_bytes() const {
        if(stored_elements>std::numeric_limits<std::uint64_t>::max()/scalar_bytes())
            throw std::overflow_error("V6 reference stored extent overflow");
        return stored_elements*scalar_bytes();
    }
    void validate() const {
        if(!logical_rows || !logical_columns ||
           logical_rows>std::numeric_limits<std::size_t>::max()/logical_columns ||
           logical_rows*logical_columns!=stored_elements)
            throw std::invalid_argument("V6 reference logical/stored extent mismatch");
        if(native_file.empty() || reference_file.empty() || native_file==reference_file ||
           native_file.is_absolute() || reference_file.is_absolute())
            throw std::invalid_argument("V6 reference artifact path separation");
        if(!exl3_vision_sha256_text(native_sha256) ||
           !exl3_vision_sha256_text(reference_sha256))
            throw std::invalid_argument("V6 reference artifact hash identity");
    }
};

struct Exl3VisionReferenceManifest {
    std::string model_revision,weights_sha256,prepared_payload_sha256;
    std::string reference_adapter_sha256,reference_runtime_identity;
    std::size_t patches=0,merged_tokens=0;
    std::vector<Exl3VisionReferenceArtifact> artifacts;

    void validate() const {
        if(model_revision.empty() || reference_runtime_identity.empty() || !patches ||
           !merged_tokens || patches!=merged_tokens*4 || artifacts.empty())
            throw std::invalid_argument("V6 reference manifest identity/geometry");
        for(const auto* hash:{&weights_sha256,&prepared_payload_sha256,&reference_adapter_sha256})
            if(!exl3_vision_sha256_text(*hash))
                throw std::invalid_argument("V6 reference manifest hash identity");
        for(std::size_t i=0;i<artifacts.size();++i) {
            artifacts[i].validate();
            for(std::size_t prior=0;prior<i;++prior)
                if(artifacts[prior].boundary==artifacts[i].boundary)
                    throw std::invalid_argument("V6 reference duplicate boundary");
        }
        const auto require=[&](int boundary) {
            for(const auto& artifact:artifacts)if(artifact.boundary==boundary)return;
            throw std::invalid_argument("V6 reference required boundary missing");
        };
        for(int boundary=0;boundary<=10;++boundary)
            require(Exl3VisionBoundaryCode::block_operator(26,boundary));
        require(Exl3VisionBoundaryCode::block_residual(26));
        require(Exl3VisionBoundaryCode::merger_normalized);
        require(Exl3VisionBoundaryCode::merger_up);
        require(Exl3VisionBoundaryCode::merger_output);
    }
};

class Exl3VisionReferenceAdapter {
    Exl3VisionReferenceManifest manifest_;
public:
    explicit Exl3VisionReferenceAdapter(Exl3VisionReferenceManifest manifest)
        :manifest_(std::move(manifest)){manifest_.validate();}
    [[nodiscard]] const Exl3VisionReferenceManifest& manifest() const noexcept{return manifest_;}
    [[nodiscard]] const Exl3VisionReferenceArtifact& require_boundary(int boundary,
        std::size_t rows,std::size_t columns,Exl3VisionReferenceScalar scalar) const {
        for(const auto& artifact:manifest_.artifacts)if(artifact.boundary==boundary) {
            if(artifact.logical_rows!=rows || artifact.logical_columns!=columns ||
               artifact.scalar!=scalar)
                throw std::invalid_argument("V6 reference output contract mismatch");
            return artifact;
        }
        throw std::invalid_argument("V6 reference boundary missing");
    }
};

} // namespace ninfer::exl3
