#pragma once

struct RingAttentionInput {
    std::vector<std::uint16_t> q, rk, rv, k, v;
    int start = 0, count = 0, queries = 1, block = 1;
    float scale = 1.0f / std::sqrt(128.0f);
};

// Independent FP64 softmax over explicit logical slots. The original GPU
// bitwise differential is the E0 gate; this bound checks numerical consistency
// including half rounding and conservative FP32 dot/sum propagation.
std::pair<double, double> ring_attention_cpu_check(const RingAttentionInput& x,
    const std::vector<std::uint16_t>& output) {
    double max_error = 0, max_ratio = 0;
    std::vector<int> queries{0};
    if (x.queries > 1) queries.push_back(x.queries - 1);
    constexpr double u = 0x1p-24;
    const double gamma_dot = 128 * u / (1 - 128 * u);
    for (int query : queries) for (int head : {0, 7, 8, 31}) {
        std::vector<const std::uint16_t*> keys, values;
        const int first = std::max(0, x.count + query - 2047);
        for (int logical = first; logical < x.count; ++logical) {
            const int slot = (x.start + logical) % 2048;
            keys.push_back(x.rk.data() + slot * 1024 + (head / 4) * 128);
            values.push_back(x.rv.data() + slot * 1024 + (head / 4) * 128);
        }
        for (int j = 0; j < x.block; ++j) {
            keys.push_back(x.k.data() + j * 1024 + (head / 4) * 128);
            values.push_back(x.v.data() + j * 1024 + (head / 4) * 128);
        }
        const auto* q = x.q.data() + (query * 32 + head) * 128;
        std::vector<double> scores;
        double max_abs_sum = 0;
        for (const auto* key : keys) {
            double dot = 0, absolute = 0;
            for (int d = 0; d < 128; ++d) {
                const double product = static_cast<double>(half_to_float(q[d])) * half_to_float(key[d]);
                dot += product; absolute += std::abs(product);
            }
            scores.push_back(dot * x.scale);
            max_abs_sum = std::max(max_abs_sum, absolute * x.scale);
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double denominator = 0;
        for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
        const double gamma_sum = scores.size() * u / (1 - scores.size() * u);
        for (int d : {0, 1, 63, 127}) {
            double acc = 0, max_v = 0;
            for (std::size_t j = 0; j < scores.size(); ++j) {
                const double value = half_to_float(values[j][d]);
                acc += scores[j] * value; max_v = std::max(max_v, std::abs(value));
            }
            const double reference = acc / denominator;
            const double actual = half_to_float(output[(query * 32 + head) * 128 + d]);
            const double bound = 0x1p-11 * std::abs(reference) + 0x1p-24 +
                (2 * gamma_sum + 4 * gamma_dot * max_abs_sum + 8 * u) * max_v;
            const double error = std::abs(actual - reference);
            require(std::isfinite(reference) && std::isfinite(actual) && error <= bound,
                    "ring attention independent FP64 consistency failed");
            max_error = std::max(max_error, error); max_ratio = std::max(max_ratio, error / bound);
        }
    }
    return {max_error, max_ratio};
}

void check_ring_attention_case(const RingAttentionInput& x, const std::string& name,
    std::ostream& report, std::ostream* timing = nullptr,
    const std::vector<std::uint16_t>* production = nullptr) {
    require(x.q.size() == static_cast<std::size_t>(x.queries) * 4096 &&
            x.rk.size() == 2048 * 1024 && x.rv.size() == x.rk.size() &&
            x.k.size() == static_cast<std::size_t>(x.block) * 1024 && x.v.size() == x.k.size(),
            "ring qualification input extents");
    constexpr int guard = 32;
    constexpr std::uint16_t canary = 0x7bff;
    const std::size_t count = x.q.size();
    std::array<const std::vector<std::uint16_t>*, 5> host{&x.q, &x.rk, &x.rv, &x.k, &x.v};
    std::array<std::unique_ptr<DeviceBuffer>, 5> device;
    for (int i = 0; i < 5; ++i) {
        device[i] = std::make_unique<DeviceBuffer>(host[i]->size() * 2);
        cuda_check(cudaMemcpy(device[i]->get(), host[i]->data(), device[i]->bytes(), cudaMemcpyHostToDevice), "ring input upload");
    }
    auto ptr = [&](int i) { return static_cast<const std::uint16_t*>(device[i]->get()); };
    DeviceBuffer out((count + 2 * guard) * 2);
    auto* base = static_cast<std::uint16_t*>(out.get());
    std::vector<std::uint16_t> initial(count + 2 * guard, canary), serial;
    double max_error = 0, max_ratio = 0;
    auto launch = [&](bool parallel) {
        ninfer::exl3::dflash2_ring_attention_for_test(ptr(0), ptr(1), ptr(2), x.start, x.count,
            ptr(3), ptr(4), base + guard, x.queries, x.block, x.scale, parallel);
    };
    for (bool parallel : {false, true}) {
        cuda_check(cudaMemcpy(base, initial.data(), out.bytes(), cudaMemcpyHostToDevice), "ring output canaries");
        launch(parallel);
        std::vector<std::uint16_t> got(initial.size()), repeat(initial.size());
        cuda_check(cudaMemcpy(got.data(), base, out.bytes(), cudaMemcpyDeviceToHost), "ring output download");
        launch(parallel);
        cuda_check(cudaMemcpy(repeat.data(), base, out.bytes(), cudaMemcpyDeviceToHost), "ring repeat download");
        require(got == repeat, "ring repeat mismatch");
        for (std::size_t i = 0; i < got.size(); ++i)
            if (i < guard || i >= guard + count) require(got[i] == canary, "ring output guard changed");
        std::vector<std::uint16_t> values(got.begin() + guard, got.end() - guard);
        require(std::all_of(values.begin(), values.end(), [](auto h) { return std::isfinite(half_to_float(h)); }),
                "ring nonfinite output");
        if (!parallel) serial = values;
        else require(values == serial, "ring original-GPU bitwise mismatch: " + name);
        if (production) require(values == *production, "ring actual production output mismatch");
        const auto error = ring_attention_cpu_check(x, values);
        max_error = std::max(max_error, error.first); max_ratio = std::max(max_ratio, error.second);
    }
    for (int i = 0; i < 5; ++i) {
        std::vector<std::uint16_t> after(host[i]->size());
        cuda_check(cudaMemcpy(after.data(), device[i]->get(), device[i]->bytes(), cudaMemcpyDeviceToHost), "ring input integrity");
        require(after == *host[i], "ring input mutated");
    }
    // Admission/no-op checks must not touch output.
    for (bool parallel : {false, true}) {
        cuda_check(cudaMemcpy(base, initial.data(), out.bytes(), cudaMemcpyHostToDevice), "ring invalid guards");
        ninfer::exl3::dflash2_ring_attention_for_test(nullptr, nullptr, nullptr, 0, 0,
            nullptr, nullptr, base + guard, 0, 1, x.scale, parallel);
        for (int bad = 0; bad < 6; ++bad) {
            bool rejected = false;
            try {
                ninfer::exl3::dflash2_ring_attention_for_test(ptr(0), ptr(1), ptr(2),
                    bad == 0 ? 2048 : x.start, bad == 1 ? 2048 : x.count, ptr(3), ptr(4), base + guard,
                    bad == 2 ? 9 : x.queries, bad == 3 ? 0 : x.block,
                    bad == 4 ? 0.0f : (bad == 5 ? std::numeric_limits<float>::quiet_NaN() : x.scale), parallel);
            } catch (const std::exception&) { rejected = true; }
            require(rejected, "ring invalid geometry accepted");
        }
        std::vector<std::uint16_t> after(initial.size());
        cuda_check(cudaMemcpy(after.data(), base, out.bytes(), cudaMemcpyDeviceToHost), "ring invalid output");
        require(after == initial, "ring invalid entry mutated output");
    }
    if (timing) {
        cudaEvent_t a, b;
        cuda_check(cudaEventCreate(&a), "ring timing create");
        cuda_check(cudaEventCreate(&b), "ring timing create");
        for (int repeat = -3; repeat < 21; ++repeat) for (bool parallel : {false, true}) {
            cuda_check(cudaEventRecord(a), "ring timer start"); launch(parallel);
            cuda_check(cudaEventRecord(b), "ring timer end"); cuda_check(cudaEventSynchronize(b), "ring timer sync");
            float ms; cuda_check(cudaEventElapsedTime(&ms, a, b), "ring timer elapsed");
            if (repeat >= 0) *timing << name << ',' << repeat << ',' << parallel << ',' << ms * 1000 << '\n';
        }
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    report << name << ',' << x.count << ',' << x.start << ',' << x.queries << ',' << x.block
           << ",PASS,PASS,PASS,PASS," << max_error << ',' << max_ratio << '\n';
    report.flush();
}

RingAttentionInput synthetic_ring_input(int count, int start, int queries, int block, int pattern) {
    RingAttentionInput x; x.count = count; x.start = start; x.queries = queries; x.block = block;
    x.q.resize(queries * 4096); x.rk.resize(2048 * 1024); x.rv.resize(x.rk.size());
    x.k.resize(block * 1024); x.v.resize(x.k.size());
    std::uint32_t state = 0x83a71;
    for (auto* values : {&x.q, &x.rk, &x.rv, &x.k, &x.v}) for (auto& h : *values) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        h = static_cast<std::uint16_t>((state & 0x8000) | (0x3000 + (state & 0x7ff)));
    }
    if (pattern != 0) {
        std::fill(x.q.begin(), x.q.end(), 0);
        std::fill(x.rk.begin(), x.rk.end(), 0);
        std::fill(x.k.begin(), x.k.end(), 0);
    }
    if (pattern == 2) {
        std::fill(x.rv.begin(), x.rv.end(), 0x3c00);
        std::fill(x.v.begin(), x.v.end(), 0xbc00);
        for (int logical = 0; logical < 8; ++logical) {
            const int slot = (start + logical) % 2048;
            std::fill_n(x.rv.begin() + slot * 1024, 1024, 0x5000);
        }
    }
    return x;
}

void run_ring_attention_qualification(const std::string& path) {
    require(!path.empty(), "ring qualification output required");
    std::ofstream out(path), timing(path + ".timing.csv");
    require(out.good() && timing.good(), "ring qualification outputs");
    out << "case,count,start,queries,block,gpu_exact,cpu_fp64,repeat,guards_input,max_abs,max_bound_ratio\n";
    timing << "case,repeat,parallel,gpu_us\n";
    int cases = 0;
    auto run = [&](int count, int start, int queries, int block, int pattern) {
        auto x = synthetic_ring_input(count, start, queries, block, pattern);
        const auto name = "c" + std::to_string(count) + "-s" + std::to_string(start) +
            "-q" + std::to_string(queries) + "-b" + std::to_string(block) + "-p" + std::to_string(pattern);
        check_ring_attention_case(x, name, out,
            pattern == 0 && start == 0 && queries == 8 && block == 8 && (count == 512 || count == 2047) ? &timing : nullptr);
        ++cases;
    };
    for (int count : {0, 1, 15, 127, 512, 2040, 2041, 2046, 2047})
        for (int start : {0, 1023, 2047}) for (int q : {1, 2, 5, 8})
            for (int pattern : {0, 1}) run(count, start, q, q, pattern);
    for (int q : {3, 4, 6, 7}) run(512, 0, q, q, 0);
    for (int count : {2040, 2041, 2046, 2047}) run(count, 2047, 8, 8, 2);
    run(2047, 1023, 8, 1, 0); run(2047, 1023, 1, 8, 0);
    require(cases == 226 && out.good() && timing.good(), "ring qualification completion");
    std::cout << "RINGQUAL PASS cases=226 gpu_exact=1 cpu_fp64=1 repeat=1 guards_input=1\n";
}

struct RingAttentionCapture {
    Exl3Dflash2DraftModel& model;
    std::filesystem::path directory;
    std::array<bool, 5> seen{};
    RingAttentionCapture(Exl3Dflash2DraftModel& m, const std::string& path) : model(m), directory(path) {
        require(!std::filesystem::exists(directory) && std::filesystem::create_directories(directory), "new ring capture directory required");
        model.set_ring_attention_observer_for_test(observe, this);
    }
    ~RingAttentionCapture() { model.set_ring_attention_observer_for_test(nullptr); }
    static void observe(const Exl3Dflash2DraftModel::RingAttentionObservation& o, void* user) {
        auto& self = *static_cast<RingAttentionCapture*>(user);
        require(o.layer >= 0 && o.layer < 5, "ring capture layer");
        if (self.seen[o.layer]) return;
        cuda_check(cudaStreamSynchronize(o.stream), "ring capture input/output ready");
        RingAttentionInput x; x.start=o.start; x.count=o.count; x.queries=o.queries; x.block=o.block; x.scale=o.scale;
        x.q.resize(o.queries*4096); x.rk.resize(2048*1024); x.rv.resize(x.rk.size()); x.k.resize(o.block*1024); x.v.resize(x.k.size());
        std::vector<std::uint16_t> production(x.q.size());
        const std::array<const std::uint16_t*,6> device{o.q,o.ring_k,o.ring_v,o.k,o.v,o.output};
        const std::array<std::vector<std::uint16_t>*,6> host{&x.q,&x.rk,&x.rv,&x.k,&x.v,&production};
        const auto dir=self.directory/("layer"+std::to_string(o.layer));std::filesystem::create_directory(dir);
        const std::array<const char*,6> names{"q","ring_k","ring_v","k","v","output"};
        for(int i=0;i<6;++i){
            cuda_check(cudaMemcpy(host[i]->data(),device[i],host[i]->size()*2,cudaMemcpyDeviceToHost),"ring real capture");
            std::ofstream raw(dir/(std::string(names[i])+".f16"),std::ios::binary);
            raw.write(reinterpret_cast<const char*>(host[i]->data()),host[i]->size()*2);require(raw.good(),"ring raw output");
        }
        std::ofstream out(dir/"cases.csv"),timing(dir/"timing.csv");
        out<<"case,count,start,queries,block,gpu_exact,cpu_fp64,repeat,guards_input,max_abs,max_bound_ratio\n";
        timing<<"case,repeat,parallel,gpu_us\n";
        check_ring_attention_case(x,"real-layer"+std::to_string(o.layer),out,&timing,&production);
        require(out.good()&&timing.good(),"ring real report");self.seen[o.layer]=true;
    }
    void finish() {
        require(std::all_of(seen.begin(),seen.end(),[](bool b){return b;}),"ring real capture incomplete");
        std::cout<<"RINGREAL PASS layers=5 gpu_exact=1 cpu_fp64=1 production_exact=1\n";
    }
};
