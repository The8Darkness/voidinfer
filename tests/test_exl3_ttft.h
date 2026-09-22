#pragma once

// Diagnostic engine-process TTFT, measured to a logical emitted token ID.
// Includes Windows process creation/loader time. No detokenization or transport
// exists in this harness. Control-after-spec is not a separate cold request.
class TtftTrace {
public:
    struct Stamp {
        std::chrono::steady_clock::time_point monotonic;
        std::uint64_t utc_100ns;
    };
    static Stamp stamp() {
        FILETIME now{};
        GetSystemTimePreciseAsFileTime(&now);
        return {std::chrono::steady_clock::now(), bits(now)};
    }
    TtftTrace(const std::string& path, Stamp entry) : entry_(entry) {
        require(!active_, "one TTFT trace per process");
        require(!std::filesystem::exists(path), "TTFT output must be new");
        FILETIME created{}, exited{}, kernel{}, user{};
        require(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user) != 0,
                "TTFT process creation query failed");
        created_ = bits(created);
        out_.open(path); out_.imbue(std::locale::classic());
        require(out_.good(), "TTFT output open failed");
        out_ << "scope,context,stage,main_elapsed_ms,process_elapsed_ms,utc_100ns,process_created_100ns,position,token_id,cuda_sync_before_mark\n";
        write("main_entry", entry_, -1, -1, false);
        active_ = this;
        mark("trace_ready");
    }
    ~TtftTrace() { active_ = nullptr; }
    static void scope(const char* name, int context = 0) {
        if (active_) { active_->scope_ = name; active_->context_ = context; }
    }
    static void mark(const char* stage, int position = -1,
                     std::int64_t token = -1, bool synchronize = false) {
        if (!active_) return;
        if (synchronize) cuda_check(cudaDeviceSynchronize(), "TTFT diagnostic completion");
        const auto now = stamp();
        active_->write(stage, now, position, token, synchronize);
        const auto label = std::string("ttft.") + active_->scope_ + "." + stage;
        nvtxMarkA(label.c_str());
    }
private:
    static std::uint64_t bits(FILETIME t) {
        return (static_cast<std::uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
    }
    void write(const char* stage, Stamp now, int position, std::int64_t token, bool sync) {
        require(now.utc_100ns >= created_ && now.utc_100ns >= last_utc_,
                "TTFT wall clock moved backwards");
        const double elapsed = std::chrono::duration<double, std::milli>(now.monotonic - entry_.monotonic).count();
        require(elapsed >= 0, "TTFT monotonic timestamp precedes entry");
        out_ << scope_ << ',' << context_ << ',' << stage << ',' << std::fixed << std::setprecision(6)
             << elapsed << ',' << (now.utc_100ns - created_) / 10000.0 << ','
             << now.utc_100ns << ',' << created_ << ',' << position << ',' << token << ',' << sync << '\n';
        out_.flush(); require(out_.good(), "TTFT output write failed");
        last_utc_ = now.utc_100ns;
    }
    inline static TtftTrace* active_ = nullptr;
    Stamp entry_;
    std::uint64_t created_ = 0, last_utc_ = 0;
    const char* scope_ = "process";
    int context_ = 0;
    std::ofstream out_;
};
