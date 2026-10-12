#include "product/load_progress/load_progress.h"
#include "serve/console_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"
#include "serve/webui_update.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <filesystem>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

std::string format_bytes(std::size_t bytes) {
    constexpr double kMiB = 1024.0 * 1024.0;
    constexpr double kGiB = 1024.0 * kMiB;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (static_cast<double>(bytes) >= kGiB) {
        out << static_cast<double>(bytes) / kGiB << " GiB";
    } else {
        out << static_cast<double>(bytes) / kMiB << " MiB";
    }
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error, exception.what());
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error, exception.what());
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    try {
        // Resolve (and, in --webui mode, auto-download) the webui directory before
        // the port is taken so a failed download aborts startup cleanly. In
        // --webui-dir mode the directory is trusted to already hold a built UI;
        // fail early if it does not.
        if (options.webui_auto) {
            options.webui_dir =
                ninfer::serve::ensure_webui_available(ninfer::serve::resolve_webui_dir(options));
        } else if (!options.webui_dir.empty()) {
            std::error_code ec;
            const bool have_index =
                std::filesystem::exists(std::filesystem::path(options.webui_dir) / "index.html", ec);
            if (!std::filesystem::is_directory(options.webui_dir, ec) || !have_index) {
                throw std::invalid_argument(
                    "--webui-dir must be a directory containing index.html: " + options.webui_dir);
            }
        }
        using Clock = std::chrono::steady_clock;
        ninfer::serve::HttpServer server(options);
        if (!server.bind()) {
            ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error,
                                             "failed to bind " + options.host + ':' +
                                                 std::to_string(options.port));
            return 1;
        }

        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, "loading model...");
        auto load_progress_options        = ninfer::product::stderr_load_progress_options();
        load_progress_options.line_prefix = [] {
            return ninfer::serve::current_console_log_prefix(ninfer::serve::ConsoleLogLevel::Info);
        };
        ninfer::product::LoadProgressRenderer load_progress(std::cerr,
                                                            std::move(load_progress_options));
        const auto load_start = Clock::now();
        ninfer::serve::GenerationService service(options, load_progress.callback());
        std::ostringstream loaded;
        loaded << "model loaded in "
               << std::chrono::duration<double>(Clock::now() - load_start).count() << " s";
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, loaded.str());

        const ninfer::MemorySummary memory       = service.memory_summary();
        const ninfer::EngineOptions& engine      = service.engine_options();
        const ninfer::ContextCacheOptions& cache = engine.context_cache;
        std::ostringstream capacity;
        capacity << "lanes=" << engine.max_concurrency << " max-context=" << engine.max_context
                 << " weights=" << format_bytes(memory.weights.capacity_bytes)
                 << " runtime=" << format_bytes(memory.runtime_reservation_bytes)
                 << " free-after-startup=" << format_bytes(memory.available_after_startup_bytes)
                 << " host-kv=" << format_bytes(cache.host_kv_capacity_bytes)
                 << " host-state=" << cache.host_state_slots
                 << " shared=" << *cache.max_shared_prefixes
                 << " speculative=dflash2-k7 vision=" << (engine.vision_enabled() ? "bf16" : "off");
        if (options.vision_enabled()) {
            const ninfer::MediaCacheSummary media = service.media_cache_summary();
            capacity << " media-workers=" << media.preprocess_threads
                     << " media-cache=" << format_bytes(media.capacity_bytes)
                     << " media-live=" << format_bytes(media.live_capacity_bytes);
        }
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, capacity.str());

        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, "warming up...");
        service.warmup();
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::ostringstream listening;
        listening << "listening on http://" << options.host << ':' << options.port
                  << " (model id: " << server.public_model_id()
                  << ", auth: " << (options.api_key.empty() ? "disabled" : "bearer") << ')';
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Info, listening.str());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error,
                                             "failed to bind " + options.host + ':' +
                                                 std::to_string(options.port));
            return 1;
        }
        return 0;
    } catch (const std::exception& exception) {
        ninfer::serve::write_console_log(ninfer::serve::ConsoleLogLevel::Error, exception.what());
        return 1;
    }
}
