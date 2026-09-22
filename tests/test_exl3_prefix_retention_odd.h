#pragma once

void run_prefix_retention_odd_qualification(
    Exl3TextModel& target, const std::vector<std::int64_t>& source_ids) {
    require(target.max_context() == 4096 && !source_ids.empty(),
            "prefixretainodd requires maxctx4096 and real prompt IDs");
    require(env("NINFER_OSCAR_EXL3") == "1",
            "prefixretainodd requires canonical OSCAR");
    std::vector<std::int64_t> ids(4096);
    for (std::size_t index = 0; index < ids.size(); ++index)
        ids[index] = source_ids[index % source_ids.size()];

    const std::array<int, 7> prefixes{63, 64, 319, 320, 575, 576, 2047};
    int cases = 0;
    for (const int prefix : prefixes) {
        auto serial = target.create_context(true);
        auto candidate = target.create_context(true);
        require(serial->try_enable_oscar_from_environment() &&
                    candidate->try_enable_oscar_from_environment(),
                "prefixretainodd OSCAR enable");
        serial->prepare_transaction();
        candidate->prepare_transaction();
        candidate->prepare_continuation(8);
        const std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + prefix);
        ingest_prefix(*serial, prompt, CommitSink{});
        ingest_prefix(*candidate, prompt, CommitSink{});
        const auto base = prefix_retention_snapshot(*candidate);
        require(base == prefix_retention_snapshot(*serial),
                "prefixretainodd paired base mismatch");
        const std::size_t persistent = candidate->persistent_bytes();

        for (const int block : {3, 5, 6}) {
            for (int retained = 1; retained <= block; ++retained) {
                const std::string label = "prefixretainodd prefix=" + std::to_string(prefix) +
                    " B=" + std::to_string(block) + " retained=" + std::to_string(retained);
                candidate->begin_transaction();
                serial->begin_transaction();
                const std::span<const std::int64_t> teacher(ids.data() + prefix, block);
                candidate->continue_rows(teacher);
                const auto attempted_logits = candidate->continuation_logits_bits_host();
                prefix_retention_finite(attempted_logits, false, label + " attempted logits");

                std::vector<std::uint16_t> serial_logits, serial_embedding;
                std::array<std::vector<std::uint16_t>, 5> serial_taps;
                for (int row = 0; row < retained; ++row) {
                    serial->decode(teacher[static_cast<std::size_t>(row)]);
                    const auto logits = target_continue_device_bits(
                        serial->logits_device(), kVocab, label + " serial logits");
                    serial_logits.insert(serial_logits.end(), logits.begin(), logits.end());
                    const auto embedding = serial->embedding_bits_host_for_test();
                    serial_embedding.insert(serial_embedding.end(), embedding.begin(), embedding.end());
                    for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
                        const auto values = target_continue_tap_bits(*serial, kTapLayers[tap], 1);
                        serial_taps[tap].insert(serial_taps[tap].end(), values.begin(), values.end());
                    }
                }

                candidate->retain_transaction_prefix(retained);
                require(candidate->transaction_active(), label + " closed recovery checkpoint");
                const auto actual = prefix_retention_snapshot(*candidate);
                const auto expected = prefix_retention_snapshot(*serial);
                require_prefix_retention_semantic(actual, expected, label + " retained state");
                require(actual.metadata == std::array<int, 6>{prefix + retained,
                            prefix + retained - 1, retained, retained, retained, retained},
                        label + " retained metadata");
                require(actual.embedding == serial_embedding && actual.taps == serial_taps,
                        label + " retained payload mismatch");
                require(candidate->continuation_logits_bits_host() == serial_logits,
                        label + " retained all-row logits mismatch");
                require(std::equal(serial_logits.begin(), serial_logits.end(),
                                   attempted_logits.begin()),
                        label + " modified precomputed logits");
                require(candidate->persistent_bytes() == persistent,
                        label + " allocated persistent bytes during retention");

                for (int step = 0; step < 4; ++step) {
                    const auto token = ids[static_cast<std::size_t>(prefix + retained + step)];
                    candidate->decode(token);
                    serial->decode(token);
                    require(prefix_retention_snapshot(*candidate) ==
                                prefix_retention_snapshot(*serial),
                            label + " following M1 state/payload mismatch");
                }
                candidate->rollback_transaction();
                serial->rollback_transaction();
                require(prefix_retention_snapshot(*candidate) == base &&
                            prefix_retention_snapshot(*serial) == base,
                        label + " reusable base rollback mismatch");
                ++cases;
                std::cout << "PREFIXRETAINODD_CASE PASS prefix=" << prefix
                          << " B=" << block << " retained=" << retained
                          << " gdn=48 conv_slots=4 oscar_live=exact payload=exact continuation=4"
                          << std::endl;
            }
        }
    }
    require(cases == 98, "prefixretainodd matrix count mismatch");
    std::cout << "PREFIXRETAINODD PASS cases=98 blocks=3|5|6 prefixes=7"
              << " gdn=48 conv_slots=4 oscar_live=exact payload=exact continuation=4"
              << " new_device_bytes=0 guards=inherited caller_ring=unqualified" << std::endl;
}
