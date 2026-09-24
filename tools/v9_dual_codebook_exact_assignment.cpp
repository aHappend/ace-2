#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

namespace {

inline int64_t bf16_significand(uint16_t bits) {
    const int64_t sign = (bits & 0x8000U) ? -1 : 1;
    const uint16_t exponent = (bits >> 7U) & 0xFFU;
    const uint16_t fraction = bits & 0x7FU;
    if (exponent == 0U && fraction == 0U) {
        return 0;
    }
    return sign * static_cast<int64_t>(exponent == 0U ? fraction : fraction + 128U);
}

inline int bf16_shift(uint16_t bits, int zero_shift) {
    const uint16_t exponent = (bits >> 7U) & 0xFFU;
    const uint16_t fraction = bits & 0x7FU;
    if (exponent == 0xFFU) {
        return std::numeric_limits<int>::max();
    }
    if (exponent == 0U && fraction == 0U) {
        return zero_shift;
    }
    return exponent == 0U ? -133 : static_cast<int>(exponent) - 134;
}

inline int record_exponent(uint32_t record) {
    const uint8_t raw = static_cast<uint8_t>((record >> 16U) & 0xFFU);
    return raw >= 128U ? static_cast<int>(raw) - 256 : static_cast<int>(raw);
}

inline __int128 objective_term(
    uint16_t source,
    uint32_t record,
    int8_t point,
    int zero_shift,
    int objective_exponent
) {
    const int64_t weight_significand = bf16_significand(source);
    if (weight_significand == 0) {
        const int64_t scale_significand = static_cast<int64_t>(record & 0xFFFFU);
        const int scale_exponent = record_exponent(record);
        const int q_shift = 2 * scale_exponent - 38 - objective_exponent;
        if (q_shift < 0 || q_shift >= 120) {
            return 0;
        }
        __int128 quadratic = static_cast<__int128>(point) * point;
        quadratic *= static_cast<__int128>(scale_significand) * scale_significand;
        return quadratic << q_shift;
    }
    const int weight_exponent = bf16_shift(source, zero_shift);
    if (weight_exponent == std::numeric_limits<int>::max()) {
        return 0;
    }
    const int64_t scale_significand = static_cast<int64_t>(record & 0xFFFFU);
    const int scale_exponent = record_exponent(record);
    const int l_shift = weight_exponent + scale_exponent - 19 - objective_exponent;
    const int q_shift = 2 * scale_exponent - 38 - objective_exponent;
    if (l_shift < 0 || q_shift < 0 || l_shift >= 120 || q_shift >= 120) {
        return 0;
    }
    __int128 linear = -2 * static_cast<__int128>(point) * weight_significand * scale_significand;
    linear <<= l_shift;
    __int128 quadratic = static_cast<__int128>(point) * point;
    quadratic *= static_cast<__int128>(scale_significand) * scale_significand;
    quadratic <<= q_shift;
    return linear + quadratic;
}

inline double bf16_value(uint16_t bits) {
    const int64_t significand = bf16_significand(bits);
    if (significand == 0) {
        return 0.0;
    }
    return std::ldexp(static_cast<double>(significand), bf16_shift(bits, 0));
}

}  // namespace

extern "C" int v9_assign_selectors_exact(
    const uint16_t* source,
    int64_t rows,
    int64_t width,
    const uint32_t* records,
    const uint8_t* assignments,
    const int8_t* codebooks,
    int64_t group_size,
    int shift_min,
    int objective_exponent,
    uint8_t* selectors,
    uint64_t* chosen_cost_low,
    int64_t* chosen_cost_high,
    uint64_t* bank_b_count,
    uint64_t* tie_count
) {
    if (!source || !records || !assignments || !codebooks || !selectors || !chosen_cost_low ||
        !chosen_cost_high || !bank_b_count || !tie_count) {
        return 1;
    }
    if (rows <= 0 || width <= 0 || width % 128 != 0 || group_size <= 0 || rows % group_size != 0) {
        return 2;
    }
    const int64_t blocks = width / 128;
    const int64_t groups = rows / group_size;
    uint64_t selected_b = 0;
    uint64_t ties = 0;

    #pragma omp parallel for schedule(static) reduction(+:selected_b,ties)
    for (int64_t group = 0; group < groups; ++group) {
        const int64_t row_start = group * group_size;
        const int64_t row_stop = row_start + group_size;
        for (int64_t block = 0; block < blocks; ++block) {
            __int128 costs[2] = {0, 0};
            for (int bank = 0; bank < 2; ++bank) {
                const int8_t* points = codebooks + (block * 2 + bank) * 16;
                for (int64_t row = row_start; row < row_stop; ++row) {
                    const uint32_t record = records[row];
                    const int64_t offset = row * width + block * 128;
                    for (int lane = 0; lane < 128; ++lane) {
                        const uint8_t index = assignments[offset + lane];
                        if (index >= 16U) {
                            continue;
                        }
                        costs[bank] += objective_term(
                            source[offset + lane], record, points[index], shift_min, objective_exponent
                        );
                    }
                }
            }
            const bool choose_b = costs[1] < costs[0];
            const int64_t selector_index = group * blocks + block;
            const __int128 chosen = costs[choose_b ? 1 : 0];
            selectors[selector_index] = choose_b ? 1U : 0U;
            chosen_cost_low[selector_index] = static_cast<uint64_t>(chosen);
            chosen_cost_high[selector_index] = static_cast<int64_t>(chosen >> 64);
            selected_b += choose_b ? 1U : 0U;
            ties += costs[0] == costs[1] ? 1U : 0U;
        }
    }
    *bank_b_count = selected_b;
    *tie_count = ties;
    return 0;
}

extern "C" int v9_assign_payload_exact(
    const uint16_t* source,
    int64_t rows,
    int64_t width,
    const uint32_t* records,
    const int8_t* codebooks,
    const uint8_t* selectors,
    int64_t group_size,
    int base_shift,
    uint8_t* assignments,
    int64_t* weight_codepoint,
    int64_t* codepoint_squared,
    int64_t* absolute_maximum
) {
    if (!source || !records || !codebooks || !selectors || !assignments || !weight_codepoint ||
        !codepoint_squared || !absolute_maximum) {
        return 1;
    }
    if (rows <= 0 || width <= 0 || width % 128 != 0 || group_size <= 0 || rows % group_size != 0) {
        return 2;
    }
    const int64_t blocks = width / 128;

    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const uint32_t record = records[row];
        const double scale = std::ldexp(
            static_cast<double>(record & 0xFFFFU), record_exponent(record) - 15
        );
        const int64_t group = row / group_size;
        int64_t row_weight_codepoint = 0;
        int64_t row_codepoint_squared = 0;
        int64_t row_absolute_maximum = 0;
        for (int64_t block = 0; block < blocks; ++block) {
            const uint8_t bank = selectors[group * blocks + block];
            const int8_t* points = codebooks + (block * 2 + bank) * 16;
            const int64_t offset = row * width + block * 128;
            for (int lane = 0; lane < 128; ++lane) {
                const double value = bf16_value(source[offset + lane]);
                uint8_t index = 0;
                while (index < 15U) {
                    const double boundary = scale * static_cast<double>(
                        static_cast<int>(points[index]) + static_cast<int>(points[index + 1])
                    ) / 32.0;
                    if (value <= boundary) {
                        break;
                    }
                    ++index;
                }
                assignments[offset + lane] = index;
                const int64_t point = points[index];
                const uint16_t bits = source[offset + lane];
                const int64_t significand = bf16_significand(bits);
                int64_t exact = 0;
                if (significand != 0) {
                    const int delta = bf16_shift(bits, base_shift) - base_shift;
                    if (delta < 0 || delta > 54) {
                        continue;
                    }
                    exact = static_cast<int64_t>(
                        static_cast<__int128>(significand) * (static_cast<__int128>(1) << delta)
                    );
                }
                row_weight_codepoint += exact * point;
                row_codepoint_squared += point * point;
                const int64_t magnitude = exact < 0 ? -exact : exact;
                if (magnitude > row_absolute_maximum) {
                    row_absolute_maximum = magnitude;
                }
            }
        }
        weight_codepoint[row] = row_weight_codepoint;
        codepoint_squared[row] = row_codepoint_squared;
        absolute_maximum[row] = row_absolute_maximum;
    }
    return 0;
}

extern "C" int v9_accumulate_codebook_aggregates(
    const uint16_t* source,
    int64_t rows,
    int64_t width,
    const uint32_t* records,
    const uint8_t* assignments,
    const uint8_t* selectors,
    int64_t group_size,
    int shift_min,
    int objective_exponent,
    uint64_t* linear_low,
    int64_t* linear_high,
    uint64_t* quadratic_low,
    int64_t* quadratic_high,
    uint64_t* counts
) {
    if (!source || !records || !assignments || !selectors || !linear_low || !linear_high ||
        !quadratic_low || !quadratic_high || !counts) {
        return 1;
    }
    if (rows <= 0 || width <= 0 || width % 128 != 0 || group_size <= 0 || rows % group_size != 0) {
        return 2;
    }
    const int64_t blocks = width / 128;

    #pragma omp parallel for schedule(static)
    for (int64_t block_bank = 0; block_bank < blocks * 2; ++block_bank) {
        const int64_t block = block_bank / 2;
        const uint8_t bank = static_cast<uint8_t>(block_bank % 2);
        __int128 linear[16] = {};
        __int128 quadratic[16] = {};
        uint64_t local_counts[16] = {};
        for (int64_t row = 0; row < rows; ++row) {
            const int64_t group = row / group_size;
            if (selectors[group * blocks + block] != bank) {
                continue;
            }
            const uint32_t record = records[row];
            const int64_t scale_significand = static_cast<int64_t>(record & 0xFFFFU);
            const int scale_exponent = record_exponent(record);
            const int q_shift = 2 * scale_exponent - 38 - objective_exponent;
            if (q_shift < 0 || q_shift >= 120) {
                continue;
            }
            const __int128 q_value = (
                static_cast<__int128>(scale_significand) * scale_significand
            ) << q_shift;
            const int64_t offset = row * width + block * 128;
            for (int lane = 0; lane < 128; ++lane) {
                const uint8_t cluster = assignments[offset + lane];
                if (cluster >= 16U) {
                    continue;
                }
                const uint16_t bits = source[offset + lane];
                const int64_t weight_significand = bf16_significand(bits);
                if (weight_significand != 0) {
                    const int weight_exponent = bf16_shift(bits, shift_min);
                    const int l_shift = weight_exponent + scale_exponent - 19 - objective_exponent;
                    if (l_shift < 0 || l_shift >= 120) {
                        continue;
                    }
                    linear[cluster] += (
                        static_cast<__int128>(weight_significand) * scale_significand
                    ) << l_shift;
                }
                quadratic[cluster] += q_value;
                local_counts[cluster] += 1U;
            }
        }
        for (int cluster = 0; cluster < 16; ++cluster) {
            const int64_t output = block_bank * 16 + cluster;
            linear_low[output] = static_cast<uint64_t>(linear[cluster]);
            linear_high[output] = static_cast<int64_t>(linear[cluster] >> 64);
            quadratic_low[output] = static_cast<uint64_t>(quadratic[cluster]);
            quadratic_high[output] = static_cast<int64_t>(quadratic[cluster] >> 64);
            counts[output] = local_counts[cluster];
        }
    }
    return 0;
}
