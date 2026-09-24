#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kSelectors = 16;
constexpr int kDeltas = 16;
constexpr int kLanes = 128;
constexpr int kMaximumGroups = 38;

struct Candidate {
  int selector;
  int delta;
  int64_t scale;
};

inline int quantize_exact(int64_t value, int64_t scale) {
  const int64_t target = value * 2;
  int low = 0;
  int high = 15;
  while (low < high) {
    const int middle = (low + high) / 2;
    const int odd = -15 + 2 * middle;
    if (odd * scale < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low - 8;
}

inline __int128 candidate_cost(
    const int64_t* prefix,
    int64_t scale,
    uint8_t positions[15]) {
  const int64_t total = prefix[kLanes];
  __int128 linear = static_cast<__int128>(7) * total;
  __int128 quadratic = static_cast<__int128>(64) * kLanes;
  for (int boundary = 0; boundary < 15; ++boundary) {
    const int position = positions[boundary];
    const int odd = -15 + 2 * boundary;
    linear -= prefix[position];
    quadratic += static_cast<__int128>(odd) * (kLanes - position);
  }
  const __int128 exact_scale = scale;
  return -2 * exact_scale * linear + exact_scale * exact_scale * quadratic;
}

}  // namespace

extern "C" int v8_assign_exact(
    const int64_t* exact,
    const int64_t* sorted_exact,
    int rows,
    int groups,
    const int64_t* table_significands,
    const int8_t* table_exponents,
    int common_shift,
    int threads,
    int8_t* payload,
    uint8_t* selectors,
    int8_t* deltas) {
  if (exact == nullptr || sorted_exact == nullptr || table_significands == nullptr ||
      table_exponents == nullptr || payload == nullptr || selectors == nullptr ||
      deltas == nullptr || rows <= 0 || groups <= 0 || groups > kMaximumGroups) {
    return 1;
  }

  std::vector<Candidate> candidates;
  candidates.reserve(kSelectors * kDeltas);
  for (int selector = 0; selector < kSelectors; ++selector) {
    const int exponent = static_cast<int>(table_exponents[selector]);
    const int64_t significand = table_significands[selector];
    if (significand < 32768 || significand > 65535 || exponent < -24 || exponent > 4) {
      return 2;
    }
    for (int delta = -8; delta <= 7; ++delta) {
      const int effective_exponent = exponent + delta;
      if (effective_exponent < -24 || effective_exponent > 4) {
        continue;
      }
      const int shift = effective_exponent - 15 - common_shift;
      if (shift < 0 || shift > 47) {
        return 3;
      }
      const int64_t scale = significand << shift;
      if (scale <= 0 || scale > std::numeric_limits<int64_t>::max() / 15) {
        return 4;
      }
      candidates.push_back({selector, delta, scale});
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& left, const Candidate& right) {
    if (left.scale != right.scale) return left.scale < right.scale;
    if (left.selector != right.selector) return left.selector < right.selector;
    return left.delta < right.delta;
  });
  if (candidates.empty() || candidates.size() > kSelectors * kDeltas) {
    return 5;
  }

#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  int error = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(max : error)
  for (int row = 0; row < rows; ++row) {
    __int128 row_cost[kSelectors] = {};
    bool row_cost_set[kSelectors] = {};
    int8_t best_delta[kSelectors][kMaximumGroups] = {};

    for (int group = 0; group < groups; ++group) {
      const int64_t* sorted = sorted_exact +
          (static_cast<int64_t>(row) * groups + group) * kLanes;
      int64_t prefix[kLanes + 1];
      prefix[0] = 0;
      for (int lane = 0; lane < kLanes; ++lane) {
        prefix[lane + 1] = prefix[lane] + sorted[lane];
      }
      uint8_t positions[kSelectors * kDeltas][15] = {};

      for (int boundary = 0; boundary < 15; ++boundary) {
        const int odd = -15 + 2 * boundary;
        int position = 0;
        if (odd < 0) {
          for (int candidate_index = static_cast<int>(candidates.size()) - 1;
               candidate_index >= 0; --candidate_index) {
            const int64_t threshold = odd * candidates[candidate_index].scale;
            while (position < kLanes && sorted[position] * 2 <= threshold) {
              ++position;
            }
            positions[candidate_index][boundary] = static_cast<uint8_t>(position);
          }
        } else {
          for (int candidate_index = 0;
               candidate_index < static_cast<int>(candidates.size()); ++candidate_index) {
            const int64_t threshold = odd * candidates[candidate_index].scale;
            while (position < kLanes && sorted[position] * 2 <= threshold) {
              ++position;
            }
            positions[candidate_index][boundary] = static_cast<uint8_t>(position);
          }
        }
      }

      __int128 group_cost[kSelectors] = {};
      bool group_cost_set[kSelectors] = {};
      int group_delta[kSelectors] = {};
      for (int candidate_index = 0;
           candidate_index < static_cast<int>(candidates.size()); ++candidate_index) {
        const Candidate& candidate = candidates[candidate_index];
        const __int128 cost = candidate_cost(
            prefix, candidate.scale, positions[candidate_index]);
        const int selector = candidate.selector;
        if (!group_cost_set[selector] || cost < group_cost[selector] ||
            (cost == group_cost[selector] && candidate.delta < group_delta[selector])) {
          group_cost_set[selector] = true;
          group_cost[selector] = cost;
          group_delta[selector] = candidate.delta;
        }
      }
      for (int selector = 0; selector < kSelectors; ++selector) {
        if (!group_cost_set[selector]) {
          error = std::max(error, 6);
          continue;
        }
        row_cost[selector] += group_cost[selector];
        row_cost_set[selector] = true;
        best_delta[selector][group] = static_cast<int8_t>(group_delta[selector]);
      }
    }

    int selected = 0;
    if (!row_cost_set[0]) {
      error = std::max(error, 7);
      continue;
    }
    for (int selector = 1; selector < kSelectors; ++selector) {
      if (!row_cost_set[selector]) {
        error = std::max(error, 7);
        continue;
      }
      if (row_cost[selector] < row_cost[selected]) {
        selected = selector;
      }
    }
    selectors[row] = static_cast<uint8_t>(selected);

    const int exponent = static_cast<int>(table_exponents[selected]);
    const int64_t significand = table_significands[selected];
    for (int group = 0; group < groups; ++group) {
      const int delta = static_cast<int>(best_delta[selected][group]);
      deltas[static_cast<int64_t>(row) * groups + group] = static_cast<int8_t>(delta);
      const int shift = exponent + delta - 15 - common_shift;
      const int64_t scale = significand << shift;
      const int64_t base = (static_cast<int64_t>(row) * groups + group) * kLanes;
      for (int lane = 0; lane < kLanes; ++lane) {
        const int64_t value = exact[base + lane];
        if (value > std::numeric_limits<int64_t>::max() / 2 ||
            value < std::numeric_limits<int64_t>::min() / 2) {
          error = std::max(error, 8);
          payload[base + lane] = 0;
        } else {
          payload[base + lane] = static_cast<int8_t>(quantize_exact(value, scale));
        }
      }
    }
  }
  return error;
}
