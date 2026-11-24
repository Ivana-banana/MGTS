// Optional Windows DLL used by the MGTS occupancy-grid benchmark.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <unordered_map>

namespace {
struct Cell {
    std::int32_t x;
    std::int32_t y;

    bool operator==(const Cell& other) const noexcept {
        return x == other.x && y == other.y;
    }
};

struct CellHash {
    std::size_t operator()(const Cell& cell) const noexcept {
        const auto high = static_cast<std::uint64_t>(static_cast<std::uint32_t>(cell.x));
        const auto low = static_cast<std::uint32_t>(cell.y);
        return std::hash<std::uint64_t>{}((high << 32) | low);
    }
};

using Evidence = std::unordered_map<Cell, int, CellHash>;

// Round the displacement before translating it by start, as in Grid.observe.
// For example, from x=1 a displacement of +0.5 lands at x=1, not x=2.
std::int64_t rounded_offset(std::uint64_t index, std::int64_t delta,
                            std::uint64_t steps) noexcept {
    const auto magnitude = static_cast<std::uint64_t>(std::abs(delta));
    std::uint64_t rounded = index;
    if (magnitude != steps) {
        // Both factors fit in uint32, including for opposite int32 endpoints.
        const auto numerator = index * magnitude;
        rounded = numerator / steps;
        const auto twice_remainder = 2 * (numerator % steps);
        if (twice_remainder > steps || (twice_remainder == steps && rounded % 2 != 0)) {
            ++rounded;
        }
    }
    const auto offset = static_cast<std::int64_t>(rounded);
    return delta < 0 ? -offset : offset;
}
}  // namespace

extern "C" __declspec(dllexport) void* grid_create() noexcept {
    try {
        return new Evidence;
    } catch (...) {
        return nullptr;
    }
}

extern "C" __declspec(dllexport) void grid_destroy(void* grid) noexcept {
    delete static_cast<Evidence*>(grid);
}

extern "C" __declspec(dllexport) int grid_observe(void* grid, const std::int32_t* start_xy,
                                                 const std::int32_t* cells_xy,
                                                 std::size_t count) noexcept {
    if (!grid || (count && (!start_xy || !cells_xy))
              || count > std::numeric_limits<std::size_t>::max() / 2) {
        return 1;
    }
    if (count == 0) {
        return 0;
    }
    try {
        Evidence updates;
        const Cell start{start_xy[0], start_xy[1]};
        for (std::size_t i = 0; i < count; ++i) {
            const Cell end{cells_xy[2*i], cells_xy[2*i+1]};
            updates.insert_or_assign(end, 3);
            const auto dx = static_cast<std::int64_t>(end.x) - start.x;
            const auto dy = static_cast<std::int64_t>(end.y) - start.y;
            const auto steps = static_cast<std::uint64_t>(std::max(std::abs(dx), std::abs(dy)));
            for (std::uint64_t j = 0; j < steps; ++j) {
                const Cell free_cell{
                    static_cast<std::int32_t>(start.x + rounded_offset(j, dx, steps)),
                    static_cast<std::int32_t>(start.y + rounded_offset(j, dy, steps))
                };
                // Repeated rays vote once. An endpoint always keeps its occupied vote.
                updates.try_emplace(free_cell, -1);
            }
        }

        auto& evidence = *static_cast<Evidence*>(grid);
        if (updates.size() > evidence.max_size() - evidence.size()) {
            return 1;
        }
        // Allocate buckets before modifying evidence. merge transfers existing
        // nodes for new cells, so committing a scan needs no node allocations.
        evidence.reserve(evidence.size() + updates.size());
        for (const auto& entry : updates) {
            auto pos = evidence.find(entry.first);
            if (pos != evidence.end()) {
                pos->second = std::clamp(pos->second + entry.second, -12, 12);
            }
        }
        evidence.merge(updates);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" __declspec(dllexport) std::size_t grid_size(void* grid) noexcept {
    return grid ? static_cast<Evidence*>(grid)->size() : 0;
}

// capacity counts cells. The caller provides three int32 values per cell.
extern "C" __declspec(dllexport) int grid_export(void* grid, std::int32_t* rows,
                                                std::size_t capacity) noexcept {
    if (!grid) {
        return 1;
    }
    const auto& evidence = *static_cast<Evidence*>(grid);
    if (capacity < evidence.size() || (!rows && !evidence.empty())
            || evidence.size() > std::numeric_limits<std::size_t>::max() / 3) {
        return 1;
    }
    for (const auto& entry : evidence) {
        *rows++ = entry.first.x;
        *rows++ = entry.first.y;
        *rows++ = entry.second;
    }
    return 0;
}
