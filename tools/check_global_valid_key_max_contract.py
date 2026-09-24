#!/usr/bin/env python3
"""Decisive 257-key discriminator for the proposal's global-row maximum."""

from __future__ import annotations

import hashlib
import json
import struct


EMPTY = "EMPTY"
SCAN = "SCAN"
SEALED = "SEALED"
EMIT = "EMIT"

HEADS = 14
SCORE_BYTES = 2
BEAT_BYTES = 16
MAX_CONTEXT = 32768
SRAM_MARGIN_BYTES = 29168
COMPOSE_SCORE_READ_PASSES = 3
NONZERO_COMPLETION_TAGS = (1 << 16) - 1


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def canonical_tiles(length: int, tile: int = 256) -> list[int]:
    return [min(tile, length - base) for base in range(0, length, tile)]


def score_row_stride(length: int) -> int:
    return BEAT_BYTES * ceil_div(SCORE_BYTES * length, BEAT_BYTES)


def score_matrix_bytes(length: int) -> int:
    return HEADS * score_row_stride(length)


def descriptor_cycles(kind: str, length: int, shift: int = 87) -> int:
    if kind == "single":
        return (
            3826 * length
            + shift * length
            + 2 * ceil_div(length, 4)
            + 2 * ceil_div(length, 2)
            + ceil_div(length, 8)
            + 6
        )
    if kind == "max_scan":
        return (
            3822 * length
            + shift * length
            + 2 * ceil_div(length, 4)
            + 8
        )
    if kind == "center_emit":
        return (
            3826 * length
            + shift * length
            + 2 * ceil_div(length, 4)
            + ceil_div(length, 8)
            + 8
        )
    raise AssertionError(f"unknown descriptor kind: {kind}")


def score_producer_cycles(length: int) -> int:
    tiles = canonical_tiles(length)
    coefficient_dma = 8 * length + 9 * len(tiles)
    raw_k_dma = 17 * length + 8 * len(tiles)
    if len(tiles) == 1:
        return HEADS * descriptor_cycles("single", length) + coefficient_dma + raw_k_dma
    descriptor_total = sum(
        descriptor_cycles("max_scan", tile_length)
        + descriptor_cycles("center_emit", tile_length)
        for tile_length in tiles
    )
    return HEADS * descriptor_total + 2 * (coefficient_dma + raw_k_dma)


def compose_score_read_cycles(length: int) -> int:
    return COMPOSE_SCORE_READ_PASSES * HEADS * ceil_div(length, 8)


def producer_input_append_external_bytes(length: int) -> int:
    phases = 1 if length <= 256 else 2
    tile_count = len(canonical_tiles(length))
    raw_k = phases * 272 * length
    coefficients = phases * (128 * length + 16 * tile_count)
    return raw_k + coefficients + 272


def complete_external_bytes(length: int) -> int:
    matrix = score_matrix_bytes(length)
    return (
        producer_input_append_external_bytes(length)
        + matrix
        + COMPOSE_SCORE_READ_PASSES * matrix
    )


def descriptor_population(length: int) -> dict[str, int]:
    tile_count = len(canonical_tiles(length))
    score_and_dma = (
        2 + HEADS if tile_count == 1 else 2 * tile_count * (2 + HEADS)
    )
    compose = COMPOSE_SCORE_READ_PASSES * HEADS * ceil_div(length, 8)
    root = 1
    return {
        "root": root,
        "score_and_dma": score_and_dma,
        "compose": compose,
        "total": root + score_and_dma + compose,
    }


def simulate_rolling_tags(descriptor_count: int) -> dict[str, int]:
    scoreboard = [False] * (NONZERO_COMPLETION_TAGS + 1)
    seen: set[int] = set()
    previous_tag = 0
    live_slots = 0
    max_live = 0
    reuse_count = 0
    first_reuse_ordinal: int | None = None

    for ordinal in range(descriptor_count):
        completion_tag = 1 + ordinal % NONZERO_COMPLETION_TAGS
        dependency_tag = 0 if ordinal == 0 else previous_tag

        if dependency_tag:
            assert scoreboard[dependency_tag]
            scoreboard[dependency_tag] = False
            live_slots -= 1

        assert not scoreboard[completion_tag]
        if completion_tag in seen:
            reuse_count += 1
            if first_reuse_ordinal is None:
                first_reuse_ordinal = ordinal
        seen.add(completion_tag)
        scoreboard[completion_tag] = True
        live_slots += 1
        previous_tag = completion_tag
        max_live = max(max_live, live_slots)

    assert first_reuse_ordinal is not None
    assert live_slots == 1
    assert scoreboard[previous_tag]
    return {
        "capacity": NONZERO_COMPLETION_TAGS,
        "descriptor_count": descriptor_count,
        "first_reuse_ordinal": first_reuse_ordinal,
        "max_live_success_slots": max_live,
        "reuse_count": reuse_count,
        "reuse_distance": NONZERO_COMPLETION_TAGS,
        "terminal_leased_tag": previous_tag,
        "terminal_live_slots": live_slots,
    }


def sram_descriptor_bytes(kind: str, length: int) -> int:
    common = 336 * length + 32 * ceil_div(length, 4)
    if kind == "single":
        return 80 + common + 32 * ceil_div(length, 2)
    if kind in {"max_scan", "center_emit"}:
        return 112 + common
    raise AssertionError(f"unknown descriptor kind: {kind}")


def center(value: int, maximum: int) -> int:
    return max(-32768, min(0, value - maximum))


def payload_sha256(values: list[int]) -> str:
    payload = b"".join(struct.pack("<h", value) for value in values)
    return hashlib.sha256(payload).hexdigest()


def scan_tile(
    state: dict[str, int | str | None], tile: list[int], key_base: int, total: int
) -> dict[str, int | str | None]:
    assert tile
    assert key_base % 256 == 0
    assert len(tile) == min(256, total - key_base)
    if key_base == 0:
        assert state == {"phase": EMPTY, "global_max": None, "next_key_base": 0}
        maximum = max(tile)
    else:
        assert state["phase"] == SCAN
        assert state["next_key_base"] == key_base
        maximum = max(int(state["global_max"]), max(tile))
    next_key_base = key_base + len(tile)
    return {
        "phase": SEALED if next_key_base == total else SCAN,
        "global_max": maximum,
        "next_key_base": next_key_base,
    }


def emit_tile(
    state: dict[str, int | str | None], tile: list[int], key_base: int, total: int
) -> tuple[dict[str, int | str | None], list[int]]:
    assert tile
    assert key_base % 256 == 0
    assert len(tile) == min(256, total - key_base)
    if key_base == 0:
        assert state["phase"] == SEALED
    else:
        assert state["phase"] == EMIT
        assert state["next_key_base"] == key_base
    maximum = int(state["global_max"])
    centered = [center(value, maximum) for value in tile]
    next_key_base = key_base + len(tile)
    if next_key_base == total:
        next_state: dict[str, int | str | None] = {
            "phase": EMPTY,
            "global_max": None,
            "next_key_base": 0,
        }
    else:
        next_state = {
            "phase": EMIT,
            "global_max": maximum,
            "next_key_base": next_key_base,
        }
    return next_state, centered


def main() -> None:
    precenter = [10, 9] + [-20] * 254 + [100]
    tiles = [(0, precenter[:256]), (256, precenter[256:])]
    state: dict[str, int | str | None] = {
        "phase": EMPTY,
        "global_max": None,
        "next_key_base": 0,
    }
    trace: list[dict[str, int | str | None]] = []
    published: list[int] = []

    for key_base, tile in tiles:
        state = scan_tile(state, tile, key_base, len(precenter))
        trace.append(dict(state))
        assert not published

    assert state == {"phase": SEALED, "global_max": 100, "next_key_base": 257}
    for key_base, tile in tiles:
        state, centered = emit_tile(state, tile, key_base, len(precenter))
        published.extend(centered)
        trace.append(dict(state))

    expected = [-90, -91] + [-120] * 254 + [0]
    independently_centered: list[int] = []
    for _, tile in tiles:
        tile_max = max(tile)
        independently_centered.extend(center(value, tile_max) for value in tile)

    expected_hash = "e570bb9241979e9891868465ac910a87cd1e57f698503f649d57d5342c63ae3f"
    independent_hash = "d2a8d6ad99ed19442c449325598449fc5e1237d013a6918efbfd1c64ec8ceaf9"
    mismatch_count = sum(
        actual != local
        for actual, local in zip(published, independently_centered, strict=True)
    )

    assert [max(tile) for _, tile in tiles] == [10, 100]
    assert published == expected
    assert payload_sha256(published) == expected_hash
    assert payload_sha256(independently_centered) == independent_hash
    assert mismatch_count == 256
    assert state == {"phase": EMPTY, "global_max": None, "next_key_base": 0}

    max_row_stride = score_row_stride(MAX_CONTEXT)
    max_matrix_bytes = score_matrix_bytes(MAX_CONTEXT)
    assert max_row_stride == 65536
    assert max_matrix_bytes == 917504
    assert max_matrix_bytes > SRAM_MARGIN_BYTES

    max_descriptor_population = descriptor_population(MAX_CONTEXT)
    assert max_descriptor_population == {
        "root": 1,
        "score_and_dma": 4096,
        "compose": 172032,
        "total": 176129,
    }
    assert max_descriptor_population["total"] > NONZERO_COMPLETION_TAGS
    rolling_tags = simulate_rolling_tags(max_descriptor_population["total"])
    assert rolling_tags == {
        "capacity": 65535,
        "descriptor_count": 176129,
        "first_reuse_ordinal": 65535,
        "max_live_success_slots": 1,
        "reuse_count": 110594,
        "reuse_distance": 65535,
        "terminal_leased_tag": 45059,
        "terminal_live_slots": 1,
    }

    external_base = 0x0000_0001_0000_0000
    ranges = [
        (
            external_base + head * max_row_stride,
            external_base + (head + 1) * max_row_stride,
        )
        for head in range(HEADS)
    ]
    assert all(start % BEAT_BYTES == 0 for start, _ in ranges)
    assert all(ranges[index][1] == ranges[index + 1][0] for index in range(HEADS - 1))
    assert ranges[-1][1] - external_base == max_matrix_bytes
    assert ranges[-1][1] < 1 << 63

    length_257_stride = score_row_stride(257)
    assert length_257_stride == 528
    assert all((external_base + offset) % BEAT_BYTES == 0 for offset in (0, 512))
    assert 512 + BEAT_BYTES == length_257_stride

    descriptor_points = {
        kind: {str(length): descriptor_cycles(kind, length) for length in (1, 128, 256)}
        for kind in ("single", "max_scan", "center_emit")
    }
    assert descriptor_points == {
        "single": {"1": 3924, "128": 501078, "256": 1002150},
        "max_scan": {"1": 3919, "128": 500424, "256": 1000840},
        "center_emit": {"1": 3924, "128": 500952, "256": 1001896},
    }

    contexts = (1, 1024, MAX_CONTEXT)
    producer_cycles = {str(length): score_producer_cycles(length) for length in contexts}
    score_row_cycles = {
        str(length): score_producer_cycles(length) + compose_score_read_cycles(length)
        for length in contexts
    }
    steady_cycles = {str(length): score_row_cycles[str(length)] + 25 for length in contexts}
    cold_cycles = {str(length): steady_cycles[str(length)] + 9 for length in contexts}
    assert producer_cycles == {
        "1": 54978,
        "1024": 112204552,
        "32768": 3590545664,
    }
    assert score_row_cycles == {
        "1": 55020,
        "1024": 112209928,
        "32768": 3590717696,
    }
    assert steady_cycles == {
        "1": 55045,
        "1024": 112209953,
        "32768": 3590717721,
    }
    assert cold_cycles == {
        "1": 55054,
        "1024": 112209962,
        "32768": 3590717730,
    }

    producer_prefill_128 = sum(25 + score_producer_cycles(length) for length in range(1, 129))
    complete_prefill_128 = sum(
        25 + score_producer_cycles(length) + compose_score_read_cycles(length)
        for length in range(1, 129)
    )
    assert producer_prefill_128 == 452693568
    assert complete_prefill_128 == 452739264

    external_bytes = {str(length): complete_external_bytes(length) for length in contexts}
    assert external_bytes == {
        "1": 1584,
        "1024": 934288,
        "32768": 29888784,
    }
    prefill_external_bytes = sum(complete_external_bytes(length) for length in range(1, 129))
    assert prefill_external_bytes == 4314112

    sram_points = {
        kind: sram_descriptor_bytes(kind, 256)
        for kind in ("single", "max_scan", "center_emit")
    }
    assert sram_points == {
        "single": 92240,
        "max_scan": 88176,
        "center_emit": 88176,
    }

    print(
        json.dumps(
            {
                "address_space_and_lifetime": {
                    "address_space": "external_addr_bit_63_zero",
                    "allocation_bytes_at_32768": max_matrix_bytes,
                    "allocation_formula": "14*16*ceil(L/8)",
                    "base_alignment_bytes": BEAT_BYTES,
                    "compose_score_read_passes": COMPOSE_SCORE_READ_PASSES,
                    "lifetime": [
                        "ALLOCATED_PROVISIONAL",
                        "PUBLISHED_AFTER_FINAL_HEAD_SCORE_COMPLETION",
                        "READ_BY_MAX_SUM_VALUE",
                        "ROW_RELEASED_AFTER_SUCCESSFUL_VALUE_LAST",
                    ],
                    "row_stride_bytes_at_257": length_257_stride,
                    "row_stride_bytes_at_32768": max_row_stride,
                    "sram_margin_bytes": SRAM_MARGIN_BYTES,
                    "sram_residency_rejected": True,
                },
                "contract_id": "layer0_relative_rope_score_fusion_v1",
                "descriptor_capacity_at_32768": {
                    "population": max_descriptor_population,
                    "rolling_tags": rolling_tags,
                    "semantics": [
                        "IN_ORDER_ONE_ACTIVE_ISSUE",
                        "CONSUME_PREDECESSOR_SUCCESS_TO_EMPTY_AT_ISSUE",
                        "REQUIRE_COMPLETION_SLOT_EMPTY_BEFORE_REQUEST",
                        "TERMINAL_TAG_CONSUMED_BY_NEXT_CHAIN_OR_CLEARED_BY_QUIESCENT_RESET",
                    ],
                },
                "discriminator": {
                    "expected_sha256": expected_hash,
                    "independent_tile_sha256": independent_hash,
                    "mismatch_count": mismatch_count,
                    "state_trace": trace,
                    "status": "pass",
                    "test_id": "global_valid_key_max_257_unequal_tile_max",
                    "tile_maxima": [10, 100],
                    "tile_sizes": [256, 1],
                    "valid_keys": 257,
                },
                "resource_accounting_checks": {
                    "cold_score_row_transport_cycles": cold_cycles,
                    "complete_external_bytes_with_append": external_bytes,
                    "compose_score_read_cycles": {
                        str(length): compose_score_read_cycles(length) for length in contexts
                    },
                    "descriptor_cycles": descriptor_points,
                    "external_score_rows_at_32768": {
                        "downstream_read_bytes": COMPOSE_SCORE_READ_PASSES * max_matrix_bytes,
                        "total_transport_bytes": (1 + COMPOSE_SCORE_READ_PASSES)
                        * max_matrix_bytes,
                        "write_bytes": max_matrix_bytes,
                    },
                    "prefill_128_cold_cycles": complete_prefill_128 + 9,
                    "prefill_128_external_bytes_after_setup": prefill_external_bytes,
                    "producer_only_cycles": producer_cycles,
                    "producer_only_prefill_128_cycles_after_setup": producer_prefill_128,
                    "score_row_transport_cycles": score_row_cycles,
                    "sram_bytes_per_head_at_256": sram_points,
                    "steady_score_row_transport_cycles_with_append": steady_cycles,
                },
                "status": "pass",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
