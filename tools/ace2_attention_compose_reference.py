#!/usr/bin/env python3
"""Independent fixed-point reference for tiled long-context attention."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from tools.ace2_softmax_reference import exp_weight_q15, round_div_even, to_sint
    from tools.ace2_attention_value_reference import round_shift_even_signed
except ModuleNotFoundError:
    from ace2_softmax_reference import exp_weight_q15, round_div_even, to_sint
    from ace2_attention_value_reference import round_shift_even_signed


HEAD_DIM = 64
CONTEXT_MAX = 32768
PROB_FRAC = 15


@dataclass(frozen=True)
class AttentionComposeCase:
    name: str
    scores_q6_9: list[int]
    values: list[list[int]]


@dataclass(frozen=True)
class AttentionComposeResult:
    max_score_q6_9: int
    exp_sum_q15: int
    probabilities_q15: list[int]
    accumulators: list[int]
    outputs: list[int]
    saturation_seen: bool


@dataclass(frozen=True)
class AttentionComposePhaseResult:
    outputs: list[int] | None
    saturation_seen: bool
    max_score_q6_9: int
    exp_sum_q15: int
    max_count: int
    sum_count: int
    value_count: int


class AttentionComposePhaseReplay:
    """Command-by-command model of ace2_attention_compose_core state."""

    def __init__(self) -> None:
        self.protocol = "empty"
        self.max_score_q6_9 = -32768
        self.exp_sum_q15 = 0
        self.max_count = 0
        self.sum_count = 0
        self.value_count = 0
        self.accumulators = [0 for _ in range(HEAD_DIM)]

    def apply(
        self,
        phase: int,
        scores_q6_9: list[int],
        values: list[list[int]] | None = None,
    ) -> AttentionComposePhaseResult:
        count = len(scores_q6_9)
        if not 1 <= count <= 8:
            raise ValueError("attention-compose phase must carry 1..8 scores")
        scores = [to_sint(value, 16) for value in scores_q6_9]
        rows = None
        if phase >= 4:
            if values is None or len(values) != count:
                raise ValueError("value phase requires one V row per score")
            if any(len(row) != HEAD_DIM for row in values):
                raise ValueError("each V row must contain HEAD_DIM elements")
            rows = [[to_sint(value, 8) for value in row] for row in values]

        outputs = None
        saturation = False
        if phase == 0:
            self.protocol = "max"
            self.max_score_q6_9 = max(scores)
            self.max_count = count
            self.sum_count = 0
            self.value_count = 0
        elif phase == 1:
            if self.protocol != "max" or self.max_count + count > CONTEXT_MAX:
                raise ValueError("MAX_MORE phase is not protocol-legal")
            self.max_score_q6_9 = max(self.max_score_q6_9, max(scores))
            self.max_count += count
        elif phase == 2:
            if self.protocol != "max":
                raise ValueError("SUM_FIRST phase is not protocol-legal")
            self.protocol = "sum"
            self.exp_sum_q15 = sum(
                exp_weight_q15(score - self.max_score_q6_9) for score in scores
            )
            self.sum_count = count
        elif phase == 3:
            if self.protocol != "sum" or self.sum_count + count > self.max_count:
                raise ValueError("SUM_MORE phase is not protocol-legal")
            self.exp_sum_q15 += sum(
                exp_weight_q15(score - self.max_score_q6_9) for score in scores
            )
            self.sum_count += count
        elif phase in (4, 5, 6):
            if phase == 4:
                if self.protocol != "sum" or self.sum_count != self.max_count:
                    raise ValueError("VALUE_FIRST phase is not protocol-legal")
                self.protocol = "value"
                self.value_count = count
                self.accumulators = [0 for _ in range(HEAD_DIM)]
            elif phase == 5:
                if (
                    self.protocol != "value"
                    or self.value_count + count >= self.max_count
                ):
                    raise ValueError("VALUE_MORE phase is not protocol-legal")
                self.value_count += count
            else:
                if (
                    self.protocol != "value"
                    or self.value_count + count != self.max_count
                ):
                    raise ValueError("VALUE_LAST phase is not protocol-legal")
                self.value_count += count

            if self.exp_sum_q15 == 0:
                raise ValueError("value phase has a zero exponential sum")
            assert rows is not None
            probabilities = [
                min(
                    0xFFFF,
                    round_div_even(
                        exp_weight_q15(score - self.max_score_q6_9) << PROB_FRAC,
                        self.exp_sum_q15,
                    ),
                )
                for score in scores
            ]
            for probability, row in zip(probabilities, rows, strict=True):
                for lane, value in enumerate(row):
                    self.accumulators[lane] += probability * value

            if phase == 6:
                rounded = [
                    round_shift_even_signed(value, PROB_FRAC)
                    for value in self.accumulators
                ]
                outputs = [max(-128, min(127, value)) for value in rounded]
                saturation = any(
                    output != value
                    for output, value in zip(outputs, rounded, strict=True)
                )
        else:
            raise ValueError("attention-compose phase must be 0..6")

        return AttentionComposePhaseResult(
            outputs=outputs,
            saturation_seen=saturation,
            max_score_q6_9=self.max_score_q6_9,
            exp_sum_q15=self.exp_sum_q15,
            max_count=self.max_count,
            sum_count=self.sum_count,
            value_count=self.value_count,
        )


def reference_attention_compose(
    case: AttentionComposeCase,
) -> AttentionComposeResult:
    context_count = len(case.scores_q6_9)
    if not 1 <= context_count <= CONTEXT_MAX:
        raise ValueError("composed attention context must be 1..CONTEXT_MAX")
    if len(case.values) != context_count:
        raise ValueError("one V row is required per score")
    if any(len(row) != HEAD_DIM for row in case.values):
        raise ValueError("each V row must contain HEAD_DIM elements")

    scores = [to_sint(value, 16) for value in case.scores_q6_9]
    values = [[to_sint(value, 8) for value in row] for row in case.values]
    max_score = max(scores)
    weights = [exp_weight_q15(score - max_score) for score in scores]
    exp_sum = sum(weights)
    probabilities = [
        round_div_even(weight << PROB_FRAC, exp_sum) for weight in weights
    ]
    accumulators = [
        sum(
            probabilities[token] * values[token][lane]
            for token in range(context_count)
        )
        for lane in range(HEAD_DIM)
    ]
    rounded = [
        round_shift_even_signed(value, PROB_FRAC) for value in accumulators
    ]
    outputs = [max(-128, min(127, value)) for value in rounded]
    return AttentionComposeResult(
        max_score_q6_9=max_score,
        exp_sum_q15=exp_sum,
        probabilities_q15=probabilities,
        accumulators=accumulators,
        outputs=outputs,
        saturation_seen=any(
            output != value for output, value in zip(outputs, rounded, strict=True)
        ),
    )
