# ACE-2 down-projection residual-fusion verification plan

This plan is limited to the active `verification` stage and candidate
`shared_down_projection_residual_fusion_v1` at RTL hash
`76ef0dda2e646558ecb3e2047d3ec00d69f416cbe0a880b3402876614a4a7a28`.
It does not run or claim PPA, prototype, benchmark, signoff, tapeout, or silicon evidence.

## Acceptance gates

- Independent oracle includes standalone and all-lane arithmetic checks plus frozen quality constraints.
- Coverage stress includes reset, boundary, stalls/backpressure, illegal/error, randomized,
  single-clock CDC disposition, X/Z, saturation/overflow, formal, and representative datasets.
- Reproducibility requires retained successful commands and non-contradictory hash-bound artifacts.
  The exact-one discriminator execution contract remains binding.

## Authoritative disposition

Technical status: `bounded_no_go`; eight frozen quality conditions fail.
Execution-integrity status: `integrity_no_go`; two successful executions violate exact-one.
An independent read-only L2 audit confirms both dispositions. Stage closing: `false`.
A third discriminator execution is prohibited; the Manager owns rollback or hold routing.
