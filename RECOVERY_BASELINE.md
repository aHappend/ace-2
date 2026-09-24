# ACE-1 Recovery Baseline

This file records historical evidence, not reproduced ACE-2 results.

## Confirmed historical structure

The inaccessible project used seven named SystemVerilog sources:

1. `argus_npu_pkg.sv`
2. `argus_npu_csr.sv`
3. `argus_npu_command.sv`
4. `argus_npu_dma.sv`
5. `argus_npu_scratchpad.sv`
6. `argus_npu_compute.sv`
7. `argus_npu_top.sv`

The architecture included CSR, command scheduling, DMA, an eight-bank logical
scratchpad, compute, and top-level integration. Historical reports describe
continuous support through MLP gate projection, with MLP up projection as the
first unsupported operator.

## Hard historical claims

Surviving reports record:

- 45 verification scenarios with zero reported failures;
- approximately 113.81 MHz SKY130 pre-layout logic timing;
- approximately 2.318 mm^2 non-SRAM standard-cell area after a cross-operator
  sharing experiment;
- SRAM treated as a logical macro/blackbox, not included in that area.

These values have not been reproduced on this machine. The operator recalls a
later result near 1.8 mm^2, but no original PPA report or source hash has been
recovered, so 1.8 mm^2 is not an accepted baseline.

## Forensic recovery status

The forensic workspace is:

`session files/ace1-recovery/`

It contains a 280-line top-level transcript fragment and a truncated historical
diff. All seven final RTL files are classified `UNRECOVERED`; no byte-identical,
buildable source tree survived.

ACE-2 must therefore treat the prior architecture and measurements as design
leads. Every RTL source, test, reference model, constraint, PPA report, and
benchmark result must be recreated and validated under new hashes.

