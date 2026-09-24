# Three-prompt RTL chat oracle regression

Compile and run the focused runner tests:

```bash
make prompt-suite-rtl-oracle-regression-test
```

Run the bounded computer-local suite from the repository root:

```bash
make prompt-suite-rtl-oracle-regression \
  ACE2_CHAT_PROMPT_SUITE_OUTPUT=reports/verification/<fresh-suite-namespace>
```

The output path must not exist. Each of the three distinct prompts receives its
own fresh RTL-attempt and independent-oracle namespace; sealed attempts are not
reused or mutated. `PASS` requires exactly four generated tokens per prompt
(12 total), continuous K/V growth through all 24 layers and all nine decode
transitions, nonzero complete oracle comparisons of K/V append bytes, full
cache prefixes, quantized layer outputs, and full-vocabulary logits, zero byte
or selected-token mismatches, and
`software_transformer_or_logits_fallback=false`.

Recorded timings are computer-local host/orchestration and Icarus simulation
measurements only. The suite performs no FPGA, synthesis, STA, PPA,
hardware-emulation, deployment, bitstream, or silicon work.
