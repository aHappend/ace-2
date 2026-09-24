# ACE-2 Stage-1 RTL-backed chat milestone

## Result

Stage-1 attempt 0002 completed with status `PASS` for the arbitrary UTF-8
prompt `Grüße,世界`.

| Field | Result |
| --- | --- |
| Generated token IDs | `15663`, `25055` |
| Decoded text | ` aside crystal` |
| Prompt tokens | 34 |
| RTL execution positions | 35 |
| RTL layer executions | 840 |
| Persistent K/V continuity | PASS |
| Independent integer mismatches | 0 |
| Readability policy | PASS |
| Total wall time | 52,397.777641 seconds |

The two full-vocabulary RTL-backed head selections agreed with the host
integer reference. Token 0 was fed back through position 34 to produce token
1, and RTL-published K/V rows replaced the corresponding host cache rows
before downstream positions consumed them.

## Model boundary

This milestone is not a separately trained W4A8 checkpoint. It binds:

- `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775` as the BF16 parent snapshot;
- the ACE-2 LoRA v4 checkpoint-176 adapter with SHA256
  `c9476d6ccc687a4d68de1951eb889b689daaf9c5d3822b4c5acfc26a1cd38a28`;
- tokenizer/config files from `Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision
  `db09cd27ead7fee40cdee309693cf83601b9c899`;
- ACE-2 W4A8 integer quantization, rounding, saturation, attention, residual,
  K/V-cache, and full-vocabulary LM-head semantics.

The host merges the BF16 parent weights and trained LoRA adapter, derives the
expected W4A8 integer boundaries before RTL execution, and uses RTL outputs
only as comparison observations. The run records
`software_transformer_or_logits_fallback=false`.

## Reproduction

From the repository root, with the bound local model and adapter artifacts:

```bash
python3 tools/ace2_v73_stage1_chat_attempt.py \
  --output reports/ace2-stage1-chat-attempt-0001 \
  --max-new-tokens 2 \
  --timeout-seconds 64800 \
  --prompt 'Grüße,世界'
```

The command requires a fresh output directory. It applies the official chat
template, performs prompt prefill and autoregressive decode through the
Icarus-backed W4A8 path, preserves per-position evidence, and fails closed on
software fallback, unreadable output, stale output, incomplete K/V continuity,
or any integer mismatch.

## Evidence boundary

The raw attempt is intentionally not committed because it is approximately
1 GB and includes generated simulator vectors and a 170 MB LM-head hex image.
The local immutable evidence is:

- `reports/ace2-stage1-chat-attempt-0002/attempt-result.json`
  SHA256 `5ec4b355a2fd3ad69c45ecc6accedd26da1528a383cde13f8b558e3f4f13e8d8`;
- `reports/ace2-stage1-chat-attempt-0002/worker-result.json`
  SHA256 `c6891d42e24ded0935ac6be7306e0e47bb38aa676c824d79ba6e07e7693709e4`;
- `reports/ace2-stage1-chat-attempt-0002/independent-reference-comparison.json`
  SHA256 `40803354b455534bc099913a8531a8552621645d1846c4921733b49ca6d90e9e`.

This is local RTL-backed functional evidence. It is not synthesis, PPA,
bitstream, PCIe/XRT, Alveo U280, or deployed-hardware evidence.
