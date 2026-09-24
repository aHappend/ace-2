#!/usr/bin/env python3
"""Audit the active ACE-2 productization specification without downstream work."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

from ace2_exact_once_runtime_supervisor import measure_source_tree
from audit_rtl_stage_contract import audit as audit_shell_contract


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SPEC = ROOT / "design/SPEC.md"
BENCHMARK = ROOT / "design/BENCHMARK_INTERFACE.json"
PIPELINE = ROOT / "research/PIPELINE_STATE.json"
PIPELINE_CHECKSUM = ROOT / "research/PIPELINE_STATE.sha256"
GROUND_TRUTH = ROOT / "research/GROUND_TRUTH.md"
LATEST = ROOT / "latest.json"
SHELL_AUDIT = ROOT / "tools/audit_rtl_stage_contract.py"
SHELL = ROOT / "rtl/ace2_shell.sv"
MAKEFILE = ROOT / "Makefile"
U280_INVENTORY = ROOT / "research/U280_HOST_INVENTORY.json"
V4_ATTESTATION = (
    ROOT
    / "research/raw/specification/"
    "qwen25-05b-instruct-bf16-lora-product-v4-cross-split-blinded-attestation.json"
)
V4_RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v4"
V4_DIAGNOSTIC_BUILDER = ROOT / "tools/build_qwen25_lora_v4_aggregate_failure_diagnostic.py"
V4_DIAGNOSTIC = ROOT / "research/diagnostics/qwen25_lora_v4_aggregate_failure_diagnostic.json"
V4_DIAGNOSTIC_CHECKSUM = V4_DIAGNOSTIC.with_suffix(V4_DIAGNOSTIC.suffix + ".sha256")
V5_RECOMMENDATION_L2 = (
    ROOT
    / "research/raw/specification/"
    "qwen25-lora-v4-aggregate-diagnostic-v5-recommendation-l2-review.json"
)
V5_RECOMMENDATION_L2_CHECKSUM = V5_RECOMMENDATION_L2.with_suffix(
    V5_RECOMMENDATION_L2.suffix + ".sha256"
)
V5_EXECUTION_CONTRACT = (
    ROOT / "design/QWEN25_05B_INSTRUCT_BF16_LORA_PRODUCT_V5_EXECUTION_PACKAGE_CONTRACT.json"
)
V5_OFFLINE_ROOT = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5"
V5_EXECUTION_MANIFEST = V5_OFFLINE_ROOT / "v5-execution-package-manifest.json"
V5_EXECUTION_SELF_TEST = V5_OFFLINE_ROOT / "v5-execution-package-self-test.json"
V5_EXECUTION_L2_DECISION = V5_OFFLINE_ROOT / "execution-package-l2-decision.json"
V5_EXECUTION_L2_ACCEPTANCE = V5_OFFLINE_ROOT / "execution-package-l2-acceptance.json"
V5_RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5"
V5_ATTEMPT_AUTHORITY = (
    ROOT
    / "research/raw/specification/"
    "qwen25-05b-instruct-bf16-lora-product-v5-attempt-operator-authority.json"
)
V5_LAUNCH_READINESS = V5_OFFLINE_ROOT / "v5-launch-readiness.json"
V5_ATTEMPT_MARKER = V5_RUN_ROOT / "attempt-0001/ATTEMPT_CONSUMPTION_MARKER.json"
V5_EXIT_STATUS = V5_RUN_ROOT / "attempt-0001-exit-status.json"
V5_ATTEMPT_CONSOLE = V5_OFFLINE_ROOT / "v5-attempt-0001-console.log"
V5_CONTAINMENT_INPUT = (
    ROOT
    / "research/raw/specification/"
    "qwen25-lora-v5-unauthorized-attempt-containment-input-v2-20260808T034640Z.json"
)
V5_CONTAINMENT_L2 = (
    ROOT
    / "research/raw/specification/"
    "qwen25-lora-v5-unauthorized-attempt-l2-review-v2-20260808.json"
)
V5_INVALID_CONTAINMENT_INPUT = (
    ROOT
    / "research/raw/specification/"
    "qwen25-lora-v5-unauthorized-attempt-containment-input-20260808T033922Z.json"
)
V5_INVALID_CONTAINMENT_L2 = (
    ROOT
    / "research/raw/specification/"
    "qwen25-lora-v5-unauthorized-attempt-l2-review-20260808.json"
)
V7_ATTEMPT_AUTHORITY = (
    ROOT
    / "research/raw/specification/"
    "qwen25-05b-instruct-bf16-full-finetune-product-v7-attempt-operator-authority.json"
)
V7_TERMINAL_AUDIT = ROOT / "build/v7-b200-full-lifecycle/terminal-evidence-audit.json"
V7_FRESH_REVIEW = (
    ROOT
    / "research/raw/specification/"
    "v7-full-lifecycle-terminal-fresh-review-20260808.json"
)
V8_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_PRODUCT_V8_EXECUTION_PACKAGE_CONTRACT.json"
V8_FREEZE = ROOT / "research/training/qwen2.5-0.5b-instruct-bf16-full-finetune-product-v8/freeze_manifest.json"
V8_MANIFEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/v8-execution-package-manifest.json"
V8_SELF_TEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/v8-execution-package-self-test.json"
V8_REVIEW_REQUEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/execution-package-l2-review-request.json"
V8_REVIEW_DECISION = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/execution-package-l2-decision.json"
V8_REVIEW_ACCEPTANCE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8/execution-package-l2-acceptance.json"
V8_RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8"
V8_AUTHORITY = ROOT / "research/raw/specification/qwen25-05b-instruct-bf16-full-finetune-product-v8-attempt-operator-authority.json"
V8_LIFECYCLE_ROOT = ROOT / "build/v8-b200-full-lifecycle"
V8_INVENTORY = V8_LIFECYCLE_ROOT / "backend-inventory-final.json"
V8_INTENT = V8_LIFECYCLE_ROOT / "backend-submission-intent.json"
V8_RESULT = V8_LIFECYCLE_ROOT / "backend-submission-result.json"
V8_POST_INVENTORY = V8_LIFECYCLE_ROOT / "backend-postsubmit-inventory.json"
V8_TERMINAL_AUDIT = V8_LIFECYCLE_ROOT / "terminal-no-execution-audit-v2.json"
V8_FRESH_REVIEW = (
    ROOT
    / "research/raw/specification/"
    "v8-full-lifecycle-terminal-fresh-review-20260808.json"
)
S1_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S1_TRANSPORT_PACKAGE_CONTRACT.json"
S1_PACKAGE_AUDIT = ROOT / "build/bf16-successor-s1-transport/package-audit.json"
S1_REPAIR_AUDIT = ROOT / "build/bf16-successor-s1-transport-repair/repair-audit.json"
S1_FRESH_REVIEW = (
    ROOT
    / "research/raw/specification/"
    "bf16-successor-s1-transport-fresh-review-20260808.json"
)
S1_AUTHORITY = (
    ROOT
    / "research/raw/specification/"
    "qwen25-bf16-successor-s1-transport-attempt-operator-authority.json"
)
S1_INTENT = ROOT / "build/bf16-successor-s1-transport/backend-submission-intent.json"
S1_RESULT = ROOT / "build/bf16-successor-s1-transport/backend-submission-result.json"
S1_RAW = ROOT / "build/bf16-successor-s1-transport/backend-submission.raw.txt"
S1_TERMINAL_AUDIT = (
    ROOT / "build/bf16-successor-s1-transport/backend-terminal-audit-20260808T111251Z.json"
)
S1_TERMINAL_REVIEW = (
    ROOT
    / "research/raw/specification/"
    "bf16-successor-s1-transport-terminal-backend-fresh-review-20260808.json"
)
S1_TERMINAL_STDOUT = (
    ROOT
    / "build/bf16-successor-s1-transport/backend-terminal-download-20260808T111251Z/"
    "logs/ace2-bf16-successor-s1-transport-20260808-preauthority/"
    "ace2-bf16-successor-s1-transport-gate/stdout.txt"
)
S1_RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-successor-s1"
S1_PACKAGE_ID = "qwen2.5-0.5b-instruct-ace2-bf16-successor-s1-transport-package-v1"
S1_PACKAGE_IDENTITY = "3b8d5c3b337701bdc318782d01abebd24d7ad5ee050ea76d42bfb6049bdaf7fb"
S3_CONTRACT = (
    ROOT
    / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S3_EXECUTION_PACKAGE_CONTRACT.json"
)
S3_PACKAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s3/package-audit.json"
S3_MANIFEST = (
    ROOT
    / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3/"
    "s3-execution-package-manifest.json"
)
S3_SELF_TEST = (
    ROOT
    / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3/"
    "s3-execution-package-self-test.json"
)
S3_L2_ACCEPTANCE = (
    ROOT
    / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3/"
    "execution-package-l2-acceptance.json"
)
S3_AUTHORITY = (
    ROOT
    / "research/raw/specification/"
    "qwen25-bf16-full-finetune-successor-s3-attempt-operator-authority.json"
)
S3_SUBMISSION_INTENT = ROOT / "build/bf16-full-finetune-successor-s3/backend-submission-intent.json"
S3_SUBMISSION_RESULT = ROOT / "build/bf16-full-finetune-successor-s3/backend-submission-result.json"
S3_SUBMISSION_RAW = ROOT / "build/bf16-full-finetune-successor-s3/backend-submission.raw.txt"
S3_RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3"
S3_PACKAGE_ID = "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s3-package-v1"
S3_PACKAGE_IDENTITY = "5da5fd12747430377beaff78e19f2ed9aa85f11e3862ee248f8764a8e84a163b"
S3_EXECUTION_TREE = "22deb788ed51f524bdd04e21e8db08618ebefd93c64f13a4760bba21ff4aae10"
S3_TERMINAL_AUDIT = ROOT / "build/bf16-full-finetune-successor-s3/backend-terminal-audit-20260808T122720Z.json"
S3_TERMINAL_REVIEW = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s3-terminal-fresh-review-final-20260808.json"
S4_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S4_EXECUTION_PACKAGE_CONTRACT.json"
S4_PACKAGE_AUDIT = ROOT / "build/bf16-full-finetune-successor-s4/package-audit.json"
S4_MANIFEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4/s4-execution-package-manifest.json"
S4_SELF_TEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4/s4-execution-package-self-test.json"
S4_L2_ACCEPTANCE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4/execution-package-l2-acceptance.json"
S4_AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s4-attempt-operator-authority.json"
S4_SUBMISSION_ROOT = ROOT / "build/bf16-full-finetune-successor-s4"
S4_RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4"
S4_TERMINAL_AUDIT = S4_SUBMISSION_ROOT / "backend-terminal-audit-v2-20260808T132245Z.json"
S4_TERMINAL_REVIEW = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s4-terminal-fresh-review-final-20260808.json"
S4_PACKAGE_ID = "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s4-package-v1"
S4_PACKAGE_IDENTITY = "8e2d4fad25dd4a102911b2e85e69147578a60727d10bf81d3d472b67c357c016"
S4_EXECUTION_TREE = "4d67d8d444681a11d3e1060b53da65ae67d2557aa9f780fe8e87106c7473de43"
S6_CONTRACT = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_FULL_FINETUNE_SUCCESSOR_S6_EXECUTION_PACKAGE_CONTRACT.json"
S6_MANIFEST = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s6/s6-execution-package-manifest.json"
S6_L2_ACCEPTANCE = ROOT / "build/offline-preflight/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s6/execution-package-l2-acceptance.json"
S6_AUTHORITY = ROOT / "research/raw/specification/qwen25-bf16-full-finetune-successor-s6-attempt-operator-authority.json"
S6_SUBMISSION_ROOT = ROOT / "build/bf16-full-finetune-successor-s6"
S6_RUN_ROOT = ROOT / "build/qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s6"
S6_CONTRACT_SHA256 = "f5604df165628155a34578aac82fd341c7ab204828c51bb11e0c10eaf45479ea"
S6_EXECUTION_TREE = "14318b0909634c6c5f72e6c2075f276d7d8a5ba35ab1979e5f3048f76533c964"
S6_SUBMISSION_INTENT = S6_SUBMISSION_ROOT / "backend-submission-intent.json"
S6_SUBMISSION_RESULT = S6_SUBMISSION_ROOT / "backend-submission-result.json"
S6_TERMINAL_AUDIT = (
    S6_SUBMISSION_ROOT / "backend-terminal-audit-20260809T062750Z.json"
)
S6_AUTHORITY_SHA256 = "3bc488540446776821702f709e436daca8ab19e45e1ab30edd1bf2e10a157aed"
S6_INTENT_SHA256 = "e30fe7737a7924561244f703b12721d3bf2fcde4df1d326e052f22840acf88ec"
S6_RESULT_SHA256 = "3421105126df3332b20228cb299f00b77d7723cb9d5d131475b3461d893681d6"
S6_TERMINAL_AUDIT_SHA256 = "f14e0a46e6725a39ab2d1c8c6578281c5a4fd7b4c8cfab18c65d00b563c83093"
S6_LATEST_SHA256 = "ac0a0560b586fc371306d3a1500f9a69d66fece051e71748a22118434934df5c"
S7_SPECIFICATION = (
    ROOT
    / "design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S7_DORA_SFT_DPO_SPECIFICATION_PACKAGE.json"
)
S7_MANIFEST = ROOT / "design/QWEN25_05B_INSTRUCT_BF16_SUCCESSOR_S7_PACKAGE_MANIFEST.json"
S7_FRESH_REVIEW_BINDING = (
    ROOT
    / "research/raw/specification/"
    "qwen25-bf16-successor-s7-specification-fresh-review-binding-20260809.json"
)
S7_SPECIFICATION_SHA256 = "c144781eda57946960453ac671501b2facb2a0dac8657b85efdfb5e65a86457a"
S7_MANIFEST_SHA256 = "40965c7bab3d381f1dabbbb102d6e40e78796f73503780597395c3274a4ce22d"
S7_REVIEW_BINDING_SHA256 = "6c2102886465e7dbc9ee012f205ee14ca7115ef3f70ae193c6619f3b425ccb62"
S7_REVIEW_HANDOFF_SHA256 = "5fcc20758d18079866c18d70a0d2e6dbd4f14d7da3137b4c79d78894d8669a66"
S7_EXECUTION_ROOT = ROOT / "research/execution/qwen25_bf16_successor_s7_20260809"
S7_EXECUTION_AUTHORITY = S7_EXECUTION_ROOT / "authority_record.txt"
S7_EXECUTION_MANIFEST = S7_EXECUTION_ROOT / "package-manifest.json"
S7_EXECUTION_AUDIT = S7_EXECUTION_ROOT / "package-audit.json"
S7_EXECUTION_FRESH_REVIEW = (
    S7_EXECUTION_ROOT / "fresh-review-preseal-compatibility-repair-acceptance.json"
)
S7_EXECUTION_RUNNER = S7_EXECUTION_ROOT / "run_future.py"
S7_NODE1_LAUNCH = S7_EXECUTION_ROOT / "node1_launch.py"
S7_ACCEPTED_BASE_ROOT = (
    S7_EXECUTION_ROOT
    / "history"
    / "accepted-d1c6be0d70c446d86574292aabd136ad4109699d626f15fcea3f69406ad5f053"
)
S7_ACCEPTED_BASE_MANIFEST = S7_ACCEPTED_BASE_ROOT / "package-manifest.json"
S7_ACCEPTED_BASE_RUNNER = S7_ACCEPTED_BASE_ROOT / "run_future.py"
S7_DATA_MANIFEST = S7_EXECUTION_ROOT / "materialized/data-manifest.json"
S7_CUSTODIAN_ATTESTATION = (
    ROOT
    / "research/raw/specification/"
    "qwen25-bf16-successor-s7-s6-disjointness-custodian-attestation.json"
)
S7_CUSTODIAN_REVIEW_BINDING = (
    ROOT
    / "research/raw/specification/"
    "qwen25-bf16-successor-s7-s6-disjointness-custodian-attestation-"
    "fresh-review-binding-20260809.json"
)
S7_CONTROL_PLANE_EVIDENCE = (
    ROOT
    / "research/raw/specification/"
    "s7-owned-reservation-control-plane-binding-refresh-20260810T031042Z.json"
)
S7_CONTROL_PLANE_REVIEW_BINDING = (
    ROOT
    / "research/raw/specification/"
    "s7-owned-reservation-control-plane-binding-refresh-fresh-l2-review-binding-20260810.json"
)
S7_V4F_TERMINAL_BINDING = (
    ROOT
    / "research/raw/specification/"
    "s7-v4f-terminal-before-spawn-fresh-l2-binding-20260810.json"
)
S7_V4F_BUNDLE_ROOT = ROOT / "build/s7-replacement-reservation-execution-v4f"
S7_V4F_LAUNCHER = S7_V4F_BUNDLE_ROOT / "launch_once_v4f.py"
S7_V4F_LAUNCHER_MANIFEST = S7_V4F_BUNDLE_ROOT / "launch-capsule-SHA256SUMS"
S7_V4F_EXECUTOR = S7_V4F_BUNDLE_ROOT / "execute_once_v4f.py"
S7_V4F_EXECUTION_CONTRACT = S7_V4F_BUNDLE_ROOT / "execution-contract.json"
S7_V4F_BUNDLE_MANIFEST = S7_V4F_BUNDLE_ROOT / "bundle-manifest.json"
S7_V4F_STATE_ROOT = Path(
    "/home/argustest/.local/state/ace2/s7-replacement-reservation-execution-v4f"
)
S7_V4F_AUTHORIZATION_TARGET = (
    S7_V4F_STATE_ROOT
    / "authorization-mgr-s7-reservation-submit-v4f-20260810t220000z-a8e42d6157bc.json"
)
S7_V4F_ATTEMPT_CONSUMED = (
    S7_V4F_STATE_ROOT
    / ".mgr-s7-reservation-submit-v4f-20260810t220000z-a8e42d6157bc."
    "471b8d96d501fa708be86e2d7b1fef783c6f5c1a390357427b8144cc28926e9d."
    "attempt-consumed"
)
S7_V4F_ATTEMPT_ROOT = (
    S7_V4F_STATE_ROOT
    / "mgr-s7-reservation-submit-v4f-20260810t220000z-a8e42d6157bc"
)
S7_V4F_EXECUTION_ATTEMPT = S7_V4F_ATTEMPT_ROOT / "execution-attempt.json"
S7_V4F_EXECUTION_RESULT = S7_V4F_ATTEMPT_ROOT / "execution-result.json"
S7_V4F_REVIEW_HANDOFF = Path(
    "/home/argustest/.argus-skill-ace2/projects/s-c8ae985b/handoffs/"
    "submit-fresh-owned-s7-reservation-once-v4f/round-0001.json"
)
DS32_CSR_ADDRESS_CONTRACT = ROOT / "design/DS32_MODEL_IDENTITY_CSR_ADDRESS_CONTRACT.json"
S7_ATTEMPT_ROOT = ROOT / "build/bf16-successor-s7"
S7_ATTEMPT_MARKER = S7_ATTEMPT_ROOT / "ATTEMPT_CONSUMPTION_MARKER.json"
S7_ATTEMPT_START_SEAL = S7_ATTEMPT_ROOT / "ATTEMPT_RUN_START_SEAL.json"
S7_GPU_PREFLIGHT = S7_ATTEMPT_ROOT / "GPU_PRESEAL_PREFLIGHT.json"
S7_SUBMISSION_INTENT = S7_ATTEMPT_ROOT / "backend-submission-intent.json"
S7_CANDIDATE = S7_ATTEMPT_ROOT / "candidate"
S7_PROBE_RESULTS = S7_ATTEMPT_ROOT / "probe-results.json"
S7_EXECUTION_AUTHORITY_SHA256 = "85d3df54acfd96a15ba71cc038d7bfc515110d33e7a971613b9c61de820e48df"
S7_EXECUTION_MANIFEST_SHA256 = "9596d730dc4532929bb0c452b937ff7db397d1299929290d8d61c4af0e21b89d"
S7_EXECUTION_AUDIT_SHA256 = "e945ee79edaa615acf6d68d66c3c2b4f07a865290f81b866b001c567f3cd1867"
S7_EXECUTION_FRESH_REVIEW_SHA256 = "bf21da491098049a92a83ce1776845d4ed1e86c0e45449d93a66ed183ee367a0"
S7_EXECUTION_RUNNER_SHA256 = "99424d35222aebc2105ace6f2efd954cbc4ad47d4c533833be87515fb70213bd"
S7_NODE1_LAUNCH_SHA256 = "113047543fa8cf60a5949135486662cde73b6fb8ccfe9056a6328375df6d8a8f"
S7_ACCEPTED_BASE_MANIFEST_SHA256 = "d1c6be0d70c446d86574292aabd136ad4109699d626f15fcea3f69406ad5f053"
S7_ACCEPTED_BASE_RUNNER_SHA256 = "eef24dfb8c6e47d075b929f2a35ce24f1e81950e70955fece47c425f21135848"
S7_DATA_MANIFEST_SHA256 = "01e13e4bfb0549df174ba7ef248631b7338e0fe58a0453a410c3be1ca55827a2"
S7_CUSTODIAN_ATTESTATION_SHA256 = "e7739e97dc2f4f22b62da5807d03bdb67ac56ba1c0327b53962dc1b3b1741025"
S7_CUSTODIAN_REVIEW_BINDING_SHA256 = "b9d125a278036ea903af4a00af698ce3366595183a683e70e408fe348cc9e559"
S7_CUSTODIAN_REVIEW_HANDOFF_SHA256 = "ca75faf3e7586880e10d44ab1aa35337952c79d1321563d76b6e3e5b899d9f3e"
S7_CONTROL_PLANE_EVIDENCE_SHA256 = "559a9b573dcd65203cd3c72a9a4b4f4b6f2b5cf7cc63493c7d98515090a8cc47"
S7_CONTROL_PLANE_REVIEW_BINDING_SHA256 = "7c8359d7a2a16adad77ba7cbee774ed52e4510f52f74b49ea6c5f8a19011053e"
S7_CONTROL_PLANE_REVIEW_HANDOFF_SHA256 = "be9bd3d357654a47765c3b1bf936b987a39b44f7c663e6a036224fa050f5a9b7"
S7_V4F_TERMINAL_BINDING_SHA256 = "33a3db795e66793569b75a782a07b831941d9e5dacbe21a9a63614be112fb864"
S7_V4F_AUTHORIZATION_SHA256 = "d099d78bd657c941675bfba4d16b848df0ff3b5593e3e5fa9f9b32982aa70189"
S7_V4F_ATTEMPT_CONSUMED_SHA256 = "4938f323296bf5b8a5157287bf08c032dc9126ca6abb6f49e0f16a235e9eff8e"
S7_V4F_EXECUTION_ATTEMPT_SHA256 = "9ae9465679ffa56257786017bfb0010f58da3da4e62690656c96732ea178de9a"
S7_V4F_EXECUTION_RESULT_SHA256 = "f52626887f31d8ed5ffaeab4824217d2c63d77f73a040bf16d070a920fa08699"
S7_V4F_REVIEW_HANDOFF_SHA256 = "38c56f2433441ab1c4e0db5e16f636c9462ee98d4d55ab98836e6f935a5000f3"
S7_V4F_LAUNCHER_SHA256 = "50f8b89057b81d161399a70db1adbf32284e44afe03fe33072be6649abf7651f"
S7_V4F_LAUNCHER_MANIFEST_SHA256 = "fa9ea7655b87697ce25a7e9613d5fbd78d9f1751c4c6996595c1d8be5f722609"
S7_V4F_EXECUTOR_SHA256 = "2c7b01d5ac5833f40583f44f52cc649e5cf36a7d932da41a655ef85c736aff25"
S7_V4F_EXECUTION_CONTRACT_SHA256 = "95f90d7886cb007090c718b6291983ac745dc2bfc464f50ef86c448324ca771c"
S7_V4F_BUNDLE_MANIFEST_SHA256 = "5c84f74eee1aea2be71023794fdacfc66ee240e01473a2833907f54628a04cd4"
DS32_CSR_ADDRESS_CONTRACT_SHA256 = "1288e4e763437623760a361a60376af44c53cfa47eac649f5b2beb031ed6148f"
INSTRUCT_DIAGNOSTIC_PACKAGE = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-future-execution-package-20260809-v2.json"
)
INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-v2-package-fresh-review-binding-20260809.json"
)
INSTRUCT_DIAGNOSTIC_OUTPUT = (
    ROOT / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001"
)
INSTRUCT_DIAGNOSTIC_PREFLIGHT = ROOT / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001-authority-preflight.json"
INSTRUCT_DIAGNOSTIC_EXECUTOR = ROOT / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001-executor.py"
INSTRUCT_DIAGNOSTIC_TERMINAL_ROOT = ROOT / "build/build-freeze-instruct-simulator-binary-v1/attempt-0001-terminal-evidence"
INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE = INSTRUCT_DIAGNOSTIC_TERMINAL_ROOT / "terminal_evidence.json"
INSTRUCT_DIAGNOSTIC_TERMINAL_SEAL = INSTRUCT_DIAGNOSTIC_TERMINAL_ROOT / "TERMINAL_NO_REPLAY_SEAL.json"
INSTRUCT_DIAGNOSTIC_EVIDENCE_SUMS = INSTRUCT_DIAGNOSTIC_TERMINAL_ROOT / "EVIDENCE_SHA256SUMS"
INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-terminal-fresh-review-binding-20260809.json"
)
INSTRUCT_DIAGNOSTIC_PACKAGE_SHA256 = (
    "ad1cd918378de3a39c9638443f392ed30dfb23e62799a33970ea0ce7b18e4913"
)
INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING_SHA256 = (
    "07600adcc5a4f47ff95823ef2a83ac4e08de20626d603573d92c252418fc1e3b"
)
INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_HANDOFF_SHA256 = (
    "9b79690d77b6bae189bf4c471123441d722ae3ad41bc0a9244d1f0dfbcdfa125"
)
INSTRUCT_DIAGNOSTIC_PREFLIGHT_SHA256 = "bc24bc81752e307899405e31c4068fe4f1ee5d9fe8a9f0e0685047445a36eccf"
INSTRUCT_DIAGNOSTIC_EXECUTOR_SHA256 = "8d8b7bcc5726e88f3911fef7fb6d56fcaf829072e38023aff1539b3c5fd95b79"
INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE_SHA256 = "594db60ebd00d246781fca1758bb2eb81b7e37494f626c9515670b2c9444ad4a"
INSTRUCT_DIAGNOSTIC_TERMINAL_SEAL_SHA256 = "2f79edcdab4d19272e85b57b0d821784f3365e879a1bcdee83bd07bb8c5913d2"
INSTRUCT_DIAGNOSTIC_EVIDENCE_SUMS_SHA256 = "eebd3f79e63df77211081856774d0235c721afb32cc0dacb570ddec4bc73a430"
INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING_SHA256 = "8565d07cb26d3b9d9d9c8052c970eb5626d60933f9787af6e92e8ce03ab51e03"
INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_HANDOFF_SHA256 = "39a61491e664595caac8492080549b5beaec6efe752480cd5730ffc53e74aacc"
DIAGNOSTIC_FROZEN_PIPELINE_STAGE = "specification"
DIAGNOSTIC_FROZEN_PIPELINE_BYTE_COUNT = 183412
DIAGNOSTIC_FROZEN_PIPELINE_SHA256 = (
    "5a941e816d114438cd551e0405052e9686513931349c026728b47df715edbcba"
)
EXACT_ONCE_SUPERVISOR = ROOT / "tools/ace2_exact_once_runtime_supervisor.py"
EXACT_ONCE_SUPERVISOR_TEST = ROOT / "tools/test_ace2_exact_once_runtime_supervisor.py"
EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER = (
    ROOT / "tools/verify_ace2_exact_once_runtime_supervisor_python310.py"
)
EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE = (
    ROOT
    / "evidence/verification/make-exact-once-supervisor-python310-compatible-v1/"
    "decisive_verification.json"
)
EXACT_ONCE_SUPERVISOR_SHA256 = "e7c2c5ab3c44463e7546b2677236f0b05bff05ec99a9d803f23a94ae529c9619"
EXACT_ONCE_SUPERVISOR_TEST_SHA256 = "8317a992d84ef99f4fa17fbc3893ee27f677d089d43dc88381181ea4a9e14595"
EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER_SHA256 = (
    "0fb343178656e8bce987477200f9e35abcc870041489f0d10675423aee9b8dba"
)
EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE_SHA256 = (
    "5b039c5fcd95f57633f7461e4a5857791cc046f9a761e85f1d17288ad5c35c53"
)
INSTRUCT_DIAGNOSTIC_V3_PACKAGE = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-exact-once-execution-package-20260809-v3.json"
)
INSTRUCT_DIAGNOSTIC_V3_PROVENANCE = INSTRUCT_DIAGNOSTIC_V3_PACKAGE.with_name(
    "instruct-rtl-diagnostic-exact-once-execution-package-20260809-v3.provenance.json"
)
INSTRUCT_DIAGNOSTIC_V3_CHECKSUMS = INSTRUCT_DIAGNOSTIC_V3_PACKAGE.with_name(
    "instruct-rtl-diagnostic-exact-once-execution-package-20260809-v3.SHA256SUMS"
)
INSTRUCT_DIAGNOSTIC_V3_VERIFIER = (
    ROOT / "tools/verify_instruct_rtl_diagnostic_exact_once_package_v3.py"
)
INSTRUCT_DIAGNOSTIC_V3_TEST = (
    ROOT / "tools/test_verify_instruct_rtl_diagnostic_exact_once_package_v3.py"
)
INSTRUCT_DIAGNOSTIC_V3_PACKAGE_SHA256 = (
    "d0721e064934b9e51501564e994bc6be3afecfa2aa339c28d38c360a2bfec469"
)
INSTRUCT_DIAGNOSTIC_V3_PROVENANCE_SHA256 = (
    "c8c693d44d730a4e6ff1049c80a95a2a3a231690dee7a34a34fc81badb406e80"
)
INSTRUCT_DIAGNOSTIC_V3_VERIFIER_SHA256 = (
    "e8696a0bbb0784c887fadff70f5c8a2ef73a5152cf4099984cc739e3d30c1c09"
)
INSTRUCT_DIAGNOSTIC_V3_TEST_SHA256 = (
    "165acb2ec935f6a7a7214dbd3d1585ff21870097bf5e6dd23fc8419574bf73b1"
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY = (
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.authority.json"
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_CHECKSUM = INSTRUCT_DIAGNOSTIC_V3_AUTHORITY.with_suffix(
    INSTRUCT_DIAGNOSTIC_V3_AUTHORITY.suffix + ".sha256"
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_REVIEW = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-v3-authority-fresh-review-binding-20260809.json"
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_REVIEW_CHECKSUM = (
    INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_REVIEW.with_suffix(
        INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_REVIEW.suffix + ".sha256"
    )
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_SHA256 = (
    "0bd0a252809915afc799f63d382c3b41f60d5987de676793df10651b3c413a4a"
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_CHECKSUM_SHA256 = (
    "7f737790ebcc3fe8c6d4c1870f18c51f8e1d8531db18d1e72547057dc1224d55"
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_REVIEW_SHA256 = (
    "b663c5b8922da70c0d1a658393b1a490ab0d394efaacba10275eebd467edf701"
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_REVIEW_CHECKSUM_SHA256 = (
    "f171620af3b9bbbc123c4e6698c67057d2df9a6ca47a1e6afe288e0f41ce91e0"
)
INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_REVIEW_HANDOFF_SHA256 = (
    "3826773e1c397bb5431bd81eaadc152dddc49b59f124da6be38677a4ff33f31c"
)
INSTRUCT_DIAGNOSTIC_V3_RUNTIME_ABSENT_PATHS = (
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.state",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809-runtime-output",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.stdout.log",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.stderr.log",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v3-20260809.terminal.json",
)
INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT = (
    ROOT
    / "evidence/quarantine/"
    "execute-instruct-rtl-diagnostic-v3-terminal-no-execution-20260810T001420Z/"
    "containment.json"
)
INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT_SHA256 = (
    "6f4f6e7c3317aca64f2a43dc865a798b317fa598a12949f4bfa75fcaf1e254bc"
)
INSTRUCT_DIAGNOSTIC_V4_PACKAGE = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-exact-once-execution-package-20260810-v4.json"
)
INSTRUCT_DIAGNOSTIC_V4_PROVENANCE = INSTRUCT_DIAGNOSTIC_V4_PACKAGE.with_name(
    "instruct-rtl-diagnostic-exact-once-execution-package-20260810-v4.provenance.json"
)
INSTRUCT_DIAGNOSTIC_V4_CHECKSUMS = INSTRUCT_DIAGNOSTIC_V4_PACKAGE.with_name(
    "instruct-rtl-diagnostic-exact-once-execution-package-20260810-v4.SHA256SUMS"
)
INSTRUCT_DIAGNOSTIC_V4_AUTHORITY = (
    ROOT / "build/freeze-instruct-rtl-diagnostic-v4-20260810.authority.json"
)
INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_CHECKSUM = (
    INSTRUCT_DIAGNOSTIC_V4_AUTHORITY.with_suffix(
        INSTRUCT_DIAGNOSTIC_V4_AUTHORITY.suffix + ".sha256"
    )
)
INSTRUCT_DIAGNOSTIC_V4_REVIEW = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-v4-authority-fresh-review-binding-20260810.json"
)
INSTRUCT_DIAGNOSTIC_V4_REVIEW_CHECKSUM = INSTRUCT_DIAGNOSTIC_V4_REVIEW.with_suffix(
    INSTRUCT_DIAGNOSTIC_V4_REVIEW.suffix + ".sha256"
)
INSTRUCT_DIAGNOSTIC_V4_PACKAGE_SHA256 = (
    "586a1ea33f026746e636a677a922b937e40ff23e14a800adc8fa708c96dabb56"
)
INSTRUCT_DIAGNOSTIC_V4_PROVENANCE_SHA256 = (
    "81e1698e44251ce50601afc35272d60dc577e1e7ed5c59c78df8f068c74407a5"
)
INSTRUCT_DIAGNOSTIC_V4_CHECKSUMS_SHA256 = (
    "ca31825b33e7de1d8eafba3f39354ea05fed061966f42da4074a0e9e3d721e27"
)
INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_SHA256 = (
    "58e935132f1e06ee179ccf43886beb7d33ca86774596f1ea8e9412e1fdcef26c"
)
INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_CHECKSUM_SHA256 = (
    "3e3146f56a8cc6bea4add8dece49e922c62cd6346558b0c1cf6bd7cb6bf4c63e"
)
INSTRUCT_DIAGNOSTIC_V4_REVIEW_SHA256 = (
    "ec0972efafa78fb61f1551c4511d86a6b3bd09c75b7319ad4cdb2ec21594d902"
)
INSTRUCT_DIAGNOSTIC_V4_REVIEW_CHECKSUM_SHA256 = (
    "49b816d215f3da26c3023df98fbf34640f0816df963e70dc718a1818cf5a2e1f"
)
INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW = (
    ROOT
    / "research/raw/specification/"
    "instruct-rtl-diagnostic-v4-terminal-fresh-review-binding-20260810.json"
)
INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW_CHECKSUM = (
    INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW.with_suffix(
        INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW.suffix + ".sha256"
    )
)
INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW_SHA256 = (
    "86b90f7fd36853cac68d1fff1c0112d46cfbfb442564b86954631eef74262af7"
)
INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST = (
    ROOT
    / "evidence/diagnostics/execute-instruct-rtl-diagnostic-v4-once/"
    "TERMINAL_EVIDENCE_MANIFEST.json"
)
INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST_CHECKSUM = (
    INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST.with_suffix(
        INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST.suffix + ".sha256"
    )
)
INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST_SHA256 = (
    "fe65a9c8b2cc540b23fb9844e0decd7bb807c35e5451b0b209dab2b223884c19"
)
INSTRUCT_DIAGNOSTIC_V4_RUNTIME_PATHS = (
    ROOT / "build/freeze-instruct-rtl-diagnostic-v4-20260810.state",
    ROOT
    / "build/freeze-instruct-rtl-diagnostic-v4-20260810.state/pre_start_intent.json",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v4-20260810.stdout.log",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v4-20260810.stderr.log",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v4-20260810.terminal.json",
    ROOT / "build/freeze-instruct-rtl-diagnostic-v4-20260810-runtime-output",
)
INSTRUCT_DIAGNOSTIC_V4_TERMINAL = (
    ROOT / "build/freeze-instruct-rtl-diagnostic-v4-20260810.terminal.json"
)
INSTRUCT_DIAGNOSTIC_V4_DURABLE_TERMINAL = (
    ROOT
    / "build/freeze-instruct-rtl-diagnostic-v4-20260810.state/terminal_record.json"
)
INSTRUCT_DIAGNOSTIC_V4_PROCESS_STARTED = (
    ROOT
    / "build/freeze-instruct-rtl-diagnostic-v4-20260810.state/process_started.json"
)
INSTRUCT_DIAGNOSTIC_V4_FIRST_FAILURE = (
    ROOT
    / "build/freeze-instruct-rtl-diagnostic-v4-20260810-runtime-output/first_failure.json"
)
INSTRUCT_DIAGNOSTIC_V4_SUMMARY = (
    ROOT
    / "build/freeze-instruct-rtl-diagnostic-v4-20260810-runtime-output/summary.json"
)
INSTRUCT_DIAGNOSTIC_V4_PROGRESS = (
    ROOT
    / "build/freeze-instruct-rtl-diagnostic-v4-20260810-runtime-output/progress.json"
)
INSTRUCT_DIAGNOSTIC_V4_COMMANDS = (
    ROOT
    / "build/freeze-instruct-rtl-diagnostic-v4-20260810-runtime-output/commands.jsonl"
)

REQUIRED_ACCEPTANCE_CLASSES = ("Normal", "Boundary", "Illegal", "Reset", "Stall", "Recovery")
CURRENT_STAGE_PROJECTION = re.compile(r"`?current_stage\s*=\s*([A-Za-z0-9_.-]+)`?")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash_without_self_binding(record: dict[str, object]) -> str:
    payload = {key: value for key, value in record.items() if key != "self_binding"}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def project_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def bound_file(record: dict[str, object], path_key: str, hash_key: str) -> tuple[Path, bool]:
    path = project_path(record.get(path_key, ""))
    return path, bool(path.is_file() and sha256(path) == record.get(hash_key))


def all_boolean_checks_pass(record: dict[str, object]) -> bool:
    checks = record.get("checks", {})
    return isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())


def checksum_matches(path: Path, companion: Path) -> bool:
    if not companion.is_file():
        return False
    line = companion.read_text(encoding="utf-8").strip()
    return line == f"{sha256(path)}  {path.name}"


def companion_matches(path: Path) -> bool:
    return checksum_matches(path, path.with_suffix(path.suffix + ".sha256"))


def verify_instruct_diagnostic_v3_state() -> dict[str, object]:
    """Verify immutable V3 terminal no-execution containment without revalidation."""

    try:
        containment = load_json(INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT)
        provenance = load_json(INSTRUCT_DIAGNOSTIC_V3_PROVENANCE)
        authority_review = load_json(INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_REVIEW)
    except Exception as exc:
        return {
            "status": "FAIL_TERMINAL_CONTAINMENT_READ",
            "error": f"{type(exc).__name__}: {exc}",
        }

    attempt = containment.get("single_attempt", {})
    retirement = containment.get("retirement", {})
    protected = containment.get("protected_baseline", {})
    provenance_pipeline = provenance.get("input_roles", {}).get("pipeline_state", {})
    authority_review_pipeline = authority_review.get("protected_state", {})
    runtime_absent = all(
        not path.exists() and not path.is_symlink()
        for path in INSTRUCT_DIAGNOSTIC_V3_RUNTIME_ABSENT_PATHS
    )
    containment_ok = bool(
        INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT.is_file()
        and INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT.stat().st_size == 5337
        and sha256(INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT)
        == INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT_SHA256
        and checksum_matches(
            INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT,
            INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT.with_suffix(".sha256"),
        )
        and INSTRUCT_DIAGNOSTIC_V3_PACKAGE.is_file()
        and sha256(INSTRUCT_DIAGNOSTIC_V3_PACKAGE)
        == INSTRUCT_DIAGNOSTIC_V3_PACKAGE_SHA256
        and INSTRUCT_DIAGNOSTIC_V3_AUTHORITY.is_file()
        and sha256(INSTRUCT_DIAGNOSTIC_V3_AUTHORITY)
        == INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_SHA256
        and containment.get("status") == "TERMINAL_NO_EXECUTION_CONTAINMENT"
        and attempt.get("supervisor_payload_invocation_count") == 1
        and attempt.get("interpreter", {}).get("path") == "/usr/bin/python3"
        and attempt.get("interpreter", {}).get("version") == "3.10.12"
        and attempt.get("exit_code") == 1
        and attempt.get("failure_taxonomy") == "supervisor_module_import_failure"
        and attempt.get("supervisor_logic_entered") is False
        and attempt.get("reservation_created") is False
        and attempt.get("simulator_process_start_count") == 0
        and attempt.get("rtl_or_runtime_correctness_conclusion") == "none"
        and retirement.get("invocation_replayable") is False
        and retirement.get("task_reusable") is False
        and retirement.get("retry_resume_replay_or_revalidation_permitted") is False
        and retirement.get("package", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V3_PACKAGE_SHA256
        and retirement.get("authority", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_SHA256
        and protected.get("pipeline_state_sha256")
        == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
        and protected.get("pipeline_stage") == DIAGNOSTIC_FROZEN_PIPELINE_STAGE
        and provenance_pipeline.get("sha256")
        == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
        and provenance_pipeline.get("byte_count")
        == DIAGNOSTIC_FROZEN_PIPELINE_BYTE_COUNT
        and authority_review_pipeline.get("pipeline_state_sha256")
        == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
        and authority_review_pipeline.get("pipeline_stage")
        == DIAGNOSTIC_FROZEN_PIPELINE_STAGE
        and runtime_absent
    )
    return {
        "status": "PASS_TERMINAL_NO_EXECUTION_CONTAINMENT" if containment_ok else "FAIL_TERMINAL_CONTAINMENT",
        "package_sha256": sha256(INSTRUCT_DIAGNOSTIC_V3_PACKAGE),
        "authority_sha256": sha256(INSTRUCT_DIAGNOSTIC_V3_AUTHORITY),
        "containment_sha256": sha256(INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT),
        "supervisor_payload_invocations": attempt.get("supervisor_payload_invocation_count"),
        "state_reservations": 0,
        "process_starts": attempt.get("simulator_process_start_count"),
        "runtime_absent_path_count": len(INSTRUCT_DIAGNOSTIC_V3_RUNTIME_ABSENT_PATHS),
        "runtime_executed": False,
        "retry_or_revalidation_permitted": False,
        "rtl_correctness_conclusion": None,
        "historical_pipeline_snapshot_sha256": protected.get(
            "pipeline_state_sha256"
        ),
        "current_pipeline_sha256": sha256(PIPELINE),
        "current_pipeline_snapshot_required_to_match": False,
    }


def verify_instruct_diagnostic_v4_terminal() -> dict[str, object]:
    """Verify the immutable V4 terminal evidence without invoking it."""

    package = load_json(INSTRUCT_DIAGNOSTIC_V4_PACKAGE)
    provenance = load_json(INSTRUCT_DIAGNOSTIC_V4_PROVENANCE)
    authority = load_json(INSTRUCT_DIAGNOSTIC_V4_AUTHORITY)
    authority_review = load_json(INSTRUCT_DIAGNOSTIC_V4_REVIEW)
    terminal_review = load_json(INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW)
    manifest = load_json(INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST)
    terminal = load_json(INSTRUCT_DIAGNOSTIC_V4_TERMINAL)
    durable_terminal = load_json(INSTRUCT_DIAGNOSTIC_V4_DURABLE_TERMINAL)
    process_started = load_json(INSTRUCT_DIAGNOSTIC_V4_PROCESS_STARTED)
    first_failure = load_json(INSTRUCT_DIAGNOSTIC_V4_FIRST_FAILURE)
    summary = load_json(INSTRUCT_DIAGNOSTIC_V4_SUMMARY)
    progress = load_json(INSTRUCT_DIAGNOSTIC_V4_PROGRESS)
    command_lines = INSTRUCT_DIAGNOSTIC_V4_COMMANDS.read_text(
        encoding="utf-8"
    ).splitlines()
    command = json.loads(command_lines[0]) if len(command_lines) == 1 else {}

    package_bindings = package.get("bindings", {})
    authority_bindings = authority.get("bindings", {})
    bound_files = package_bindings.get("bound_files", [])
    source_tree = package_bindings.get("source_tree", {})
    tree_path = Path(str(source_tree.get("path", "/nonexistent")))
    tree_measurement = measure_source_tree(tree_path)
    pipeline_bound_files = [
        record
        for record in bound_files
        if isinstance(record, dict)
        and Path(str(record.get("path", ""))).resolve() == PIPELINE.resolve()
    ]
    bound_files_match = bool(
        isinstance(bound_files, list)
        and len(bound_files) == 23
        and all(
            isinstance(record, dict)
            and (
                (
                    Path(str(record.get("path", ""))).resolve()
                    == PIPELINE.resolve()
                    and record.get("byte_count")
                    == DIAGNOSTIC_FROZEN_PIPELINE_BYTE_COUNT
                    and record.get("sha256")
                    == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
                )
                or (
                    Path(str(record.get("path", ""))).resolve()
                    != PIPELINE.resolve()
                    and Path(str(record.get("path", ""))).is_file()
                    and Path(str(record.get("path", ""))).stat().st_size
                    == record.get("byte_count")
                    and sha256(Path(str(record.get("path", ""))))
                    == record.get("sha256")
                )
            )
            for record in bound_files
        )
        and len(pipeline_bound_files) == 1
    )
    manifest_files = manifest.get("files", [])
    manifest_files_match = bool(
        isinstance(manifest_files, list)
        and len(manifest_files) == 13
        and all(
            isinstance(record, dict)
            and project_path(record.get("path", "")).is_file()
            and project_path(record.get("path", "")).stat().st_size
            == record.get("bytes")
            and sha256(project_path(record.get("path", "")))
            == record.get("sha256")
            for record in manifest_files
        )
    )
    checksum_lines = INSTRUCT_DIAGNOSTIC_V4_CHECKSUMS.read_text(
        encoding="ascii"
    ).splitlines()
    expected_checksum_lines = [
        f"{INSTRUCT_DIAGNOSTIC_V4_PACKAGE_SHA256}  research/raw/specification/{INSTRUCT_DIAGNOSTIC_V4_PACKAGE.name}",
        f"{INSTRUCT_DIAGNOSTIC_V4_PROVENANCE_SHA256}  research/raw/specification/{INSTRUCT_DIAGNOSTIC_V4_PROVENANCE.name}",
    ]
    runtime_present = all(path.exists() for path in INSTRUCT_DIAGNOSTIC_V4_RUNTIME_PATHS)
    protected_paths = authority_review.get("runtime_observation", {}).get(
        "protected_paths", []
    )
    expected_protected_paths = [str(path) for path in INSTRUCT_DIAGNOSTIC_V4_RUNTIME_PATHS]
    authority_checksum_line = INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_CHECKSUM.read_text(
        encoding="ascii"
    ).strip()
    authority_review_checksum_line = INSTRUCT_DIAGNOSTIC_V4_REVIEW_CHECKSUM.read_text(
        encoding="ascii"
    ).strip()
    authority_review_runtime = authority_review.get("runtime_observation", {})
    authority_review_validation = authority_review.get("validation", {})
    authority_review_fresh_l2 = authority_review.get("fresh_l2_review", {})
    terminal_scope = terminal_review.get("accepted_scope", {})
    terminal_process = terminal.get("process", {})
    terminal_manifest_outcome = manifest.get("terminal_outcome", {})
    provenance_pipeline = provenance.get("protected_baselines", {}).get(
        "pipeline_state", {}
    )
    authority_review_pipeline = authority_review.get("protected_baselines", {}).get(
        "pipeline_state", {}
    )
    terminal_review_pipeline = terminal_review.get("protected_state", {})
    ok = bool(
        INSTRUCT_DIAGNOSTIC_V4_PACKAGE.stat().st_size == 7423
        and INSTRUCT_DIAGNOSTIC_V4_PACKAGE.stat().st_mode & 0o777 == 0o444
        and sha256(INSTRUCT_DIAGNOSTIC_V4_PACKAGE)
        == INSTRUCT_DIAGNOSTIC_V4_PACKAGE_SHA256
        and INSTRUCT_DIAGNOSTIC_V4_PROVENANCE.stat().st_size == 15316
        and INSTRUCT_DIAGNOSTIC_V4_PROVENANCE.stat().st_mode & 0o777 == 0o444
        and sha256(INSTRUCT_DIAGNOSTIC_V4_PROVENANCE)
        == INSTRUCT_DIAGNOSTIC_V4_PROVENANCE_SHA256
        and INSTRUCT_DIAGNOSTIC_V4_CHECKSUMS.stat().st_size == 337
        and sha256(INSTRUCT_DIAGNOSTIC_V4_CHECKSUMS)
        == INSTRUCT_DIAGNOSTIC_V4_CHECKSUMS_SHA256
        and checksum_lines == expected_checksum_lines
        and INSTRUCT_DIAGNOSTIC_V4_AUTHORITY.stat().st_size == 8321
        and sha256(INSTRUCT_DIAGNOSTIC_V4_AUTHORITY)
        == INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_SHA256
        and INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_CHECKSUM.stat().st_size == 152
        and sha256(INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_CHECKSUM)
        == INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_CHECKSUM_SHA256
        and authority_checksum_line
        == f"{INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_SHA256}  {INSTRUCT_DIAGNOSTIC_V4_AUTHORITY}"
        and INSTRUCT_DIAGNOSTIC_V4_REVIEW.stat().st_size == 7947
        and sha256(INSTRUCT_DIAGNOSTIC_V4_REVIEW)
        == INSTRUCT_DIAGNOSTIC_V4_REVIEW_SHA256
        and INSTRUCT_DIAGNOSTIC_V4_REVIEW_CHECKSUM.stat().st_size == 187
        and sha256(INSTRUCT_DIAGNOSTIC_V4_REVIEW_CHECKSUM)
        == INSTRUCT_DIAGNOSTIC_V4_REVIEW_CHECKSUM_SHA256
        and authority_review_checksum_line
        == f"{INSTRUCT_DIAGNOSTIC_V4_REVIEW_SHA256}  {INSTRUCT_DIAGNOSTIC_V4_REVIEW}"
        and INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW.stat().st_size == 2763
        and sha256(INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW)
        == INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW_SHA256
        and checksum_matches(
            INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW,
            INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW_CHECKSUM,
        )
        and INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST.stat().st_size == 4216
        and sha256(INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST)
        == INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST_SHA256
        and checksum_matches(
            INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST,
            INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST_CHECKSUM,
        )
        and package.get("schema") == "ace2-exact-once-execution-package-v1"
        and package.get("package_id")
        == "freeze-instruct-rtl-diagnostic-exact-once-package-v4-20260810"
        and provenance.get("package", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_PACKAGE_SHA256
        and authority.get("schema") == "ace2-exact-once-external-authority-v1"
        and authority.get("authority_id")
        == "mgr-exec-instruct-v4-586a1ea3-20260810T0052Z"
        and authority.get("decision") == "execute_once"
        and authority.get("package", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_PACKAGE_SHA256
        and authority_bindings == package_bindings
        and bound_files_match
        and tree_measurement.file_count == source_tree.get("file_count")
        and tree_measurement.byte_count == source_tree.get("byte_count")
        and tree_measurement.sha256 == source_tree.get("sha256")
        and authority_review.get("mission_id")
        == "prepare-instruct-rtl-diagnostic-v4-authority-v1"
        and authority_review_fresh_l2.get("status") == "ACCEPTED_FRESH_L2"
        and all(authority_review_fresh_l2.get("confirmed", {}).values())
        and authority_review_validation.get("result")
        == "PASS_ENGINEER_AND_FRESH_L2"
        and authority_review_validation.get("non_consuming") is True
        and authority_review_runtime.get("protected_paths")
        == expected_protected_paths
        and provenance_pipeline.get("sha256")
        == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
        and provenance_pipeline.get("byte_count")
        == DIAGNOSTIC_FROZEN_PIPELINE_BYTE_COUNT
        and provenance_pipeline.get("stage")
        == DIAGNOSTIC_FROZEN_PIPELINE_STAGE
        and authority_review_pipeline.get("before_sha256")
        == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
        and authority_review_pipeline.get("after_sha256")
        == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
        and authority_review_pipeline.get("byte_count")
        == DIAGNOSTIC_FROZEN_PIPELINE_BYTE_COUNT
        and authority_review_pipeline.get("unchanged") is True
        and terminal_review_pipeline.get("pipeline_state_sha256")
        == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
        and terminal_review_pipeline.get("pipeline_stage")
        == DIAGNOSTIC_FROZEN_PIPELINE_STAGE
        and terminal_review.get("mission_id")
        == "execute-instruct-rtl-diagnostic-v4-once"
        and terminal_review.get("review_round") == 1
        and terminal_review.get("producer_role") == "reviewer"
        and terminal_review.get("status") == "done"
        and terminal_review.get("source_handoff_sha256")
        == "8dd19e8343b35231d232009ab92ad44e46ffe86db2ceab3e93a2f33009ef7f91"
        and terminal_review.get("reviewed_terminal_record", {}).get("sha256")
        == "77ed483fac23b7c72281a7a3051f7d9765aa7797562388e3b62a758964c3159f"
        and terminal_scope.get("authority_consumed") is True
        and terminal_scope.get("supervisor_submission_count") == 1
        and terminal_scope.get("runtime_process_starts") == 1
        and terminal_scope.get("classification") == "child_nonzero_exit"
        and terminal_scope.get("child_exit_code") == 2
        and terminal_scope.get("failure_ordinal") == 0
        and terminal_scope.get("failure_operator") == "input_rmsnorm"
        and terminal_scope.get("failure_category") == "command_error"
        and terminal_scope.get("generated_token_count") == 0
        and terminal_scope.get("no_replay") is True
        and all(
            value is False
            for value in terminal_review.get("acceptance_effect", {}).values()
        )
        and terminal == durable_terminal
        and sha256(INSTRUCT_DIAGNOSTIC_V4_TERMINAL)
        == "77ed483fac23b7c72281a7a3051f7d9765aa7797562388e3b62a758964c3159f"
        and terminal.get("outcome") == "child_nonzero_exit"
        and terminal.get("process_start_count") == 1
        and terminal.get("process_started") is True
        and terminal.get("post_start_errors") == []
        and terminal_process.get("pid") == 258499
        and terminal_process.get("exit_code") == 2
        and terminal_process.get("signal") is None
        and terminal_process.get("wall_time_seconds") == 0.345733434
        and terminal_process.get("peak_rss_kib") == 193204
        and process_started.get("pid") == 258499
        and process_started.get("process_start_count") == 1
        and first_failure.get("status") == "STOPPED_AT_FIRST_GENUINE_BOUNDARY"
        and first_failure.get("ordinal") == 0
        and first_failure.get("operator") == "input_rmsnorm"
        and first_failure.get("category") == "command_error"
        and first_failure.get("generated_token_ids") == []
        and summary.get("status") == "STOPPED_AT_FIRST_GENUINE_BOUNDARY"
        and summary.get("generated_token_ids") == []
        and summary.get("resume_count") == 0
        and summary.get("first_failure") is None
        and progress.get("status") == "STOPPED_AT_FIRST_GENUINE_BOUNDARY"
        and progress.get("generated_token_ids") == []
        and progress.get("resume_count") == 0
        and command.get("ordinal") == 0
        and command.get("operator") == "input_rmsnorm"
        and command.get("cycles") == 21
        and command.get("completion_error") is True
        and command.get("generated_token_ids_after") == []
        and manifest.get("mission_id") == "execute-instruct-rtl-diagnostic-v4-once"
        and terminal_manifest_outcome.get("classification") == "child_nonzero_exit"
        and terminal_manifest_outcome.get("process_start_count") == 1
        and terminal_manifest_outcome.get("child_exit_code") == 2
        and terminal_manifest_outcome.get("failure_ordinal") == 0
        and terminal_manifest_outcome.get("failure_operator") == "input_rmsnorm"
        and terminal_manifest_outcome.get("generated_token_ids") == []
        and manifest_files_match
        and not any(manifest.get("claim_boundary", {}).values())
        and runtime_present
    )
    return {
        "status": (
            "PASS_TERMINAL_FRESH_L2_CHILD_NONZERO_NO_REPLAY"
            if ok
            else "FAIL_V4_TERMINAL"
        ),
        "package_sha256": sha256(INSTRUCT_DIAGNOSTIC_V4_PACKAGE),
        "authority_sha256": sha256(INSTRUCT_DIAGNOSTIC_V4_AUTHORITY),
        "authority_review_sha256": sha256(INSTRUCT_DIAGNOSTIC_V4_REVIEW),
        "terminal_review_sha256": sha256(INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW),
        "terminal_manifest_sha256": sha256(INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST),
        "terminal_record_sha256": sha256(INSTRUCT_DIAGNOSTIC_V4_TERMINAL),
        "authority_cardinality": 1,
        "authority_activated": True,
        "authority_consumed": True,
        "state_reservations": 1,
        "process_starts": 1,
        "runtime_present_path_count": len(INSTRUCT_DIAGNOSTIC_V4_RUNTIME_PATHS),
        "runtime_executed": True,
        "terminal_outcome": terminal.get("outcome"),
        "child_exit_code": terminal_process.get("exit_code"),
        "generated_token_count": len(summary.get("generated_token_ids", [])),
        "replay_permitted": False,
        "planner_execution_permitted": False,
        "rtl_correctness_conclusion": None,
        "historical_pipeline_snapshot_sha256": provenance_pipeline.get("sha256"),
        "current_pipeline_sha256": sha256(PIPELINE),
        "current_pipeline_snapshot_required_to_match": False,
    }


def markdown_binding_records(lines: list[str]) -> tuple[list[str], list[str]]:
    records: list[str] = []
    unexpected: list[str] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("- "):
            current = [line[2:].strip()]
            records.append(current[0])
        elif line.startswith("  ") and current is not None:
            records[-1] = f"{records[-1]} {line.strip()}"
        elif not line.strip() or line == "# Binding Facts and Unknowns" or line.startswith("## "):
            current = None
        else:
            current = None
            unexpected.append(line)
    return records, unexpected


def current_stage_projection_claims(records: list[str]) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for record in records:
        for match in CURRENT_STAGE_PROJECTION.finditer(record):
            claims.append(
                {
                    "projected_stage": match.group(1),
                    "statement": record,
                }
            )
    return claims


def table_classes(section: str) -> list[str]:
    classes: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in {"Class", "---"} and not set(cells[0]) <= {"-", ":"}:
            classes.append(cells[0])
    return classes


def contains_all(text: str, fragments: tuple[str, ...]) -> tuple[bool, list[str]]:
    missing = [fragment for fragment in fragments if fragment not in text]
    return not missing, missing


def freeze_artifacts_match(freeze: dict[str, object]) -> bool:
    artifacts = freeze.get("artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for relative, record in artifacts.items():
        if not isinstance(record, dict):
            return False
        path = ROOT / relative
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256(path) != record.get("sha256")
        ):
            return False
    return True


def package_manifest_artifacts_match(manifest: dict[str, object], expected_count: int) -> bool:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict) or len(artifacts) != expected_count:
        return False
    for record in artifacts.values():
        if not isinstance(record, dict):
            return False
        path = project_path(record.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256(path) != record.get("sha256")
        ):
            return False
    return True


def listed_artifacts_match(record: dict[str, object]) -> bool:
    artifacts = record.get("immutable_negative_evidence", [])
    if not isinstance(artifacts, list) or len(artifacts) != 10:
        return False
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False
        path = project_path(artifact.get("path", ""))
        expected_hash = artifact.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            or not path.is_file()
            or path.stat().st_size != artifact.get("bytes")
            or sha256(path) != expected_hash
        ):
            return False
    return True


def l2_project_bindings_match(l2: dict[str, object]) -> bool:
    bindings = l2.get("artifact_bindings", {})
    if not isinstance(bindings, dict):
        return False
    checked = 0
    for name, record in bindings.items():
        if name == "mission":
            continue
        if not isinstance(record, dict):
            return False
        path = project_path(record.get("path", ""))
        if not path.is_file() or sha256(path) != record.get("sha256"):
            return False
        checked += 1
    return checked >= 7


def audit() -> dict[str, object]:
    spec = SPEC.read_text(encoding="utf-8")
    benchmark = load_json(BENCHMARK)
    pipeline = load_json(PIPELINE)
    shell = audit_shell_contract()
    v4_diagnostic = load_json(V4_DIAGNOSTIC)
    v5_recommendation_l2 = load_json(V5_RECOMMENDATION_L2)
    v5_execution_contract = load_json(V5_EXECUTION_CONTRACT)
    v5_execution_manifest = load_json(V5_EXECUTION_MANIFEST)
    v5_execution_self_test = load_json(V5_EXECUTION_SELF_TEST)
    v5_execution_l2_decision = load_json(V5_EXECUTION_L2_DECISION)
    v5_execution_l2_acceptance = load_json(V5_EXECUTION_L2_ACCEPTANCE)
    v5_containment_input = load_json(V5_CONTAINMENT_INPUT)
    v5_containment_l2 = load_json(V5_CONTAINMENT_L2)
    v7_attempt_authority = load_json(V7_ATTEMPT_AUTHORITY)
    v7_terminal_audit = load_json(V7_TERMINAL_AUDIT)
    v7_fresh_review = load_json(V7_FRESH_REVIEW)
    v8_contract = load_json(V8_CONTRACT)
    v8_freeze = load_json(V8_FREEZE)
    v8_manifest = load_json(V8_MANIFEST)
    v8_self_test = load_json(V8_SELF_TEST)
    v8_review_acceptance = load_json(V8_REVIEW_ACCEPTANCE)
    v8_authority = load_json(V8_AUTHORITY)
    v8_inventory = load_json(V8_INVENTORY)
    v8_intent = load_json(V8_INTENT)
    v8_result = load_json(V8_RESULT)
    v8_post_inventory = load_json(V8_POST_INVENTORY)
    v8_terminal_audit = load_json(V8_TERMINAL_AUDIT)
    v8_fresh_review = load_json(V8_FRESH_REVIEW)
    s1_contract = load_json(S1_CONTRACT)
    s1_package_audit = load_json(S1_PACKAGE_AUDIT)
    s1_repair_audit = load_json(S1_REPAIR_AUDIT)
    s1_fresh_review = load_json(S1_FRESH_REVIEW)
    s1_authority = load_json(S1_AUTHORITY)
    s1_intent = load_json(S1_INTENT)
    s1_result = load_json(S1_RESULT)
    s1_terminal_audit = load_json(S1_TERMINAL_AUDIT)
    s1_terminal_review = load_json(S1_TERMINAL_REVIEW)
    s3_contract = load_json(S3_CONTRACT)
    s3_package_audit = load_json(S3_PACKAGE_AUDIT)
    s3_manifest = load_json(S3_MANIFEST)
    s3_self_test = load_json(S3_SELF_TEST)
    s3_l2_acceptance = load_json(S3_L2_ACCEPTANCE)
    s3_terminal_audit = load_json(S3_TERMINAL_AUDIT)
    s3_terminal_review = load_json(S3_TERMINAL_REVIEW)
    s4_contract = load_json(S4_CONTRACT)
    s4_package_audit = load_json(S4_PACKAGE_AUDIT)
    s4_manifest = load_json(S4_MANIFEST)
    s4_self_test = load_json(S4_SELF_TEST)
    s4_l2_acceptance = load_json(S4_L2_ACCEPTANCE)
    s4_terminal_audit = load_json(S4_TERMINAL_AUDIT)
    s4_terminal_review = load_json(S4_TERMINAL_REVIEW)
    s6_contract = load_json(S6_CONTRACT)
    s6_manifest = load_json(S6_MANIFEST)
    s6_l2_acceptance = load_json(S6_L2_ACCEPTANCE)
    s6_authority = load_json(S6_AUTHORITY)
    s6_submission_intent = load_json(S6_SUBMISSION_INTENT)
    s6_submission_result = load_json(S6_SUBMISSION_RESULT)
    s6_terminal_audit = load_json(S6_TERMINAL_AUDIT)
    latest = load_json(LATEST)
    s7_specification = load_json(S7_SPECIFICATION)
    s7_manifest = load_json(S7_MANIFEST)
    s7_fresh_review_binding = load_json(S7_FRESH_REVIEW_BINDING)
    s7_execution_manifest = load_json(S7_EXECUTION_MANIFEST)
    s7_execution_audit = load_json(S7_EXECUTION_AUDIT)
    s7_execution_fresh_review = load_json(S7_EXECUTION_FRESH_REVIEW)
    s7_custodian_attestation = load_json(S7_CUSTODIAN_ATTESTATION)
    s7_custodian_review_binding = load_json(S7_CUSTODIAN_REVIEW_BINDING)
    s7_control_plane_evidence = load_json(S7_CONTROL_PLANE_EVIDENCE)
    s7_control_plane_review_binding = load_json(S7_CONTROL_PLANE_REVIEW_BINDING)
    s7_v4f_terminal_binding = load_json(S7_V4F_TERMINAL_BINDING)
    s7_v4f_execution_attempt = load_json(S7_V4F_EXECUTION_ATTEMPT)
    s7_v4f_execution_result = load_json(S7_V4F_EXECUTION_RESULT)
    s7_v4f_review_handoff = load_json(S7_V4F_REVIEW_HANDOFF)
    ds32_csr_address_contract = load_json(DS32_CSR_ADDRESS_CONTRACT)
    instruct_diagnostic_package = load_json(INSTRUCT_DIAGNOSTIC_PACKAGE)
    instruct_diagnostic_package_review = load_json(INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING)
    instruct_diagnostic_terminal = load_json(INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE)
    instruct_diagnostic_terminal_seal = load_json(INSTRUCT_DIAGNOSTIC_TERMINAL_SEAL)
    instruct_diagnostic_terminal_review = load_json(INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING)
    exact_once_supervisor_test_evidence = load_json(EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE)
    instruct_diagnostic_v3_report = verify_instruct_diagnostic_v3_state()
    instruct_diagnostic_v4_report = verify_instruct_diagnostic_v4_terminal()

    matrix = spec.split("### Mission-specific acceptance matrix", 1)[1].split(
        "### Ordered Stage 2 U280 hold and host inventory", 1
    )[0]
    observed_classes = table_classes(matrix)
    missing_classes = [name for name in REQUIRED_ACCEPTANCE_CLASSES if name not in observed_classes]

    behavior_ok, behavior_missing = contains_all(
        spec,
        (
            "Local Qwen chat/runtime productization contract",
            "non-interactive UTF-8 stdin",
            "`--conversation-file PATH`",
            "deterministic greedy argmax",
            "KV cache",
            "Decode position `p` may not begin until position",
            "Public `ace2_shell` parameter and port contract",
            "`V4_TERMINAL_DEV_QUALITY_GATE_FAILURE`",
            "The bounded V4 mechanism-change freeze was completed",
            "Exactly one V4 attempt marker was created",
            "The one permitted official dev evaluation scored",
            "arbitrary natural-language accelerator-backed run",
            "`ABORTED_UNAUTHORIZED_PRETRAIN`",
            "Mission `e0bd8868635e` is `ABORTED_BLOCKED`",
            "`EVALUATOR_EXECUTION_FAILURE`",
            "ace2-v7-full-lifecycle-20260808-attempt-0001--16c53d80",
            "Fresh Reviewer mission `8e30ce003e79`",
            "TERMINAL_SUBMISSION_NO_GO",
            "BACKEND_CLIENT_INTERACTIVE_PROMPT_NO_EXECUTION",
            "`714e985f50df`",
            "ace2_score_gates_v1",
            "e7843803e930a9af491fe7e6ee7f9fe0a16fe7eb3b32faecd0eacf74b677ddb7",
            S1_PACKAGE_ID,
            S1_PACKAGE_IDENTITY,
            "PASS_ADDITIVE_S1_V1_REPAIR",
            "ffc5e2b1f181",
            "ace2-bf16-successor-s1-transport-20260808-pre-0e51f05b",
            "INERT_TRANSPORT_GATE_EXECUTED_TERMINAL_FAILED",
            "34/34",
            S3_PACKAGE_ID,
            S3_PACKAGE_IDENTITY,
            S3_EXECUTION_TREE,
            "PREATTEMPT_SOURCE_SNAPSHOT_AUDIT_OMITTED_FROM_STAGE",
            "64/64",
            "019fe161-c929-7c51-ac5d-6046a6cf989f",
            S4_PACKAGE_ID,
            S4_PACKAGE_IDENTITY,
            S4_EXECUTION_TREE,
            "84/84",
            "18/18",
            "39/39",
            "63/63",
            "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S4_PACKAGE_NO_ATTEMPT",
            "019fe16c-78a0-7903-bb3f-cc29d5be75cb",
            "Reviewed diagnostic Instruct runtime attempt",
            INSTRUCT_DIAGNOSTIC_PACKAGE_SHA256,
            INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE_SHA256,
            INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING_SHA256,
            "orchestration_supervisor_post_start_metadata_crash",
            "Future exactly-once runtime-supervisor contract",
            EXACT_ONCE_SUPERVISOR_SHA256,
            EXACT_ONCE_SUPERVISOR_TEST_SHA256,
            EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER_SHA256,
            EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE_SHA256,
            "Retired V3 diagnostic execution package",
            INSTRUCT_DIAGNOSTIC_V3_PACKAGE_SHA256,
            INSTRUCT_DIAGNOSTIC_V3_PROVENANCE_SHA256,
            INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT_SHA256,
            "Terminal V4 diagnostic execution (consumed, no replay)",
            INSTRUCT_DIAGNOSTIC_V4_PACKAGE_SHA256,
            INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_SHA256,
            INSTRUCT_DIAGNOSTIC_V4_REVIEW_SHA256,
            "S7 owned-reservation control-plane binding",
            "NO_SAFE_QUERY_BINDING",
            S7_CONTROL_PLANE_REVIEW_BINDING_SHA256,
            "Terminal S7 v4f reservation launch (consumed before candidate spawn)",
            S7_V4F_TERMINAL_BINDING_SHA256,
            S7_V4F_EXECUTION_ATTEMPT_SHA256,
            S7_V4F_EXECUTION_RESULT_SHA256,
            S7_V4F_REVIEW_HANDOFF_SHA256,
            "DS32 model-identity CSR",
            "`0x0A0`",
            DS32_CSR_ADDRESS_CONTRACT_SHA256,
            "DS32 command before `DS32_MODEL_IDENTITY` is programmed",
        ),
    )
    protocol_ok, protocol_missing = contains_all(
        spec,
        (
            "single-clock digital accelerator subsystem",
            "Asynchronous active-low reset",
            "deassert synchronously before state machines leave reset",
            "earliest read response on clock edge",
            "remain stable until accepted",
            "Completion valid; held until ready",
            "extend either latency without changing data, tags, ordering, or completion",
        ),
    )

    benchmark_active_contract = benchmark.get("active_productization_contract", {})
    benchmark_ds32 = (
        benchmark_active_contract.get("ds32_model_identity_csr", {})
        if isinstance(benchmark_active_contract, dict)
        else {}
    )
    local = benchmark.get("local_contract", {})
    internal = benchmark.get("internal_acceptance_contract", {})
    compression = benchmark.get("product_compression_gate", {})
    v3 = local.get("terminal_v3_mechanism_response_contract", {}) if isinstance(local, dict) else {}
    v4 = local.get("terminal_v4_mechanism_change_contract", {}) if isinstance(local, dict) else {}

    v3_paths: list[Path] = []
    v3_bindings = []
    for path_key, hash_key in (
        ("attempt_marker", "attempt_marker_sha256"),
        ("training_result", "training_result_sha256"),
        ("terminal_result", "terminal_result_sha256"),
        ("dev_selection", "dev_selection_sha256"),
    ):
        path, matches = bound_file(v3, path_key, hash_key) if isinstance(v3, dict) else (ROOT, False)
        v3_paths.append(path)
        v3_bindings.append(matches)
    v3_evidence_ok = bool(
        isinstance(v3, dict)
        and all(v3_bindings)
        and v3.get("status") == "TERMINAL_DEV_QUALITY_GATE_FAILURE_FRESH_REVIEW_DONE"
        and v3.get("attempt_marker_consumed") is True
        and v3.get("dev_hard_pass_counts") == [33, 31, 26, 27]
        and v3.get("selected_epoch") is None
        and v3.get("holdout_accessed") is False
        and v3.get("replay_resume_repair_or_checkpoint_reuse_allowed") is False
    )

    v4_paths: list[Path] = []
    v4_bindings: dict[str, bool] = {}
    for path_key, hash_key in (
        ("freeze_contract", "freeze_contract_sha256"),
        ("training_recipe", "training_recipe_sha256"),
        ("dataset_manifest", "dataset_manifest_sha256"),
        ("freeze_manifest", "freeze_manifest_sha256"),
        ("package_self_test", "package_self_test_sha256"),
        ("freeze_audit", "freeze_audit_sha256"),
        ("independent_l2_acceptance", "independent_l2_acceptance_sha256"),
    ):
        path, matches = bound_file(v4, path_key, hash_key) if isinstance(v4, dict) else (ROOT, False)
        v4_paths.append(path)
        v4_bindings[path_key] = matches

    v4_terminal_paths: dict[str, Path] = {}
    v4_terminal_bindings: dict[str, bool] = {}
    for path_key, hash_key in (
        ("execution_contract", "execution_contract_sha256"),
        ("execution_manifest", "execution_manifest_sha256"),
        ("execution_l2", "execution_l2_sha256"),
        ("attempt_authority", "attempt_authority_sha256"),
        ("attempt_marker", "attempt_marker_sha256"),
        ("training_result", "training_result_sha256"),
        ("checkpoint_lock", "checkpoint_lock_sha256"),
        ("dev_result", "dev_result_sha256"),
        ("terminal_result", "terminal_result_sha256"),
        ("process_exit_status", "process_exit_status_sha256"),
        ("fresh_reviewer_handoff", "fresh_reviewer_handoff_sha256"),
    ):
        path, matches = bound_file(v4, path_key, hash_key) if isinstance(v4, dict) else (ROOT, False)
        v4_terminal_paths[path_key] = path
        v4_terminal_bindings[path_key] = matches

    freeze_contract = load_json(v4_paths[0]) if v4_bindings.get("freeze_contract") else {}
    recipe = load_json(v4_paths[1]) if v4_bindings.get("training_recipe") else {}
    dataset = load_json(v4_paths[2]) if v4_bindings.get("dataset_manifest") else {}
    freeze = load_json(v4_paths[3]) if v4_bindings.get("freeze_manifest") else {}
    self_test = load_json(v4_paths[4]) if v4_bindings.get("package_self_test") else {}
    freeze_audit = load_json(v4_paths[5]) if v4_bindings.get("freeze_audit") else {}
    l2 = load_json(v4_paths[6]) if v4_bindings.get("independent_l2_acceptance") else {}
    attestation = load_json(V4_ATTESTATION)
    attempt_authority = (
        load_json(v4_terminal_paths["attempt_authority"])
        if v4_terminal_bindings.get("attempt_authority")
        else {}
    )
    attempt_marker = (
        load_json(v4_terminal_paths["attempt_marker"])
        if v4_terminal_bindings.get("attempt_marker")
        else {}
    )
    training_result = (
        load_json(v4_terminal_paths["training_result"])
        if v4_terminal_bindings.get("training_result")
        else {}
    )
    checkpoint_lock = (
        load_json(v4_terminal_paths["checkpoint_lock"])
        if v4_terminal_bindings.get("checkpoint_lock")
        else {}
    )
    dev_result = (
        load_json(v4_terminal_paths["dev_result"])
        if v4_terminal_bindings.get("dev_result")
        else {}
    )
    terminal_result = (
        load_json(v4_terminal_paths["terminal_result"])
        if v4_terminal_bindings.get("terminal_result")
        else {}
    )
    exit_status = (
        load_json(v4_terminal_paths["process_exit_status"])
        if v4_terminal_bindings.get("process_exit_status")
        else {}
    )
    reviewer_handoff = (
        load_json(v4_terminal_paths["fresh_reviewer_handoff"])
        if v4_terminal_bindings.get("fresh_reviewer_handoff")
        else {}
    )

    interlocks = freeze_contract.get("execution_interlocks", {})
    attempt_policy = recipe.get("attempt_policy", {})
    checkpoint_lifecycle = recipe.get("checkpoint_lifecycle", {})
    dataset_counts = dataset.get("counts", {})
    comparisons = attestation.get("results", {}).get("comparisons", {})
    blinded_zero = bool(
        attestation.get("method", {}).get("scan_ordinal") == 1
        and attestation.get("results", {}).get("status") == "PASS"
        and comparisons
        and all(
            item.get("exact_normalized_prompt_collision_count") == 0
            and item.get("near_duplicate_collision_count") == 0
            and item.get("template_family_collision_count") == 0
            for item in comparisons.values()
        )
    )
    v4_freeze_ok = bool(
        isinstance(v4, dict)
        and all(v4_bindings.values())
        and v4.get("aggregate_train_rows") == 352
        and v4.get("v1_byte_identical_backbone_rows") == 280
        and v4.get("additive_patch_rows") == 72
        and v4.get("fresh_disjoint_probe_rows") == 28
        and v4.get("dev_minimum_hard_passes") == 48
        and v4.get("dev_thresholds_changed") is False
        and v4.get("selected_policy_id") is None
        and freeze.get("attempt_authority") is False
        and freeze.get("attempt_marker_creation_authorized") is False
        and freeze.get("training_executed") is False
        and freeze.get("model_namespace_created") is False
        and interlocks.get("attempt_authority") is False
        and interlocks.get("attempt_marker_creation") is False
        and interlocks.get("training_authority") is False
        and attempt_policy.get("attempt_authority_granted") is False
        and attempt_policy.get("attempt_marker_creation_authorized") is False
        and attempt_policy.get("training_authority_granted") is False
        and checkpoint_lifecycle.get("selection_source") == "fresh disjoint synthetic probes only"
        and checkpoint_lifecycle.get("checkpoint_lock_before_official_dev_access") is True
        and checkpoint_lifecycle.get("official_dev_role") == "QUALIFY_LOCKED_CHECKPOINT_ONLY"
        and checkpoint_lifecycle.get("official_dev_selects_or_reselects_checkpoint") is False
        and dataset_counts.get("v1_backbone") == 280
        and dataset_counts.get("additive_patch") == 72
        and dataset_counts.get("train") == 352
        and dataset_counts.get("synthetic_probes") == 28
        and self_test.get("status") == "PASS"
        and self_test.get("check_count") == 44
        and all_boolean_checks_pass(self_test)
        and freeze_audit.get("status") == "PASS_FROZEN_PENDING_INDEPENDENT_L2"
        and freeze_audit.get("check_count") == 36
        and freeze_audit.get("freeze_manifest_sha256") == sha256(v4_paths[3])
        and all_boolean_checks_pass(freeze_audit)
        and l2.get("accepted") is True
        and l2.get("status") == "ACCEPTED_FROZEN_PACKAGE_NO_ATTEMPT_NO_TRAINING"
        and l2.get("attempt_authority_granted") is False
        and l2.get("training_authority_granted") is False
        and all_boolean_checks_pass(l2)
        and l2_project_bindings_match(l2)
        and freeze_artifacts_match(freeze)
        and companion_matches(v4_paths[3])
        and blinded_zero
    )

    attempt_markers = list(V4_RUN_ROOT.rglob("ATTEMPT_CONSUMPTION_MARKER.json"))
    v4_terminal_ok = bool(
        v4_freeze_ok
        and all(v4_terminal_bindings.values())
        and v4.get("status") == "TERMINAL_DEV_QUALITY_GATE_FAILURE_FRESH_REVIEW_DONE"
        and v4.get("official_model_namespace_exists") is True
        and v4.get("attempt_authority_granted") is True
        and v4.get("attempt_marker_creation_authorized") is True
        and v4.get("attempt_marker_consumed") is True
        and v4.get("training_authority_granted") is True
        and v4.get("training_executed") is True
        and v4.get("training_optimizer_steps") == 264
        and v4.get("candidate_epochs") == [1, 2, 3, 4, 5, 6]
        and v4.get("probe_hard_pass_counts") == [10, 13, 13, 15, 13, 14]
        and v4.get("probe_locked_epoch") == 4
        and v4.get("dev_accessed") is True
        and v4.get("dev_hard_pass_count") == 36
        and v4.get("dev_status") == "NO_GO"
        and v4.get("first_failed_gate") == "minimum_hard_passes"
        and v4.get("checkpoint_reselection_after_dev") is False
        and v4.get("merge_executed") is False
        and v4.get("retention_executed") is False
        and v4.get("holdout_accessed") is False
        and v4.get("process_exit_code") == 2
        and v4.get("fresh_reviewer_status") == "done"
        and v4.get("fresh_reviewer_mission_id") == "8e59f69debde"
        and v4.get("replay_resume_repair_or_reselection_allowed") is False
        and attempt_authority.get("status") == "AUTHORIZE_SINGLE_V4_ATTEMPT"
        and attempt_authority.get("attempt") == 1
        and attempt_authority.get("resume_allowed") is False
        and attempt_marker.get("attempt") == 1
        and attempt_marker.get("resume_allowed") is False
        and len(attempt_markers) == 1
        and attempt_markers[0] == v4_terminal_paths["attempt_marker"]
        and training_result.get("status") == "PASS_CHECKPOINT_LOCKED_BEFORE_DEV"
        and training_result.get("optimizer_steps") == 264
        and training_result.get("selected_epoch") == 4
        and checkpoint_lock.get("checkpoint_epoch") == 4
        and checkpoint_lock.get("dev_accessed") is False
        and checkpoint_lock.get("reason")
        == "epoch6_fallback_highest_hard_pass_then_category_gates_then_lower_epoch"
        and dev_result.get("status") == "NO_GO"
        and dev_result.get("hard_pass_count") == 36
        and dev_result.get("official_dev_checkpoint_count") == 1
        and dev_result.get("official_dev_reselection_allowed") is False
        and dev_result.get("critical_safety_failures") == 0
        and terminal_result.get("status") == "NO_GO"
        and terminal_result.get("failure_taxonomy") == "DEV_QUALITY_GATE_FAILURE"
        and terminal_result.get("first_failed_gate") == "minimum_hard_passes"
        and exit_status.get("status") == "FAIL_CLOSED"
        and exit_status.get("exit_code") == 2
        and reviewer_handoff.get("producer_role") == "reviewer"
        and reviewer_handoff.get("mission_id") == "8e59f69debde"
        and reviewer_handoff.get("review", {}).get("status") == "done"
        and companion_matches(v4_terminal_paths["fresh_reviewer_handoff"])
    )

    v5_recommendation = v4_diagnostic.get("v5_recommendation", {})
    v5_authority = v4_diagnostic.get("authority_and_review", {})
    diagnostic_boundary = v4_diagnostic.get("evidence_boundary", {})
    benchmark_v5 = v4.get("recommended_v5", {}) if isinstance(v4, dict) else {}
    v5_l2_reason = v5_recommendation_l2.get("review", {}).get("reason", "")
    v5_recommendation_l2_ok = bool(
        checksum_matches(V5_RECOMMENDATION_L2, V5_RECOMMENDATION_L2_CHECKSUM)
        and v5_recommendation_l2.get("kind") == "round_reviewed_handoff"
        and v5_recommendation_l2.get("mission_id") == "3ba321fcf5a1"
        and v5_recommendation_l2.get("producer_role") == "reviewer"
        and v5_recommendation_l2.get("round") == 2
        and v5_recommendation_l2.get("review", {}).get("status") == "done"
        and "5c05d70b…d34d8e2" in v5_l2_reason
        and "V5 recommendation only" in v5_l2_reason
        and "no execution authority is granted" in v5_l2_reason
    )
    v4_diagnostic_ok = bool(
        companion_matches(V4_DIAGNOSTIC)
        and v4_diagnostic.get("status")
        == "RECOMMEND_V5_SPECIFICATION_ONLY_PENDING_INDEPENDENT_L2"
        and v4_diagnostic.get("failure_taxonomy") == "DEV_QUALITY_GATE_FAILURE"
        and v4_diagnostic.get("observed_aggregate_facts", {})
        .get("comparisons", {})
        .get("selected_probe_to_dev_gate_mismatch_count")
        == 5
        and v4_diagnostic.get("failure_mechanism_assessment", {})
        .get("selector_to_gate_misalignment", {})
        .get("classification")
        == "SUPPORTED_HIGH"
        and diagnostic_boundary.get("official_dev_scored_rows_read") is False
        and diagnostic_boundary.get("official_dev_or_holdout_prompts_answers_or_responses_read")
        is False
        and diagnostic_boundary.get("checkpoint_inference_replayed") is False
        and diagnostic_boundary.get("training_resumed_repaired_or_started") is False
        and v5_recommendation.get("candidate_id")
        == "qwen2.5-0.5b-instruct-bf16-lora-product-v5-staged-v1-parent-r8a16"
        and v5_recommendation.get("status") == "RECOMMENDATION_ONLY_NOT_FROZEN_NOT_AUTHORIZED"
        and v5_recommendation.get("parent", {}).get("adapter", {}).get("sha256")
        == "37acd8f25f78eeeec102840a7a21c978b3391470993b8e7c685dec0aaf2b2116"
        and v5_recommendation.get("proposed_training_bounds", {}).get("rank") == 8
        and v5_recommendation.get("proposed_training_bounds", {}).get("scaling_alpha") == 16
        and v5_recommendation.get("proposed_training_bounds", {}).get("maximum_patch_fraction")
        == 0.125
        and v5_recommendation.get("prospective_selector", {}).get("fallback_if_no_candidate_passes")
        == "PREDEV_NO_GO_WITH_NO_OFFICIAL_DEV_ACCESS"
        and v5_authority.get("independent_l2_required") is True
        and v5_authority.get("v5_dataset_or_selector_frozen") is False
        and v5_authority.get("attempt_marker_creation_authorized") is False
        and v5_authority.get("training_authorized") is False
        and v5_authority.get("official_dev_access_authorized") is False
        and isinstance(v4, dict)
        and v4.get("aggregate_failure_diagnostic")
        == V4_DIAGNOSTIC.relative_to(ROOT).as_posix()
        and v4.get("aggregate_failure_diagnostic_sha256") == sha256(V4_DIAGNOSTIC)
        and benchmark_v5.get("candidate_id") == v5_recommendation.get("candidate_id")
        and benchmark_v5.get("recommendation_source_status") == v5_recommendation.get("status")
        and benchmark_v5.get("status")
        == "TERMINAL_ABORTED_UNAUTHORIZED_PRETRAIN_L2_DONE"
        and benchmark_v5.get("selector_fallback_allowed") is False
        and benchmark_v5.get("recommendation_independent_l2_accepted") is True
        and benchmark_v5.get("exact_preexecution_package_independent_l2_required") is True
        and benchmark_v5.get("dataset_or_selector_frozen") is True
        and benchmark_v5.get("training_authorized") is False
        and v4.get("aggregate_failure_diagnostic_review_status")
        == "ACCEPT_EXACT_DIAGNOSTIC_NO_EXECUTION_AUTHORITY"
        and v4.get("aggregate_failure_diagnostic_l2_mission_id") == "3ba321fcf5a1"
        and v4.get("aggregate_failure_diagnostic_l2_handoff")
        == V5_RECOMMENDATION_L2.relative_to(ROOT).as_posix()
        and v4.get("aggregate_failure_diagnostic_l2_handoff_sha256")
        == sha256(V5_RECOMMENDATION_L2)
    )

    expected_v5_namespace_files = {
        "attempt-0001/ATTEMPT_CONSUMPTION_MARKER.json",
        "attempt-0001-exit-status.json",
    }
    observed_v5_namespace_files = {
        path.relative_to(V5_RUN_ROOT).as_posix()
        for path in V5_RUN_ROOT.rglob("*")
        if path.is_file()
    } if V5_RUN_ROOT.is_dir() else set()
    v5_checkpoint_dir = V5_RUN_ROOT / "attempt-0001/checkpoints"
    v5_l2_artifacts = v5_containment_l2.get("per_artifact_table", [])

    v5_package_ok = bool(
        companion_matches(V5_EXECUTION_CONTRACT)
        and companion_matches(V5_EXECUTION_MANIFEST)
        and companion_matches(V5_EXECUTION_SELF_TEST)
        and companion_matches(V5_EXECUTION_L2_DECISION)
        and companion_matches(V5_EXECUTION_L2_ACCEPTANCE)
        and companion_matches(V5_CONTAINMENT_INPUT)
        and companion_matches(V5_CONTAINMENT_L2)
        and companion_matches(V5_ATTEMPT_AUTHORITY)
        and companion_matches(V5_LAUNCH_READINESS)
        and v5_execution_contract.get("contract_id")
        == "qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5-execution-package-v1"
        and v5_execution_contract.get("attempt_authority", {}).get(
            "separate_fresh_operator_authority_required"
        )
        is True
        and v5_execution_manifest.get("tree_sha256")
        == "6713a11ad5c4da32d5ef700a3c9df1196f8619f836544b50f902b2f3cd28d70e"
        and v5_execution_self_test.get("status") == "PASS"
        and v5_execution_self_test.get("check_count") == 25
        and all_boolean_checks_pass(v5_execution_self_test)
        and v5_execution_self_test.get("official_namespace_exists") is False
        and v5_execution_self_test.get("execution_tree_sha256")
        == v5_execution_manifest.get("tree_sha256")
        and v5_execution_self_test.get("execution_manifest_sha256")
        == sha256(V5_EXECUTION_MANIFEST)
        and v5_execution_l2_decision.get("reviewer_status") == "done"
        and v5_execution_l2_decision.get("execution_tree_sha256")
        == v5_execution_manifest.get("tree_sha256")
        and v5_execution_l2_acceptance.get("status")
        == "ACCEPTED_V5_EXACT_PACKAGE_NO_ATTEMPT"
        and v5_execution_l2_acceptance.get("accepted") is True
        and v5_execution_l2_acceptance.get("attempt_authority_granted") is False
        and v5_execution_l2_acceptance.get("independent_from_constructor") is True
        and v5_execution_l2_acceptance.get("execution_tree_sha256")
        == v5_execution_manifest.get("tree_sha256")
        and v5_execution_l2_acceptance.get("execution_manifest_sha256")
        == sha256(V5_EXECUTION_MANIFEST)
        and v5_execution_l2_acceptance.get("contract_sha256")
        == sha256(V5_EXECUTION_CONTRACT)
        and v5_execution_l2_acceptance.get("reviewer_decision_sha256")
        == sha256(V5_EXECUTION_L2_DECISION)
        and listed_artifacts_match(v5_containment_input)
        and v5_containment_input.get("mission_id") == "e0bd8868635e"
        and v5_containment_input.get("attempt_id") == "attempt-0001"
        and v5_containment_input.get("classification_requested")
        == "ABORTED_UNAUTHORIZED_PRETRAIN"
        and v5_containment_input.get("task_disposition") == "ABORTED_BLOCKED"
        and v5_containment_input.get("authority_finding", {}).get(
            "fresh_operator_authority_supplied"
        )
        is False
        and v5_containment_input.get("authority_finding", {}).get(
            "authority_file_self_authored_by_engineer"
        )
        is True
        and v5_containment_input.get("execution_finding", {}).get(
            "attempt_marker_consumed"
        )
        is True
        and v5_containment_input.get("execution_finding", {}).get(
            "training_or_evaluation_outcome_established"
        )
        is False
        and v5_containment_input.get("invalid_first_l2_review", {}).get("sha256")
        == sha256(V5_INVALID_CONTAINMENT_L2)
        and v5_containment_input.get("invalid_first_l2_review", {}).get(
            "acceptance_or_closure_use_allowed"
        )
        is False
        and v5_containment_input.get("supersedes_invalid_review_input", {}).get(
            "sha256"
        )
        == sha256(V5_INVALID_CONTAINMENT_INPUT)
        and v5_containment_l2.get("producer_role") == "reviewer"
        and v5_containment_l2.get("review_level") == "L2"
        and v5_containment_l2.get("independent_from_engineer_planner_and_first_reviewer")
        is True
        and v5_containment_l2.get("status") == "done"
        and v5_containment_l2.get("review_input", {}).get("sha256")
        == sha256(V5_CONTAINMENT_INPUT)
        and v5_containment_l2.get("review_input", {}).get(
            "sha256_is_lowercase_hex_64"
        )
        is True
        and isinstance(v5_l2_artifacts, list)
        and len(v5_l2_artifacts) == 10
        and all(
            isinstance(record, dict)
            and record.get("bytes_match") is True
            and record.get("sha256_match") is True
            and record.get("expected_sha256_is_lowercase_hex_64") is True
            and record.get("observed_sha256_is_lowercase_hex_64") is True
            for record in v5_l2_artifacts
        )
        and v5_containment_l2.get("integrity_summary", {}).get(
            "all_expected_sizes_match_observed_sizes"
        )
        is True
        and v5_containment_l2.get("integrity_summary", {}).get(
            "all_expected_sha256_match_independently_observed_sha256"
        )
        is True
        and v5_containment_l2.get("classification", {}).get("failure_taxonomy")
        == "ABORTED_UNAUTHORIZED_PRETRAIN"
        and v5_containment_l2.get("classification", {}).get("v5_terminal") is True
        and v5_containment_l2.get("classification", {}).get("attempt_0002_prohibited")
        is True
        and v5_containment_l2.get("console_and_exit_record", {}).get(
            "code_0_establishes_training_or_evaluation_success"
        )
        is False
        and v5_containment_l2.get("official_v5_namespace", {}).get(
            "training_or_evaluation_outcome_established"
        )
        is False
        and v5_containment_l2.get("task_disposition", {}).get("disposition")
        == "ABORTED_BLOCKED"
        and v5_containment_l2.get("invalid_first_review_disposition", {}).get(
            "excluded_from_acceptance_and_closure"
        )
        is True
        and V5_ATTEMPT_AUTHORITY.is_file()
        and V5_LAUNCH_READINESS.is_file()
        and V5_ATTEMPT_MARKER.is_file()
        and V5_EXIT_STATUS.is_file()
        and V5_ATTEMPT_CONSOLE.is_file()
        and observed_v5_namespace_files == expected_v5_namespace_files
        and v5_checkpoint_dir.is_dir()
        and not any(v5_checkpoint_dir.iterdir())
        and not (V5_RUN_ROOT / "attempt-0002").exists()
        and benchmark_v5.get("execution_dataset_id")
        == "qwen2.5-0.5b-instruct-ace2-bf16-lora-product-v5"
        and benchmark_v5.get("execution_package_contract")
        == V5_EXECUTION_CONTRACT.relative_to(ROOT).as_posix()
        and benchmark_v5.get("execution_package_contract_sha256")
        == sha256(V5_EXECUTION_CONTRACT)
        and benchmark_v5.get("execution_package_manifest")
        == V5_EXECUTION_MANIFEST.relative_to(ROOT).as_posix()
        and benchmark_v5.get("execution_package_manifest_sha256")
        == sha256(V5_EXECUTION_MANIFEST)
        and benchmark_v5.get("execution_package_tree_sha256")
        == v5_execution_manifest.get("tree_sha256")
        and benchmark_v5.get("execution_package_self_test")
        == V5_EXECUTION_SELF_TEST.relative_to(ROOT).as_posix()
        and benchmark_v5.get("execution_package_self_test_sha256")
        == sha256(V5_EXECUTION_SELF_TEST)
        and benchmark_v5.get("execution_package_self_test_check_count") == 25
        and benchmark_v5.get("execution_package_l2_acceptance")
        == V5_EXECUTION_L2_ACCEPTANCE.relative_to(ROOT).as_posix()
        and benchmark_v5.get("execution_package_l2_acceptance_sha256")
        == sha256(V5_EXECUTION_L2_ACCEPTANCE)
        and benchmark_v5.get("execution_package_l2_status")
        == "ACCEPTED_V5_EXACT_PACKAGE_NO_ATTEMPT"
        and benchmark_v5.get("attempt_authority_granted") is False
        and benchmark_v5.get("attempt_authority_file_present") is True
        and benchmark_v5.get("attempt_authority_file_binding") is False
        and benchmark_v5.get("attempt_authority_file_sha256")
        == sha256(V5_ATTEMPT_AUTHORITY)
        and benchmark_v5.get("official_namespace_exists") is True
        and benchmark_v5.get("attempt_marker_exists") is True
        and benchmark_v5.get("attempt_marker_sha256") == sha256(V5_ATTEMPT_MARKER)
        and benchmark_v5.get("launch_readiness_exists") is True
        and benchmark_v5.get("launch_readiness_sha256") == sha256(V5_LAUNCH_READINESS)
        and benchmark_v5.get("attempt_console_sha256") == sha256(V5_ATTEMPT_CONSOLE)
        and benchmark_v5.get("observed_optimizer_steps_completed") == 0
        and benchmark_v5.get("checkpoint_present") is False
        and benchmark_v5.get("training_result_present") is False
        and benchmark_v5.get("selector_result_present") is False
        and benchmark_v5.get("official_dev_result_present") is False
        and benchmark_v5.get("merge_result_present") is False
        and benchmark_v5.get("retention_result_present") is False
        and benchmark_v5.get("holdout_result_present") is False
        and benchmark_v5.get("quantization_result_present") is False
        and benchmark_v5.get("training_or_evaluation_outcome_established") is False
        and benchmark_v5.get("exit_status_sha256") == sha256(V5_EXIT_STATUS)
        and benchmark_v5.get("containment_input_sha256") == sha256(V5_CONTAINMENT_INPUT)
        and benchmark_v5.get("containment_l2_review_sha256") == sha256(V5_CONTAINMENT_L2)
        and benchmark_v5.get("invalid_first_containment_input_sha256")
        == sha256(V5_INVALID_CONTAINMENT_INPUT)
        and benchmark_v5.get("invalid_first_l2_review_sha256")
        == sha256(V5_INVALID_CONTAINMENT_L2)
        and benchmark_v5.get("invalid_first_l2_review_disposition")
        == "INVALID_FALSE_PASS_DIGEST_MISMATCH_NOT_DETECTED_EXCLUDED_FROM_ACCEPTANCE"
        and benchmark_v5.get("containment_l2_status") == "done"
        and benchmark_v5.get("failure_taxonomy") == "ABORTED_UNAUTHORIZED_PRETRAIN"
        and benchmark_v5.get("task_disposition") == "ABORTED_BLOCKED"
        and benchmark_v5.get("attempt_0002_allowed") is False
        and benchmark_v5.get("replay_resume_repair_or_rescore_allowed") is False
    )

    benchmark_v7 = benchmark.get("v7_terminal_lifecycle", {})
    benchmark_v7_authority = (
        benchmark_v7.get("authority", {}) if isinstance(benchmark_v7, dict) else {}
    )
    benchmark_v7_backend = (
        benchmark_v7.get("backend", {}) if isinstance(benchmark_v7, dict) else {}
    )
    benchmark_v7_audit = (
        benchmark_v7.get("terminal_evidence_audit", {})
        if isinstance(benchmark_v7, dict)
        else {}
    )
    benchmark_v7_review = (
        benchmark_v7.get("fresh_reviewer", {}) if isinstance(benchmark_v7, dict) else {}
    )
    v7_terminal_ok = bool(
        companion_matches(V7_ATTEMPT_AUTHORITY)
        and companion_matches(V7_TERMINAL_AUDIT)
        and companion_matches(V7_FRESH_REVIEW)
        and v7_attempt_authority.get("status") == "AUTHORIZE_SINGLE_V7_FULL_FINETUNE_ATTEMPT"
        and v7_attempt_authority.get("attempt") == 1
        and v7_attempt_authority.get("execution_tree_sha256")
        == "99adb88967a4ebd16f8f6b342147a6dea1d9e8f55b6042a8d6fe88d4139ec2c2"
        and v7_attempt_authority.get("submission", {}).get("maximum_backend_jobs") == 1
        and v7_attempt_authority.get("no_relaunch") is True
        and v7_attempt_authority.get("no_resume") is True
        and v7_terminal_audit.get("status") == "TERMINAL_NO_GO_COMPLETE_EVIDENCE"
        and v7_terminal_audit.get("failure_taxonomy") == "EVALUATOR_EXECUTION_FAILURE"
        and v7_terminal_audit.get("evaluator_no_execution") is False
        and v7_terminal_audit.get("first_failed_gate") == "dev.evaluator_gate_application"
        and v7_terminal_audit.get("check_count") == 26
        and all_boolean_checks_pass(v7_terminal_audit)
        and v7_terminal_audit.get("model_quality_conclusion") is None
        and v7_terminal_audit.get("backend", {}).get("job_count") == 1
        and v7_terminal_audit.get("backend", {}).get("job_id")
        == "ace2-v7-full-lifecycle-20260808-attempt-0001--16c53d80"
        and v7_terminal_audit.get("backend", {}).get("job_status") == "failed"
        and v7_terminal_audit.get("checks", {}).get("training.completed") is True
        and v7_terminal_audit.get("checks", {}).get("dev.completed_56_rows") is True
        and v7_terminal_audit.get("checks", {}).get("retention.not_executed") is True
        and v7_terminal_audit.get("checks", {}).get("holdout.not_consumed") is True
        and v7_fresh_review.get("kind") == "round_reviewed_handoff"
        and v7_fresh_review.get("producer_role") == "reviewer"
        and v7_fresh_review.get("mission_id") == "8e30ce003e79"
        and v7_fresh_review.get("round") == 1
        and v7_fresh_review.get("review", {}).get("status") == "done"
        and isinstance(benchmark_v7, dict)
        and benchmark_v7.get("status") == "TERMINAL_NO_GO_FRESH_REVIEW_DONE"
        and benchmark_v7.get("failure_taxonomy") == "EVALUATOR_EXECUTION_FAILURE"
        and benchmark_v7.get("model_quality_conclusion") is None
        and benchmark_v7.get("retention_executed") is False
        and benchmark_v7.get("holdout_accessed") is False
        and benchmark_v7.get("quality_or_downstream_authority_opened") is False
        and benchmark_v7_authority.get("sha256") == sha256(V7_ATTEMPT_AUTHORITY)
        and benchmark_v7_authority.get("consumed") is True
        and benchmark_v7_backend.get("job_count") == 1
        and benchmark_v7_backend.get("job_id")
        == v7_terminal_audit.get("backend", {}).get("job_id")
        and benchmark_v7_audit.get("sha256") == sha256(V7_TERMINAL_AUDIT)
        and benchmark_v7_audit.get("all_checks_pass") is True
        and benchmark_v7_review.get("path") == V7_FRESH_REVIEW.relative_to(ROOT).as_posix()
        and benchmark_v7_review.get("handoff_sha256") == sha256(V7_FRESH_REVIEW)
        and benchmark_v7_review.get("status") == "done"
    )

    benchmark_v8 = benchmark.get("authoritative_v8_terminal_containment", {})
    benchmark_v8_snapshot = (
        local.get("historical_marker_free_v8_preexecution_snapshot", {})
        if isinstance(local, dict)
        else {}
    )
    benchmark_recovery = benchmark.get("historical_disallowed_v7_eval_recovery", {})
    required_v8_checks = {
        "dev_policy_exactly_matches_source_contract",
        "dev_policy_schema_validates",
        "dev_gate_positive_path",
        "dev_gate_all_isolated_negative_paths",
        "legacy_v7_policy_shape_rejected_before_scoring",
    }
    v8_package_freeze_ok = bool(
        all(
            companion_matches(path)
            for path in (
                V8_CONTRACT,
                V8_FREEZE,
                V8_MANIFEST,
                V8_SELF_TEST,
                V8_REVIEW_REQUEST,
                V8_REVIEW_DECISION,
                V8_REVIEW_ACCEPTANCE,
            )
        )
        and v8_contract.get("status") == "FROZEN_MARKER_FREE_PENDING_FRESH_INDEPENDENT_L2"
        and {
            "category_gates",
            "critical_safety_failures_maximum",
            "minimum_hard_passes",
            "response_count",
        }
        <= set(v8_contract.get("evaluator_contract", {}).get("dev", {}))
        and "hard_pass_minimum"
        not in v8_contract.get("evaluator_contract", {}).get("dev", {})
        and "category_minimum_hard_passes"
        not in v8_contract.get("evaluator_contract", {}).get("dev", {})
        and v8_contract.get("future_authorized_lifecycle_order")
        == [
            "full_training_from_pinned_source_model",
            "ordered_synthetic_probe_selector",
            "single_locked_checkpoint_dev_qualification",
            "selected_bf16_materialization",
            "retention",
            "exactly_once_holdout",
        ]
        and v8_contract.get("supersession", {}).get("v7_eval_recovery_permitted") is False
        and v8_contract.get("supersession", {}).get("v7_replay_resume_rescore_or_checkpoint_reuse_permitted") is False
        and v8_freeze.get("status") == "FROZEN_MARKER_FREE_PENDING_FRESH_L2_NO_ATTEMPT_NO_TRAINING"
        and v8_freeze.get("attempt_authority") is False
        and v8_freeze.get("attempt_marker_creation_authorized") is False
        and v8_freeze.get("training_executed") is False
        and v8_manifest.get("tree_sha256")
        == "e7843803e930a9af491fe7e6ee7f9fe0a16fe7eb3b32faecd0eacf74b677ddb7"
        and v8_self_test.get("status") == "PASS"
        and v8_self_test.get("check_count") == 39
        and all_boolean_checks_pass(v8_self_test)
        and required_v8_checks <= set(v8_self_test.get("checks", {}))
        and v8_review_acceptance.get("status")
        == "ACCEPTED_V8_FULL_FINETUNE_EXECUTION_PACKAGE_NO_ATTEMPT"
        and v8_review_acceptance.get("accepted") is True
        and v8_review_acceptance.get("attempt_authority_granted") is False
        and v8_review_acceptance.get("detached_launch_authorized") is False
        and v8_review_acceptance.get("fresh_successor_no_v7_execution_evidence_dependency") is True
        and v8_review_acceptance.get("score_policy_schema") == "ace2_score_gates_v1"
        and v8_review_acceptance.get("execution_tree_sha256") == v8_manifest.get("tree_sha256")
        and v8_review_acceptance.get("contract_sha256") == sha256(V8_CONTRACT)
        and v8_review_acceptance.get("execution_manifest_sha256") == sha256(V8_MANIFEST)
        and isinstance(benchmark_v8_snapshot, dict)
        and benchmark_v8_snapshot.get("candidate_id")
        == "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-product-v8"
        and benchmark_v8_snapshot.get("status")
        == "ACCEPTED_MARKER_FREE_FRESH_SUCCESSOR_PACKAGE_NO_ATTEMPT"
        and benchmark_v8_snapshot.get("execution_manifest", {}).get("tree_sha256")
        == v8_manifest.get("tree_sha256")
        and benchmark_v8_snapshot.get("independent_l2_acceptance", {}).get("sha256")
        == sha256(V8_REVIEW_ACCEPTANCE)
        and benchmark_v8_snapshot.get("attempt_authority_granted") is False
        and benchmark_v8_snapshot.get("official_namespace_exists") is False
        and benchmark_v8_snapshot.get("training_executed") is False
        and benchmark_v8_snapshot.get("official_dev_accessed") is False
        and benchmark_v8_snapshot.get("fresh_successor_no_v7_execution_evidence_dependency") is True
        and benchmark_recovery.get("status")
        == "HISTORICAL_DISALLOWED_BY_LIVE_SUCCESSOR_BOUNDARY"
        and benchmark_recovery.get("permitted_as_active_direction") is False
    )

    v8_authority_count = len(
        list(
            (ROOT / "research/raw/specification").glob(
                "qwen25-05b-instruct-bf16-full-finetune-product-v8-attempt-operator-authority.json"
            )
        )
    )
    v8_intent_count = len(list(V8_LIFECYCLE_ROOT.glob("backend-submission-intent.json")))
    benchmark_v8_authority = (
        benchmark_v8.get("authority", {}) if isinstance(benchmark_v8, dict) else {}
    )
    benchmark_v8_intent = (
        benchmark_v8.get("submission_intent", {}) if isinstance(benchmark_v8, dict) else {}
    )
    benchmark_v8_result = (
        benchmark_v8.get("submission_result", {}) if isinstance(benchmark_v8, dict) else {}
    )
    benchmark_v8_audit = (
        benchmark_v8.get("terminal_audit", {}) if isinstance(benchmark_v8, dict) else {}
    )
    benchmark_v8_review = (
        benchmark_v8.get("fresh_reviewer", {}) if isinstance(benchmark_v8, dict) else {}
    )
    v8_terminal_ok = bool(
        v8_package_freeze_ok
        and all(
            companion_matches(path)
            for path in (
                V8_AUTHORITY,
                V8_INVENTORY,
                V8_INTENT,
                V8_RESULT,
                V8_POST_INVENTORY,
                V8_TERMINAL_AUDIT,
                V8_FRESH_REVIEW,
            )
        )
        and v8_authority_count == 1
        and v8_intent_count == 1
        and v8_authority.get("status") == "AUTHORIZE_SINGLE_V8_FULL_FINETUNE_ATTEMPT"
        and v8_authority.get("attempt") == 1
        and v8_authority.get("execution_tree_sha256")
        == "e7843803e930a9af491fe7e6ee7f9fe0a16fe7eb3b32faecd0eacf74b677ddb7"
        and v8_authority.get("no_relaunch") is True
        and v8_authority.get("no_resume") is True
        and v8_authority.get("submission", {}).get("maximum_backend_jobs") == 1
        and v8_inventory.get("status") == "CLEAR_NO_MATCHING_V8_LIFECYCLE"
        and v8_inventory.get("query_exit_code") == 0
        and v8_inventory.get("matching_experiment_count") == 0
        and v8_inventory.get("matching_job_count") == 0
        and v8_intent.get("status") == "AUTHORIZED_SINGLE_BACKEND_SUBMISSION_INTENT"
        and v8_intent.get("authority_sha256") == sha256(V8_AUTHORITY)
        and v8_result.get("status") == "TERMINAL_SUBMISSION_NO_GO"
        and v8_result.get("client_exit_code") == -15
        and v8_result.get("job_count") == 0
        and v8_result.get("submission_intent_sha256") == sha256(V8_INTENT)
        and v8_post_inventory.get("status") == "POSTSUBMIT_INVENTORY_NO_GO"
        and v8_post_inventory.get("job_count") == 0
        and v8_terminal_audit.get("status") == "TERMINAL_NO_EXECUTION_EVIDENCE_COMPLETE"
        and v8_terminal_audit.get("failure_taxonomy")
        == "BACKEND_CLIENT_INTERACTIVE_PROMPT_NO_EXECUTION"
        and v8_terminal_audit.get("check_count") == 23
        and all_boolean_checks_pass(v8_terminal_audit)
        and v8_terminal_audit.get("model_execution") is False
        and v8_terminal_audit.get("evaluator_no_execution") is True
        and v8_terminal_audit.get("model_quality_conclusion") is None
        and not V8_RUN_ROOT.exists()
        and v8_fresh_review.get("producer_role") == "reviewer"
        and v8_fresh_review.get("mission_id") == "714e985f50df"
        and v8_fresh_review.get("round") == 1
        and v8_fresh_review.get("review", {}).get("status") == "done"
        and v8_fresh_review.get("evidence", {}).get("authority_count") == 1
        and v8_fresh_review.get("evidence", {}).get("submission_intent_count") == 1
        and v8_fresh_review.get("evidence", {}).get("terminal_audit_check_count") == 23
        and v8_fresh_review.get("evidence", {}).get("terminal_audit_failed_check_count") == 0
        and isinstance(benchmark_v8, dict)
        and benchmark_v8.get("status") == "TERMINAL_SUBMISSION_NO_GO"
        and benchmark_v8.get("failure_taxonomy")
        == "BACKEND_CLIENT_INTERACTIVE_PROMPT_NO_EXECUTION"
        and benchmark_v8.get("execution_tree_sha256") == v8_manifest.get("tree_sha256")
        and benchmark_v8_authority.get("count") == 1
        and benchmark_v8_authority.get("sha256") == sha256(V8_AUTHORITY)
        and benchmark_v8_authority.get("consumed") is True
        and benchmark_v8_intent.get("count") == 1
        and benchmark_v8_intent.get("sha256") == sha256(V8_INTENT)
        and benchmark_v8_intent.get("consumed") is True
        and benchmark_v8_result.get("sha256") == sha256(V8_RESULT)
        and benchmark_v8_result.get("client_exit_code") == -15
        and benchmark_v8_result.get("job_count") == 0
        and benchmark_v8_audit.get("sha256") == sha256(V8_TERMINAL_AUDIT)
        and benchmark_v8_audit.get("check_count") == 23
        and benchmark_v8_audit.get("all_checks_pass") is True
        and benchmark_v8_review.get("mission_id") == "714e985f50df"
        and benchmark_v8_review.get("status") == "done"
        and benchmark_v8_review.get("sha256") == sha256(V8_FRESH_REVIEW)
        and benchmark_v8.get("model_execution") is False
        and benchmark_v8.get("evaluator_execution") is False
        and benchmark_v8.get("model_quality_conclusion") is None
        and benchmark_v8.get("retry_relaunch_resume_resubmit_or_second_job_permitted") is False
        and benchmark_v8.get("quality_or_downstream_authority_opened") is False
    )

    benchmark_s1 = benchmark.get("marker_free_bf16_successor_s1_transport", {})
    benchmark_s1_contract = (
        benchmark_s1.get("contract", {}) if isinstance(benchmark_s1, dict) else {}
    )
    benchmark_s1_package_audit = (
        benchmark_s1.get("package_audit", {}) if isinstance(benchmark_s1, dict) else {}
    )
    benchmark_s1_repair_audit = (
        benchmark_s1.get("additive_repair_audit", {})
        if isinstance(benchmark_s1, dict)
        else {}
    )
    benchmark_s1_package_review = (
        benchmark_s1.get("package_fresh_reviewer", {})
        if isinstance(benchmark_s1, dict)
        else {}
    )
    benchmark_s1_terminal_review = (
        benchmark_s1.get("terminal_fresh_reviewer", {})
        if isinstance(benchmark_s1, dict)
        else {}
    )
    benchmark_s1_terminal_audit = (
        benchmark_s1.get("terminal_audit", {})
        if isinstance(benchmark_s1, dict)
        else {}
    )
    benchmark_s1_authority = (
        benchmark_s1.get("authority", {}) if isinstance(benchmark_s1, dict) else {}
    )
    benchmark_s1_intent = (
        benchmark_s1.get("submission_intent", {})
        if isinstance(benchmark_s1, dict)
        else {}
    )
    s1_authority_count = len(list(S1_AUTHORITY.parent.glob(S1_AUTHORITY.name)))
    s1_intent_count = len(list(S1_INTENT.parent.glob("backend-submission-intent.json")))
    s1_ok = bool(
        all(
            companion_matches(path)
            for path in (
                S1_CONTRACT,
                S1_PACKAGE_AUDIT,
                S1_REPAIR_AUDIT,
                S1_FRESH_REVIEW,
                S1_AUTHORITY,
                S1_INTENT,
                S1_RESULT,
                S1_TERMINAL_AUDIT,
                S1_TERMINAL_REVIEW,
            )
        )
        and s1_contract.get("package_id") == S1_PACKAGE_ID
        and s1_contract.get("status") == "MARKER_FREE_PREAUTHORITY_TRANSPORT_PACKAGE"
        and s1_contract.get("authority", {}).get("granted") is False
        and s1_package_audit.get("status") == "PASS_MARKER_FREE_SUCCESSOR_PACKAGE"
        and s1_package_audit.get("check_count") == 36
        and all_boolean_checks_pass(s1_package_audit)
        and s1_package_audit.get("package_identity_sha256") == S1_PACKAGE_IDENTITY
        and s1_repair_audit.get("status") == "PASS_ADDITIVE_S1_V1_REPAIR"
        and s1_repair_audit.get("check_count") == 22
        and all_boolean_checks_pass(s1_repair_audit)
        and s1_repair_audit.get("frozen_v1", {}).get("package_identity_sha256")
        == S1_PACKAGE_IDENTITY
        and s1_repair_audit.get("state_before")
        == {"authority": False, "intent": False, "raw": False, "result": False}
        and s1_repair_audit.get("state_after")
        == {"authority": False, "intent": False, "raw": False, "result": False}
        and s1_fresh_review.get("mission_id") == "ffc5e2b1f181"
        and s1_fresh_review.get("round") == 2
        and s1_fresh_review.get("producer_role") == "reviewer"
        and s1_fresh_review.get("review", {}).get("status") == "done"
        and s1_fresh_review.get("sealed_round_sha256")
        == "d341cc7156b8a736a1633ea7e033f0ec63bfd005ef334547a5d2e2a35dc068c3"
        and s1_authority_count == 1
        and s1_intent_count == 1
        and s1_authority.get("maximum_backend_jobs") == 1
        and s1_authority.get("no_relaunch") is True
        and s1_authority.get("no_replay") is True
        and s1_authority.get("no_resume") is True
        and s1_intent.get("authority_sha256") == sha256(S1_AUTHORITY)
        and s1_result.get("status") == "TERMINAL_SUBMISSION_NO_GO"
        and s1_result.get("client_exit_code") == 0
        and s1_result.get("prompt_detected") is True
        and s1_result.get("submission_intent_sha256") == sha256(S1_INTENT)
        and s1_result.get("raw_output_sha256") == sha256(S1_RAW)
        and s1_terminal_audit.get("status")
        == "TERMINAL_S1_TRANSPORT_EVIDENCE_COMPLETE"
        and s1_terminal_audit.get("failure_taxonomy")
        == "INERT_TRANSPORT_GATE_EXECUTED_TERMINAL_FAILED"
        and s1_terminal_audit.get("check_count") == 34
        and s1_terminal_audit.get("failed_check_count") == 0
        and all_boolean_checks_pass(s1_terminal_audit)
        and s1_terminal_audit.get("cardinality")
        == {
            "authority_count": 1,
            "backend_experiment_count": 1,
            "backend_job_count": 1,
            "intent_count": 1,
            "retry_count": 0,
            "submit_invocation_count": 1,
        }
        and s1_terminal_audit.get("backend", {}).get("job_id")
        == "ace2-bf16-successor-s1-transport-20260808-pre-0e51f05b"
        and s1_terminal_audit.get("backend", {}).get("status") == "failed"
        and s1_terminal_audit.get("download", {}).get("stdout", {}).get("sha256")
        == sha256(S1_TERMINAL_STDOUT)
        and not S1_RUN_ROOT.exists()
        and s1_terminal_review.get("mission_id")
        == "s1-transport-terminal-backend-evidence-review-v1"
        and s1_terminal_review.get("review", {}).get("status") == "done"
        and isinstance(benchmark_s1, dict)
        and benchmark_s1.get("package_id") == S1_PACKAGE_ID
        and benchmark_s1.get("status")
        == "TERMINAL_TRANSPORT_GATE_EXECUTED_FAILED_FRESH_REVIEW_DONE"
        and benchmark_s1.get("package_identity_sha256") == S1_PACKAGE_IDENTITY
        and benchmark_s1_contract.get("sha256") == sha256(S1_CONTRACT)
        and benchmark_s1_package_audit.get("sha256") == sha256(S1_PACKAGE_AUDIT)
        and benchmark_s1_package_audit.get("check_count") == 36
        and benchmark_s1_repair_audit.get("sha256") == sha256(S1_REPAIR_AUDIT)
        and benchmark_s1_repair_audit.get("check_count") == 22
        and benchmark_s1_package_review.get("sha256") == sha256(S1_FRESH_REVIEW)
        and benchmark_s1_package_review.get("mission_id") == "ffc5e2b1f181"
        and benchmark_s1_package_review.get("status") == "done"
        and benchmark_s1_terminal_review.get("sha256") == sha256(S1_TERMINAL_REVIEW)
        and benchmark_s1_terminal_review.get("mission_id")
        == "s1-transport-terminal-backend-evidence-review-v1"
        and benchmark_s1_terminal_review.get("status") == "done"
        and benchmark_s1_terminal_audit.get("sha256") == sha256(S1_TERMINAL_AUDIT)
        and benchmark_s1_terminal_audit.get("check_count") == 34
        and benchmark_s1_terminal_audit.get("failed_check_count") == 0
        and benchmark_s1_authority.get("count") == 1
        and benchmark_s1_authority.get("sha256") == sha256(S1_AUTHORITY)
        and benchmark_s1_authority.get("consumed") is True
        and benchmark_s1_intent.get("count") == 1
        and benchmark_s1_intent.get("sha256") == sha256(S1_INTENT)
        and benchmark_s1_intent.get("consumed") is True
        and benchmark_s1.get("authority_granted") is True
        and benchmark_s1.get("authority_consumed") is True
        and benchmark_s1.get("submission_intent_exists") is True
        and benchmark_s1.get("backend_experiment_exists") is True
        and benchmark_s1.get("backend_job_exists") is True
        and benchmark_s1.get("backend_job_terminal") is True
        and benchmark_s1.get("backend_job_status") == "failed"
        and benchmark_s1.get("retry_count") == 0
        and benchmark_s1.get("transport_payload_executed") is True
        and benchmark_s1.get("model_execution") is False
        and benchmark_s1.get("evaluator_execution") is False
        and benchmark_s1.get("model_quality_conclusion") is None
        and benchmark_s1.get("retry_relaunch_resume_cancel_or_second_job_permitted")
        is False
        and benchmark_s1.get("quality_or_downstream_authority_opened") is False
    )

    benchmark_s3_terminal = benchmark.get("authoritative_s3_terminal_containment", {})
    benchmark_s3_terminal_spec = benchmark.get("specification_stage_evidence", {}).get(
        "authoritative_s3_terminal_containment", {}
    )
    s3_terminal_ok = bool(
        companion_matches(S3_TERMINAL_AUDIT)
        and companion_matches(S3_TERMINAL_REVIEW)
        and s3_terminal_audit.get("status") == "TERMINAL_S3_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE"
        and s3_terminal_audit.get("check_count") == 64
        and all_boolean_checks_pass(s3_terminal_audit)
        and s3_terminal_audit.get("failure_taxonomy") == "PREATTEMPT_SOURCE_SNAPSHOT_AUDIT_OMITTED_FROM_STAGE"
        and s3_terminal_audit.get("cardinality", {}).get("authority_count") == 1
        and s3_terminal_audit.get("cardinality", {}).get("intent_count") == 1
        and s3_terminal_audit.get("cardinality", {}).get("backend_experiment_count") == 1
        and s3_terminal_audit.get("cardinality", {}).get("backend_job_count") == 1
        and s3_terminal_audit.get("cardinality", {}).get("retry_count") == 0
        and s3_terminal_audit.get("execution_boundaries", {}).get("attempt_marker_created") is False
        and s3_terminal_audit.get("execution_boundaries", {}).get("model_executed") is False
        and s3_terminal_audit.get("execution_boundaries", {}).get("evaluator_executed") is False
        and s3_terminal_audit.get("execution_boundaries", {}).get("quality_conclusion") is None
        and s3_terminal_review.get("review", {}).get("status") == "done"
        and S3_AUTHORITY.exists()
        and S3_SUBMISSION_INTENT.exists()
        and S3_SUBMISSION_RESULT.exists()
        and S3_SUBMISSION_RAW.exists()
        and not S3_RUN_ROOT.exists()
        and benchmark_s3_terminal.get("package_identity_sha256") == S3_PACKAGE_IDENTITY
        and benchmark_s3_terminal.get("status") == "TERMINAL_S3_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE"
        and benchmark_s3_terminal.get("failure_taxonomy") == "PREATTEMPT_SOURCE_SNAPSHOT_AUDIT_OMITTED_FROM_STAGE"
        and benchmark_s3_terminal.get("authority_count") == 1
        and benchmark_s3_terminal.get("backend_job_count") == 1
        and benchmark_s3_terminal.get("retry_count") == 0
        and benchmark_s3_terminal.get("retry_resume_relaunch_repair_or_attempt_0002_allowed") is False
        and benchmark_s3_terminal_spec.get("status") == "TERMINAL_S3_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE"
        and benchmark_s3_terminal_spec.get("retry_or_repair_allowed") is False
    )

    benchmark_s4 = benchmark.get("authoritative_s4_terminal_containment", {})
    benchmark_s4_spec = benchmark.get("specification_stage_evidence", {}).get(
        "authoritative_s4_terminal_containment", {}
    )
    s4_review_reason = str(s4_terminal_review.get("review", {}).get("reason", "")).lower()
    s4_required_rulings = (
        "s4.cardinality: supported",
        "s4.backend-terminal: supported",
        "s4.download-binding: supported",
        "s4.preattempt-interlock: supported",
        "s4.binding-root-cause: supported",
        "s4.evaluator-no-execution: supported",
        "s4.hygiene-complete: supported",
        "s4.no-retry-downstream: supported",
        "s4.checkpoint-current: supported",
    )
    s4_expected_empty_state = {
        "authority": False,
        "intent": False,
        "namespace": False,
        "raw": False,
        "result": False,
    }
    s4_ok = bool(
        all(
            companion_matches(path)
            for path in (
                S4_CONTRACT,
                S4_PACKAGE_AUDIT,
                S4_MANIFEST,
                S4_SELF_TEST,
                S4_L2_ACCEPTANCE,
                S4_AUTHORITY,
                S4_TERMINAL_AUDIT,
                S4_TERMINAL_REVIEW,
            )
        )
        and s4_contract.get("contract_id") == S4_PACKAGE_ID
        and s4_contract.get("package_id") == S4_PACKAGE_ID
        and s4_contract.get("transport_stage", {}).get("required_source_audit", {}).get("sha256")
        == "5ea75b61105ff5ce30803c5c7b33e2bde36acb0511ffddc2da55303909cf657d"
        and s4_manifest.get("tree_sha256") == S4_EXECUTION_TREE
        and s4_self_test.get("status") == "PASS"
        and s4_self_test.get("check_count") == 39
        and all_boolean_checks_pass(s4_self_test)
        and s4_self_test.get("execution_tree_sha256") == S4_EXECUTION_TREE
        and s4_package_audit.get("status") == "PASS_MARKER_FREE_FULL_TRAINING_SUCCESSOR_S4_PACKAGE"
        and s4_package_audit.get("check_count") == 84
        and all_boolean_checks_pass(s4_package_audit)
        and s4_package_audit.get("failed_checks") == []
        and s4_package_audit.get("package_identity_sha256") == S4_PACKAGE_IDENTITY
        and s4_package_audit.get("runner", {}).get("execution_tree_sha256") == S4_EXECUTION_TREE
        and s4_package_audit.get("inventory", {}).get("matching_experiment_count") == 0
        and s4_package_audit.get("inventory", {}).get("matching_job_count") == 0
        and s4_package_audit.get("state_before") == s4_expected_empty_state
        and s4_package_audit.get("state_after") == s4_expected_empty_state
        and s4_package_audit.get("hygiene", {}).get("file_count") == 57
        and s4_package_audit.get("hygiene", {}).get("bytes_scanned") == 715486
        and s4_package_audit.get("hygiene", {}).get("skipped_file_count") == 0
        and s4_package_audit.get("hygiene", {}).get("findings") == []
        and s4_package_audit.get("stage_closure", {}).get("required_file_count") == 63
        and len(s4_package_audit.get("stage_closure", {}).get("omission_rejections", {})) == 63
        and s4_l2_acceptance.get("accepted") is True
        and s4_l2_acceptance.get("status") == "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S4_PACKAGE_NO_ATTEMPT"
        and s4_l2_acceptance.get("package_identity_sha256") == S4_PACKAGE_IDENTITY
        and s4_l2_acceptance.get("execution_tree_sha256") == S4_EXECUTION_TREE
        and s4_l2_acceptance.get("contract_sha256") == sha256(S4_CONTRACT)
        and s4_l2_acceptance.get("attempt_authority_granted") is False
        and s4_l2_acceptance.get("detached_launch_authorized") is False
        and s4_l2_acceptance.get("reviewer_status") == "done"
        and S4_AUTHORITY.exists()
        and (S4_SUBMISSION_ROOT / "backend-submission-intent.json").exists()
        and (S4_SUBMISSION_ROOT / "backend-submission-result.json").exists()
        and (S4_SUBMISSION_ROOT / "backend-submission.raw.txt").exists()
        and not S4_RUN_ROOT.exists()
        and s4_terminal_audit.get("status") == "TERMINAL_S4_PREATTEMPT_NO_GO_EVIDENCE_COMPLETE"
        and s4_terminal_audit.get("failure_taxonomy") == "PREATTEMPT_RUNTIME_CHECKSUM_SIDECARS_OMITTED_FROM_STAGE"
        and s4_terminal_audit.get("check_count") == 62
        and s4_terminal_audit.get("failed_check_count") == 0
        and all(s4_terminal_audit.get("checks", {}).values())
        and s4_terminal_audit.get("cardinality") == {
            "authority_count": 1,
            "intent_count": 1,
            "submit_invocation_count": 1,
            "backend_experiment_count": 1,
            "backend_job_count": 1,
            "retry_count": 0,
        }
        and s4_terminal_audit.get("backend", {}).get("job_id")
        == "ace2-bf16-full-finetune-successor-s4-20260808-a70f167d"
        and s4_terminal_audit.get("backend", {}).get("status") == "failed"
        and s4_terminal_audit.get("failure", {}).get("first_failed_gate") == "frozen_bindings_exact"
        and len(s4_terminal_audit.get("failure", {}).get("runtime_required_missing_sidecars", [])) == 3
        and s4_terminal_audit.get("execution_boundaries", {}).get("attempt_marker_created") is False
        and s4_terminal_audit.get("execution_boundaries", {}).get("model_executed") is False
        and s4_terminal_audit.get("execution_boundaries", {}).get("evaluator_executed") is False
        and s4_terminal_audit.get("execution_boundaries", {}).get("downstream_authority_opened") is False
        and s4_terminal_review.get("review", {}).get("status") == "done"
        and all(ruling in s4_review_reason for ruling in s4_required_rulings)
        and benchmark.get("contract_status")
        == "specification_checklist_complete_ds32_csr_address_reconciled_v1_unapplied_s7_v4f_terminal_before_spawn_authority_consumed_no_reservation_success_v4_terminal_one_start_child_nonzero_fresh_l2_no_replay_v3_terminal_no_execution_zero_runtime_start_supervisor_python310_repair_pass_terminal_consumed_instruct_diagnostic_orchestration_failure_s7_package_and_custodian_fresh_review_accepted_model_attempt_absent_downstream_blocked_no_checkpoint_no_w4a8_no_u280_toolchain"
        and benchmark.get("specification_stage_evidence", {}).get("status")
        == "PASS_CURRENT_PRODUCTIZATION_SPECIFICATION_CLOSURE"
        and benchmark.get("specification_stage_evidence", {})
        .get("benchmark_interface_closure", {})
        .get("closure")
        == "external_non_benchmark_with_fresh_review_bound_terminal_s7_v4f_before_spawn_zero_candidate_invocations_no_reservation_success_terminal_v4_one_start_child_nonzero_no_replay_terminal_consumed_instruct_diagnostic_orchestration_failure_terminal_s6_and_nonconsuming_s7_model_package_custodian_pass"
        and benchmark_s4.get("package_id") == S4_PACKAGE_ID
        and benchmark_s4.get("package_identity_sha256") == S4_PACKAGE_IDENTITY
        and benchmark_s4.get("execution_manifest", {}).get("tree_sha256") == S4_EXECUTION_TREE
        and benchmark_s4.get("package_audit", {}).get("sha256") == sha256(S4_PACKAGE_AUDIT)
        and benchmark_s4.get("fresh_l2_acceptance", {}).get("sha256") == sha256(S4_L2_ACCEPTANCE)
        and benchmark_s4.get("terminal_audit", {}).get("sha256") == sha256(S4_TERMINAL_AUDIT)
        and benchmark_s4.get("terminal_fresh_review", {}).get("sha256") == sha256(S4_TERMINAL_REVIEW)
        and benchmark_s4.get("authority_count") == 1
        and benchmark_s4.get("backend_job_count") == 1
        and benchmark_s4.get("retry_count") == 0
        and benchmark_s4.get("quality_or_downstream_authority_opened") is False
        and benchmark_s4_spec.get("package_identity_sha256") == S4_PACKAGE_IDENTITY
        and benchmark_s4_spec.get("execution_tree_sha256") == S4_EXECUTION_TREE
        and benchmark_s4_spec.get("terminal_audit_sha256") == sha256(S4_TERMINAL_AUDIT)
        and benchmark_s4_spec.get("terminal_fresh_review_sha256") == sha256(S4_TERMINAL_REVIEW)
        and benchmark_s4_spec.get("attempt_authority_granted") is True
        and benchmark_s4_spec.get("backend_job_count") == 1
        and benchmark_s4_spec.get("retry_count") == 0
        and benchmark_s4_spec.get("quality_or_downstream_authority_opened") is False
    )

    s6_terminal_ok = bool(
        companion_matches(S6_AUTHORITY)
        and companion_matches(S6_SUBMISSION_INTENT)
        and companion_matches(S6_SUBMISSION_RESULT)
        and companion_matches(S6_TERMINAL_AUDIT)
        and companion_matches(LATEST)
        and sha256(S6_AUTHORITY) == S6_AUTHORITY_SHA256
        and sha256(S6_SUBMISSION_INTENT) == S6_INTENT_SHA256
        and sha256(S6_SUBMISSION_RESULT) == S6_RESULT_SHA256
        and sha256(S6_TERMINAL_AUDIT) == S6_TERMINAL_AUDIT_SHA256
        and sha256(LATEST) == S6_LATEST_SHA256
        and s6_authority.get("attempt") == 1
        and s6_authority.get("maximum_attempts") == 1
        and s6_authority.get("execution_contract_sha256") == S6_CONTRACT_SHA256
        and s6_authority.get("execution_tree_sha256") == S6_EXECUTION_TREE
        and s6_submission_intent.get("authority_sha256") == S6_AUTHORITY_SHA256
        and s6_submission_intent.get("status")
        == "AUTHORIZED_SINGLE_BACKEND_SUBMISSION_INTENT"
        and s6_submission_result.get("authority_sha256") == S6_AUTHORITY_SHA256
        and s6_submission_result.get("submission_intent_sha256") == S6_INTENT_SHA256
        and s6_terminal_audit.get("status") == "TERMINAL_NO_GO"
        and s6_terminal_audit.get("failure_taxonomy") == "PROBE_QUALITY_GATE_FAILURE"
        and len(s6_terminal_audit.get("checks", {})) == 32
        and all_boolean_checks_pass(s6_terminal_audit)
        and s6_terminal_audit.get("probe_quality", {}).get("selected_checkpoint") is None
        and s6_terminal_audit.get("probe_quality", {}).get("official_dev_accessed") is False
        and s6_terminal_audit.get("execution_boundaries", {}).get("retention_executed") is False
        and s6_terminal_audit.get("execution_boundaries", {}).get("holdout_executed") is False
        and s6_terminal_audit.get("execution_boundaries", {}).get("w4a8_executed") is False
        and s6_terminal_audit.get("execution_boundaries", {}).get("rtl_retarget_executed") is False
        and s6_terminal_audit.get("execution_boundaries", {}).get("fpga_executed") is False
        and s6_terminal_audit.get("execution_boundaries", {}).get("u280_executed") is False
        and s6_terminal_audit.get("execution_boundaries", {}).get("downstream_authority_opened") is False
        and latest.get("status") == "TERMINAL_NO_GO"
        and latest.get("failure_taxonomy") == "PROBE_QUALITY_GATE_FAILURE"
        and latest.get("s6_terminal", {}).get("audit", {}).get("sha256")
        == S6_TERMINAL_AUDIT_SHA256
        and latest.get("independent_review", {}).get("mission_id") == "8d0ffc1dfcb9"
        and latest.get("independent_review", {}).get("status") == "ACCEPTED"
        and latest.get("seal", {}).get("retry") is False
        and latest.get("seal", {}).get("successor_authority_created") is False
    )

    s7_no_downstream = s7_specification.get("no_downstream_state", {})
    s7_review = s7_fresh_review_binding.get("review", {})
    s7_accepted = s7_fresh_review_binding.get("accepted_package", {})
    s7_specification_ok = bool(
        companion_matches(S7_SPECIFICATION)
        and companion_matches(S7_MANIFEST)
        and companion_matches(S7_FRESH_REVIEW_BINDING)
        and sha256(S7_SPECIFICATION) == S7_SPECIFICATION_SHA256
        and sha256(S7_MANIFEST) == S7_MANIFEST_SHA256
        and sha256(S7_FRESH_REVIEW_BINDING) == S7_REVIEW_BINDING_SHA256
        and s7_specification.get("stage") == "specification"
        and s7_specification.get("status")
        == "FROZEN_CANDIDATE_PACKAGE_PENDING_FRESH_L2_NO_AUTHORITY"
        and isinstance(s7_no_downstream, dict)
        and bool(s7_no_downstream)
        and all(value is False for value in s7_no_downstream.values())
        and s7_manifest.get("package_id")
        == "qwen2.5-0.5b-instruct-ace2-bf16-successor-s7-design-package-v1"
        and s7_manifest.get("consuming_state_created") is False
        and s7_review.get("mission_id") == "design-distinct-bf16-successor-s7-v1"
        and s7_review.get("producer_role") == "reviewer"
        and s7_review.get("round") == 3
        and s7_review.get("status") == "done"
        and s7_accepted.get("specification_sha256") == S7_SPECIFICATION_SHA256
        and s7_accepted.get("manifest_sha256") == S7_MANIFEST_SHA256
        and s7_fresh_review_binding.get("source_handoff", {}).get("sha256")
        == S7_REVIEW_HANDOFF_SHA256
        and s7_fresh_review_binding.get("acceptance_effect")
        == s7_specification.get("fresh_l2_review_contract", {}).get("acceptance_effect")
    )

    s7_consuming_paths = (
        S7_ATTEMPT_MARKER,
        S7_ATTEMPT_START_SEAL,
        S7_GPU_PREFLIGHT,
        S7_SUBMISSION_INTENT,
        S7_CANDIDATE,
        S7_PROBE_RESULTS,
    )
    s7_execution_package_ok = bool(
        sha256(S7_EXECUTION_AUTHORITY) == S7_EXECUTION_AUTHORITY_SHA256
        and companion_matches(S7_EXECUTION_MANIFEST)
        and companion_matches(S7_EXECUTION_AUDIT)
        and companion_matches(S7_EXECUTION_FRESH_REVIEW)
        and sha256(S7_EXECUTION_MANIFEST) == S7_EXECUTION_MANIFEST_SHA256
        and sha256(S7_EXECUTION_AUDIT) == S7_EXECUTION_AUDIT_SHA256
        and sha256(S7_EXECUTION_FRESH_REVIEW) == S7_EXECUTION_FRESH_REVIEW_SHA256
        and sha256(S7_EXECUTION_RUNNER) == S7_EXECUTION_RUNNER_SHA256
        and sha256(S7_NODE1_LAUNCH) == S7_NODE1_LAUNCH_SHA256
        and sha256(S7_ACCEPTED_BASE_MANIFEST) == S7_ACCEPTED_BASE_MANIFEST_SHA256
        and sha256(S7_ACCEPTED_BASE_RUNNER) == S7_ACCEPTED_BASE_RUNNER_SHA256
        and sha256(S7_DATA_MANIFEST) == S7_DATA_MANIFEST_SHA256
        and package_manifest_artifacts_match(s7_execution_manifest, 74)
        and s7_execution_manifest.get("continuation_id") == "c5fd93463239"
        and s7_execution_manifest.get("package_id")
        == "qwen2.5-0.5b-instruct-ace2-bf16-successor-s7-data-probe-execution-v1"
        and s7_execution_manifest.get("counts")
        == {
            "dpo_pairs": 256,
            "probe_candidates": 144,
            "selected_probes": 56,
            "sft_rows": 448,
        }
        and s7_execution_manifest.get("attempt_marker_created") is False
        and s7_execution_manifest.get("execution_or_evaluation_performed") is False
        and s7_execution_manifest.get("submission_or_job_created") is False
        and s7_execution_manifest.get("protected_outputs_accessed") is False
        and s7_execution_manifest.get("lineage", {}).get(
            "accepted_base_package_manifest_sha256"
        )
        == S7_ACCEPTED_BASE_MANIFEST_SHA256
        and s7_execution_manifest.get("lineage", {}).get(
            "accepted_custodian_fresh_review_binding_sha256"
        )
        == S7_CUSTODIAN_REVIEW_BINDING_SHA256
        and s7_execution_audit.get("result") == "PASS"
        and s7_execution_audit.get("audit_scope")
        == "STATIC_LOCAL_NONCONSUMING_S7_PACKAGE_AUDIT"
        and s7_execution_audit.get("package_manifest_sha256")
        == S7_EXECUTION_MANIFEST_SHA256
        and s7_execution_fresh_review.get("status")
        == "ACCEPTED_S7_PRESEAL_COMPATIBILITY_REPAIR_NO_ATTEMPT"
        and s7_execution_fresh_review.get("package_manifest_sha256")
        == S7_EXECUTION_MANIFEST_SHA256
        and s7_execution_fresh_review.get("runner_sha256")
        == S7_EXECUTION_RUNNER_SHA256
        and s7_execution_fresh_review.get("node1_launch_sha256")
        == S7_NODE1_LAUNCH_SHA256
        and s7_execution_fresh_review.get("custodian_review_binding_sha256")
        == S7_CUSTODIAN_REVIEW_BINDING_SHA256
        and s7_execution_fresh_review.get("accepted_base_package_manifest_sha256")
        == S7_ACCEPTED_BASE_MANIFEST_SHA256
        and s7_execution_fresh_review.get("authority_expansion_allowed") is False
        and not any(path.exists() for path in s7_consuming_paths)
    )

    s7_custodian_frozen = s7_custodian_attestation.get("frozen_s7_hashes", {})
    s7_custodian_self_binding = s7_custodian_attestation.get("self_binding", {})
    s7_custodian_aggregate = s7_custodian_attestation.get("aggregate_counts", {})
    s7_custodian_bound_attestation = s7_custodian_review_binding.get(
        "attestation", {}
    )
    s7_custodian_bound_manifests = s7_custodian_review_binding.get(
        "frozen_bindings", {}
    )
    s7_custodian_review = s7_custodian_review_binding.get(
        "independent_review", {}
    )
    s7_custodian_verification = s7_custodian_review_binding.get(
        "verification", {}
    )
    s7_custodian_effect = s7_custodian_review_binding.get(
        "acceptance_effect", {}
    )
    s7_custodian_ok = bool(
        S7_CUSTODIAN_ATTESTATION.is_file()
        and S7_CUSTODIAN_ATTESTATION.stat().st_size == 2974
        and sha256(S7_CUSTODIAN_ATTESTATION) == S7_CUSTODIAN_ATTESTATION_SHA256
        and companion_matches(S7_CUSTODIAN_REVIEW_BINDING)
        and sha256(S7_CUSTODIAN_REVIEW_BINDING)
        == S7_CUSTODIAN_REVIEW_BINDING_SHA256
        and s7_custodian_attestation.get("schema_id")
        == "ace2-opaque-s7-s6-disjointness-custodian-attestation"
        and s7_custodian_attestation.get("schema_version") == 1
        and s7_custodian_attestation.get("producer_role") == "evidence_custodian"
        and s7_custodian_attestation.get("status") == "PASS"
        and isinstance(s7_custodian_frozen, dict)
        and s7_custodian_frozen.get("data_manifest_sha256")
        == S7_DATA_MANIFEST_SHA256
        and s7_custodian_frozen.get("accepted_package_manifest_sha256")
        == S7_ACCEPTED_BASE_MANIFEST_SHA256
        and isinstance(s7_custodian_self_binding, dict)
        and s7_custodian_self_binding.get("value")
        == canonical_hash_without_self_binding(s7_custodian_attestation)
        and isinstance(s7_custodian_aggregate, dict)
        and s7_custodian_aggregate.get("input_manifests_verified") == 11
        and s7_custodian_aggregate.get("required_s6_partitions") == 5
        and s7_custodian_aggregate.get("covered_s6_partitions") == 5
        and s7_custodian_aggregate.get("canonical_hash_comparisons") == 94097040
        and s7_custodian_aggregate.get("five_gram_signature_comparisons")
        == 94097040
        and s7_custodian_aggregate.get("exact_match_count") == 0
        and s7_custodian_aggregate.get("threshold_match_count") == 0
        and isinstance(s7_custodian_bound_attestation, dict)
        and s7_custodian_bound_attestation.get("sha256")
        == S7_CUSTODIAN_ATTESTATION_SHA256
        and s7_custodian_bound_attestation.get("bytes") == 2974
        and s7_custodian_bound_attestation.get("status") == "PASS"
        and isinstance(s7_custodian_bound_manifests, dict)
        and s7_custodian_bound_manifests.get("data_manifest_sha256")
        == S7_DATA_MANIFEST_SHA256
        and s7_custodian_bound_manifests.get("accepted_package_manifest_sha256")
        == S7_ACCEPTED_BASE_MANIFEST_SHA256
        and isinstance(s7_custodian_review, dict)
        and s7_custodian_review.get("mission_id") == "544ba67a61ed"
        and s7_custodian_review.get("round") == 1
        and s7_custodian_review.get("producer_role") == "reviewer"
        and s7_custodian_review.get("status") == "done"
        and s7_custodian_review.get("source_sha256")
        == S7_CUSTODIAN_REVIEW_HANDOFF_SHA256
        and isinstance(s7_custodian_verification, dict)
        and s7_custodian_verification.get("result") == "PASS"
        and s7_custodian_verification.get("check_count") == 32
        and s7_custodian_verification.get("failed_check_count") == 0
        and isinstance(s7_custodian_effect, dict)
        and s7_custodian_effect.get("custodian_prerequisite_satisfied") is True
        and s7_custodian_effect.get("consuming_execution_authority_granted")
        is False
        and s7_custodian_effect.get("attempt_marker_authorized") is False
        and s7_custodian_effect.get("stage_advance_authorized") is False
        and s7_custodian_effect.get("rtl_or_u280_authorized") is False
        and not any(path.exists() for path in s7_consuming_paths)
    )

    required_control_plane_fields = [
        "amlt_project_applicability_or_exact_project",
        "subscription",
        "resource_group",
        "azureml_workspace",
        "backend_job_name_or_id",
        "expected_owner_binding",
    ]
    control_decision = s7_control_plane_evidence.get("decision", {})
    control_prerequisite = s7_control_plane_evidence.get(
        "sole_infrastructure_prerequisite", {}
    )
    control_scope = s7_control_plane_evidence.get("scope_accounting", {})
    control_review_scope = s7_control_plane_review_binding.get(
        "accepted_scope", {}
    )
    control_review_effect = s7_control_plane_review_binding.get(
        "acceptance_effect", {}
    )
    s7_control_plane_ok = bool(
        companion_matches(S7_CONTROL_PLANE_EVIDENCE)
        and companion_matches(S7_CONTROL_PLANE_REVIEW_BINDING)
        and sha256(S7_CONTROL_PLANE_EVIDENCE)
        == S7_CONTROL_PLANE_EVIDENCE_SHA256
        and sha256(S7_CONTROL_PLANE_REVIEW_BINDING)
        == S7_CONTROL_PLANE_REVIEW_BINDING_SHA256
        and s7_control_plane_evidence.get("classification")
        == "NO_SAFE_QUERY_BINDING"
        and control_decision.get("binding_complete") is False
        and control_decision.get("authoritative_control_plane_query_count") == 0
        and control_decision.get("network_access_performed") is False
        and control_prerequisite.get("missing_fields")
        == required_control_plane_fields
        and control_scope.get("amlt_local_lookup_performed") is False
        and control_scope.get("ssh_or_websocket_used") is False
        and control_scope.get("gpu_or_s7_execution_performed") is False
        and s7_control_plane_review_binding.get("mission_id")
        == "refresh-s7-owned-reservation-control-plane-binding-v1"
        and s7_control_plane_review_binding.get("review_round") == 1
        and s7_control_plane_review_binding.get("producer_role") == "reviewer"
        and s7_control_plane_review_binding.get("status") == "done"
        and s7_control_plane_review_binding.get("source_handoff_sha256")
        == S7_CONTROL_PLANE_REVIEW_HANDOFF_SHA256
        and control_review_scope.get("classification")
        == "NO_SAFE_QUERY_BINDING"
        and control_review_scope.get("binding_complete") is False
        and control_review_scope.get("experiment") == "a100s-idle"
        and control_review_scope.get("reservation_job_label")
        == "gpu4x2-a100-idle"
        and control_review_scope.get("reservation_job_label_is_backend_job_id")
        is False
        and control_review_scope.get("missing_exact_fields")
        == required_control_plane_fields
        and control_review_scope.get("authoritative_control_plane_query_count")
        == 0
        and control_review_effect.get("control_plane_query_authorized") is False
        and control_review_effect.get(
            "missing_value_inference_or_substitution_authorized"
        )
        is False
        and control_review_effect.get("s7_execution_authorized") is False
        and control_review_effect.get("stage_transition_authorized") is False
    )

    s7_v4f_identities = s7_v4f_terminal_binding.get(
        "frozen_execution_identities", {}
    )
    s7_v4f_state = s7_v4f_terminal_binding.get("durable_terminal_state", {})
    s7_v4f_launcher_outcome = s7_v4f_terminal_binding.get(
        "launcher_outcome", {}
    )
    s7_v4f_review = s7_v4f_terminal_binding.get("fresh_l2_review", {})
    s7_v4f_claim_boundary = s7_v4f_terminal_binding.get("claim_boundary", {})
    s7_v4f_review_reason = s7_v4f_review_handoff.get("review", {}).get(
        "reason", ""
    )
    s7_v4f_terminal_ok = bool(
        companion_matches(S7_V4F_TERMINAL_BINDING)
        and sha256(S7_V4F_TERMINAL_BINDING) == S7_V4F_TERMINAL_BINDING_SHA256
        and s7_v4f_terminal_binding.get("artifact_kind")
        == "ace2_s7_v4f_terminal_before_spawn_fresh_l2_binding"
        and s7_v4f_terminal_binding.get("schema_version") == 1
        and s7_v4f_terminal_binding.get("mission_id")
        == "submit-fresh-owned-s7-reservation-once-v4f"
        and s7_v4f_terminal_binding.get("classification")
        == "TERMINAL_PRE_CANDIDATE_SPAWN_FAILURE_NO_RESERVATION_SUCCESS"
        and s7_v4f_identities.get("authorization_source_sha256")
        == S7_V4F_AUTHORIZATION_SHA256
        and s7_v4f_identities.get("launcher_sha256")
        == S7_V4F_LAUNCHER_SHA256
        and s7_v4f_identities.get("launcher_manifest_sha256")
        == S7_V4F_LAUNCHER_MANIFEST_SHA256
        and s7_v4f_identities.get("executor_sha256")
        == S7_V4F_EXECUTOR_SHA256
        and s7_v4f_identities.get("execution_contract_sha256")
        == S7_V4F_EXECUTION_CONTRACT_SHA256
        and s7_v4f_identities.get("bundle_manifest_sha256")
        == S7_V4F_BUNDLE_MANIFEST_SHA256
        and sha256(S7_V4F_LAUNCHER) == S7_V4F_LAUNCHER_SHA256
        and sha256(S7_V4F_LAUNCHER_MANIFEST)
        == S7_V4F_LAUNCHER_MANIFEST_SHA256
        and sha256(S7_V4F_EXECUTOR) == S7_V4F_EXECUTOR_SHA256
        and sha256(S7_V4F_EXECUTION_CONTRACT)
        == S7_V4F_EXECUTION_CONTRACT_SHA256
        and sha256(S7_V4F_BUNDLE_MANIFEST) == S7_V4F_BUNDLE_MANIFEST_SHA256
        and S7_V4F_AUTHORIZATION_TARGET.stat().st_size == 919
        and sha256(S7_V4F_AUTHORIZATION_TARGET) == S7_V4F_AUTHORIZATION_SHA256
        and S7_V4F_ATTEMPT_CONSUMED.stat().st_size == 467
        and sha256(S7_V4F_ATTEMPT_CONSUMED)
        == S7_V4F_ATTEMPT_CONSUMED_SHA256
        and S7_V4F_EXECUTION_ATTEMPT.stat().st_size == 261
        and sha256(S7_V4F_EXECUTION_ATTEMPT)
        == S7_V4F_EXECUTION_ATTEMPT_SHA256
        and S7_V4F_EXECUTION_RESULT.stat().st_size == 48
        and sha256(S7_V4F_EXECUTION_RESULT) == S7_V4F_EXECUTION_RESULT_SHA256
        and s7_v4f_execution_attempt.get("artifact_kind")
        == "s7_v4f_execution_attempt_sentinel"
        and s7_v4f_execution_attempt.get("submission_process_invocation_count")
        == 0
        and s7_v4f_execution_attempt.get("submission_process_started") is False
        and s7_v4f_execution_attempt.get("replay_forbidden") is True
        and s7_v4f_execution_result
        == {"allowlisted_phase_evidence": ["before_spawn"]}
        and s7_v4f_launcher_outcome.get("launcher_invocation_count") == 1
        and s7_v4f_launcher_outcome.get("launcher_exit_code") == 1
        and s7_v4f_launcher_outcome.get("candidate_process_invocation_count")
        == 0
        and s7_v4f_launcher_outcome.get("candidate_process_started") is False
        and s7_v4f_launcher_outcome.get("reservation_success_established")
        is False
        and s7_v4f_launcher_outcome.get("backend_identity_established") is False
        and s7_v4f_review.get("sha256") == S7_V4F_REVIEW_HANDOFF_SHA256
        and s7_v4f_review.get("producer_role") == "reviewer"
        and s7_v4f_review.get("round") == 1
        and s7_v4f_review.get("status") == "done"
        and sha256(S7_V4F_REVIEW_HANDOFF) == S7_V4F_REVIEW_HANDOFF_SHA256
        and s7_v4f_review_handoff.get("mission_id")
        == "submit-fresh-owned-s7-reservation-once-v4f"
        and s7_v4f_review_handoff.get("producer_role") == "reviewer"
        and s7_v4f_review_handoff.get("round") == 1
        and s7_v4f_review_handoff.get("review", {}).get("status") == "done"
        and all(
            fragment in s7_v4f_review_reason
            for fragment in (
                "one launcher attempt",
                "exit code 1",
                "before_spawn",
                "zero candidate-process invocations",
                "authority is consumed",
                "no reservation success is claimed",
            )
        )
        and s7_v4f_terminal_binding.get("unknowns")
        == ["exact_pre_spawn_validation_condition_that_caused_launcher_exit_1"]
        and s7_v4f_claim_boundary.get("authority_consumed") is True
        and s7_v4f_claim_boundary.get(
            "retry_replay_resume_or_reuse_allowed"
        )
        is False
        and s7_v4f_claim_boundary.get(
            "replacement_or_reconciliation_requires_separate_authority"
        )
        is True
        and s7_v4f_claim_boundary.get("gpu_or_model_execution") is False
        and s7_v4f_claim_boundary.get("rtl_correctness_conclusion") is None
        and s7_v4f_claim_boundary.get("stage_transition_authorized") is False
        and s7_v4f_claim_boundary.get("product_acceptance_effect") is False
    )

    ds32_contract_csr = ds32_csr_address_contract.get("csr", {})
    ds32_v1 = ds32_csr_address_contract.get("v1_package_disposition", {})
    ds32_successor = ds32_csr_address_contract.get(
        "required_successor_package", {}
    )
    csr_section = spec.split("### CSR register map", 1)[1].split(
        "### Interrupts and errors", 1
    )[0]
    ds32_csr_contract_ok = bool(
        companion_matches(DS32_CSR_ADDRESS_CONTRACT)
        and sha256(DS32_CSR_ADDRESS_CONTRACT)
        == DS32_CSR_ADDRESS_CONTRACT_SHA256
        and ds32_csr_address_contract.get("stage") == "specification"
        and ds32_csr_address_contract.get("status")
        == "FROZEN_SPECIFICATION_ADDRESS_RECONCILIATION"
        and ds32_csr_address_contract.get("public_parameter_or_port_change")
        is False
        and ds32_contract_csr.get("name")
        == "ACE2_CSR_DS32_MODEL_IDENTITY"
        and ds32_contract_csr.get("address_hex") == "0x0A0"
        and ds32_contract_csr.get("data_width_bits") == 64
        and ds32_contract_csr.get("programmed_valid_reset_value") is False
        and ds32_contract_csr.get("accepted_write_strobe_hex") == "0xFF"
        and ds32_v1.get("status")
        == "IMMUTABLE_ACCEPTED_BUT_UNAPPLIED_SUPERSEDED_FOR_ADDRESS_COLLISION"
        and ds32_v1.get("conflicting_address_hex") == "0x088"
        and ds32_v1.get("conflicting_architectural_csr") == "DESC_SIZE"
        and ds32_v1.get("application_authorized") is False
        and ds32_successor.get("must_use_address_hex") == "0x0A0"
        and ds32_successor.get("must_preserve_existing_csr_addresses") is True
        and ds32_successor.get(
            "must_receive_independent_fresh_review_before_application"
        )
        is True
        and "| `0x088` | `DESC_SIZE` |" in csr_section
        and "| `0x090` | `LAST_RESULT` |" in csr_section
        and "| `0x098` | `LAST_ERROR_INFO` |" in csr_section
        and "| `0x0A0` | `DS32_MODEL_IDENTITY` |" in csr_section
        and "| `0x088` | `DS32_MODEL_IDENTITY` |" not in csr_section
    )

    instruct_diagnostic_claims = instruct_diagnostic_package.get("claims", {})
    instruct_diagnostic_authority = instruct_diagnostic_package.get("authority", {})
    instruct_package_review_scope = instruct_diagnostic_package_review.get("accepted_scope", {})
    instruct_terminal_review_scope = instruct_diagnostic_terminal_review.get("accepted_scope", {})
    instruct_terminal_review_effect = instruct_diagnostic_terminal_review.get("acceptance_effect", {})
    benchmark_stage1_diagnostic = (
        benchmark.get("active_productization_contract", {})
        .get("stage_1", {})
        .get("diagnostic_instruct_runtime_package", {})
    )
    instruct_diagnostic_ok = bool(
        INSTRUCT_DIAGNOSTIC_PACKAGE.with_suffix(INSTRUCT_DIAGNOSTIC_PACKAGE.suffix + ".sha256").read_text(encoding="ascii").strip()
        == f"{INSTRUCT_DIAGNOSTIC_PACKAGE_SHA256}  research/raw/specification/{INSTRUCT_DIAGNOSTIC_PACKAGE.name}"
        and companion_matches(INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING)
        and companion_matches(INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING)
        and INSTRUCT_DIAGNOSTIC_PACKAGE.stat().st_size == 17306
        and sha256(INSTRUCT_DIAGNOSTIC_PACKAGE) == INSTRUCT_DIAGNOSTIC_PACKAGE_SHA256
        and sha256(INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING)
        == INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING_SHA256
        and sha256(INSTRUCT_DIAGNOSTIC_PREFLIGHT) == INSTRUCT_DIAGNOSTIC_PREFLIGHT_SHA256
        and sha256(INSTRUCT_DIAGNOSTIC_EXECUTOR) == INSTRUCT_DIAGNOSTIC_EXECUTOR_SHA256
        and sha256(INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE)
        == INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE_SHA256
        and sha256(INSTRUCT_DIAGNOSTIC_TERMINAL_SEAL)
        == INSTRUCT_DIAGNOSTIC_TERMINAL_SEAL_SHA256
        and sha256(INSTRUCT_DIAGNOSTIC_EVIDENCE_SUMS)
        == INSTRUCT_DIAGNOSTIC_EVIDENCE_SUMS_SHA256
        and sha256(INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING)
        == INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING_SHA256
        and instruct_diagnostic_package.get("status")
        == "READY_FOR_FRESH_L2_REVIEW_ZERO_EXECUTION_AUTHORITY"
        and instruct_diagnostic_package.get("classification")
        == "non_executing_hash_bound_diagnostic_instruct_runtime_build_and_future_execution_package"
        and isinstance(instruct_diagnostic_claims, dict)
        and bool(instruct_diagnostic_claims)
        and all(value is False for value in instruct_diagnostic_claims.values())
        and instruct_diagnostic_authority.get("authority_cardinality") == 0
        and instruct_diagnostic_authority.get("execution_authority_granted") is False
        and instruct_diagnostic_package_review.get("mission_id")
        == "build-freeze-instruct-simulator-binary-v1"
        and instruct_diagnostic_package_review.get("status") == "done"
        and instruct_diagnostic_package_review.get("source_handoff_sha256")
        == INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_HANDOFF_SHA256
        and instruct_diagnostic_package_review.get("reviewed_package", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_PACKAGE_SHA256
        and instruct_package_review_scope.get("authority_cardinality_zero_at_review") is True
        and instruct_package_review_scope.get("runtime_process_starts_at_review") == 0
        and instruct_diagnostic_terminal.get("authority_consumed") is True
        and instruct_diagnostic_terminal.get("process_starts") == 1
        and instruct_diagnostic_terminal.get("classification")
        == "partial_evidence_orchestration_failure_after_runtime_start"
        and instruct_diagnostic_terminal.get("failure_taxonomy")
        == "orchestration_supervisor_post_start_metadata_crash"
        and instruct_diagnostic_terminal.get("runtime_process_present_at_seal") is False
        and instruct_diagnostic_terminal.get("output_directory_exists") is False
        and instruct_diagnostic_terminal.get("generated_token_ids") == []
        and instruct_diagnostic_terminal.get("exit_code") is None
        and instruct_diagnostic_terminal.get("wall_time_seconds") is None
        and instruct_diagnostic_terminal.get("peak_rss_kib") is None
        and instruct_diagnostic_terminal_seal.get("authority_consumed") is True
        and instruct_diagnostic_terminal_seal.get("process_starts") == 1
        and instruct_diagnostic_terminal_seal.get("retry") == "PROHIBITED"
        and instruct_diagnostic_terminal_review.get("mission_id")
        == "execute-instruct-rtl-diagnostic-once-v2"
        and instruct_diagnostic_terminal_review.get("status") == "done"
        and instruct_diagnostic_terminal_review.get("source_handoff_sha256")
        == INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_HANDOFF_SHA256
        and instruct_terminal_review_scope.get("independent_check_count") == 39
        and instruct_terminal_review_scope.get("failed_check_count") == 0
        and instruct_terminal_review_scope.get("authority_consumed") is True
        and instruct_terminal_review_scope.get("runtime_process_starts") == 1
        and instruct_terminal_review_effect.get("diagnostic_run_success") is False
        and instruct_terminal_review_effect.get("retry_replay_resume_or_repair_authorized") is False
        and instruct_diagnostic_terminal_review.get("protected_state", {}).get(
            "pipeline_state_sha256"
        )
        == DIAGNOSTIC_FROZEN_PIPELINE_SHA256
        and instruct_diagnostic_terminal_review.get("protected_state", {}).get(
            "pipeline_stage"
        )
        == DIAGNOSTIC_FROZEN_PIPELINE_STAGE
        and not INSTRUCT_DIAGNOSTIC_OUTPUT.exists()
        and benchmark_stage1_diagnostic.get("package", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_PACKAGE_SHA256
        and benchmark_stage1_diagnostic.get("package_fresh_review_binding", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING_SHA256
        and benchmark_stage1_diagnostic.get("terminal_evidence", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE_SHA256
        and benchmark_stage1_diagnostic.get("terminal_fresh_review_binding", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING_SHA256
        and benchmark_stage1_diagnostic.get("runtime_applicability")
        == "EXECUTED_ONCE_TERMINAL_ORCHESTRATION_FAILURE_AFTER_POPEN"
        and benchmark_stage1_diagnostic.get("execution_authority_consumed") is True
        and benchmark_stage1_diagnostic.get("runtime_process_starts") == 1
        and benchmark_stage1_diagnostic.get("official_output_namespace_absent") is True
        and benchmark_stage1_diagnostic.get("generated_token_count") == 0
        and benchmark_stage1_diagnostic.get("replay_allowed") is False
    )

    benchmark_stage1_supervisor = (
        benchmark.get("active_productization_contract", {})
        .get("stage_1", {})
        .get("future_exactly_once_runtime_supervisor", {})
    )
    benchmark_stage1_v3_package = benchmark_stage1_supervisor.get(
        "reviewed_zero_authority_package", {}
    )
    supervisor_semantics = benchmark_stage1_supervisor.get("required_semantics", [])
    benchmark_stage1_v3_terminal = benchmark_stage1_supervisor.get(
        "reviewed_execute_once_authority", {}
    )
    benchmark_stage1_v4_terminal = benchmark_stage1_supervisor.get(
        "reviewed_v4_terminal_authority", {}
    )
    exact_once_supervisor_ok = bool(
        EXACT_ONCE_SUPERVISOR.is_file()
        and EXACT_ONCE_SUPERVISOR.stat().st_size == 34560
        and sha256(EXACT_ONCE_SUPERVISOR) == EXACT_ONCE_SUPERVISOR_SHA256
        and EXACT_ONCE_SUPERVISOR_TEST.is_file()
        and EXACT_ONCE_SUPERVISOR_TEST.stat().st_size == 16634
        and sha256(EXACT_ONCE_SUPERVISOR_TEST) == EXACT_ONCE_SUPERVISOR_TEST_SHA256
        and EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER.is_file()
        and EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER.stat().st_size == 7704
        and sha256(EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER)
        == EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER_SHA256
        and EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE.is_file()
        and EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE.stat().st_size == 4164
        and sha256(EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE)
        == EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE_SHA256
        and exact_once_supervisor_test_evidence.get("status") == "PASS_PENDING_FRESH_L2"
        and exact_once_supervisor_test_evidence.get("exit_code") == 0
        and exact_once_supervisor_test_evidence.get("interpreter", {}).get("path")
        == "/usr/bin/python3"
        and exact_once_supervisor_test_evidence.get("interpreter", {}).get("version")
        == "3.10.12"
        and exact_once_supervisor_test_evidence.get("regressions", {}).get("tests_run") == 11
        and exact_once_supervisor_test_evidence.get("regressions", {}).get("result") == "PASS"
        and exact_once_supervisor_test_evidence.get("regressions", {}).get("child_scope")
        == "temporary harmless fake child only"
        and exact_once_supervisor_test_evidence.get("source", {}).get(
            "supervisor_after_sha256"
        )
        == EXACT_ONCE_SUPERVISOR_SHA256
        and exact_once_supervisor_test_evidence.get("source", {}).get(
            "test_after_sha256"
        )
        == EXACT_ONCE_SUPERVISOR_TEST_SHA256
        and exact_once_supervisor_test_evidence.get("source", {}).get("verifier_sha256")
        == EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER_SHA256
        and exact_once_supervisor_test_evidence.get("retired_v3", {}).get(
            "ace2_runtime_process_start_count"
        )
        == 0
        and exact_once_supervisor_test_evidence.get("retired_v3", {}).get(
            "package_or_authority_invocations_in_repair"
        )
        == 0
        and exact_once_supervisor_test_evidence.get("authorization_effect", {}).get(
            "authorizes_ace2_runtime"
        )
        is False
        and exact_once_supervisor_test_evidence.get("authorization_effect", {}).get(
            "authorizes_stage_transition"
        )
        is False
        and benchmark_stage1_supervisor.get("status")
        == "PYTHON310_COMPATIBLE_V4_TERMINAL_ONE_START_CHILD_NONZERO_V3_RETIRED_ZERO_RUNTIME_START"
        and benchmark_stage1_supervisor.get("implementation", {}).get("sha256")
        == EXACT_ONCE_SUPERVISOR_SHA256
        and benchmark_stage1_supervisor.get("test_source", {}).get("sha256")
        == EXACT_ONCE_SUPERVISOR_TEST_SHA256
        and benchmark_stage1_supervisor.get("python310_verifier", {}).get("sha256")
        == EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER_SHA256
        and benchmark_stage1_supervisor.get("test_evidence", {}).get("sha256")
        == EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE_SHA256
        and benchmark_stage1_supervisor.get("test_evidence", {}).get("test_count") == 11
        and benchmark_stage1_supervisor.get("test_evidence", {}).get("status")
        == "PASS_PENDING_FRESH_L2"
        and isinstance(supervisor_semantics, list)
        and len(supervisor_semantics) == 9
        and "durable_fsynced_pre_start_intent_before_popen" in supervisor_semantics
        and "at_most_one_process_start" in supervisor_semantics
        and "post_start_fault_never_abandons_child" in supervisor_semantics
        and benchmark_stage1_supervisor.get("external_execution_authority_granted") is False
        and benchmark_stage1_supervisor.get("authority_consumed") is True
        and benchmark_stage1_supervisor.get("production_process_starts") == 1
        and benchmark_stage1_supervisor.get("simulator_executed") is True
        and benchmark_stage1_supervisor.get("rtl_correctness_conclusion") is None
        and benchmark_stage1_supervisor.get("product_acceptance_effect") is False
        and benchmark_stage1_supervisor.get("retroactive_repair_of_consumed_diagnostic") is False
    )
    instruct_diagnostic_v4_ok = bool(
        instruct_diagnostic_v4_report.get("status")
        == "PASS_TERMINAL_FRESH_L2_CHILD_NONZERO_NO_REPLAY"
        and benchmark_stage1_v4_terminal.get("status")
        == "TERMINAL_FRESH_L2_CHILD_NONZERO_EXIT_NO_REPLAY"
        and benchmark_stage1_v4_terminal.get("package", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_PACKAGE_SHA256
        and benchmark_stage1_v4_terminal.get("provenance", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_PROVENANCE_SHA256
        and benchmark_stage1_v4_terminal.get("checksum_manifest", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_CHECKSUMS_SHA256
        and benchmark_stage1_v4_terminal.get("authority", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_SHA256
        and benchmark_stage1_v4_terminal.get("authority", {}).get("decision")
        == "execute_once"
        and benchmark_stage1_v4_terminal.get("fresh_review_binding", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_REVIEW_SHA256
        and benchmark_stage1_v4_terminal.get("fresh_review_binding", {}).get("status")
        == "ACCEPTED_FRESH_L2"
        and benchmark_stage1_v4_terminal.get("terminal_fresh_review_binding", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW_SHA256
        and benchmark_stage1_v4_terminal.get("terminal_evidence_manifest", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST_SHA256
        and benchmark_stage1_v4_terminal.get("authority_cardinality") == 1
        and benchmark_stage1_v4_terminal.get("authority_activated") is True
        and benchmark_stage1_v4_terminal.get("authority_consumed") is True
        and benchmark_stage1_v4_terminal.get("state_reservations") == 1
        and benchmark_stage1_v4_terminal.get("runtime_process_starts") == 1
        and benchmark_stage1_v4_terminal.get("simulator_process_starts") == 1
        and benchmark_stage1_v4_terminal.get("runtime_absent_path_count") == 0
        and benchmark_stage1_v4_terminal.get("terminal_outcome") == "child_nonzero_exit"
        and benchmark_stage1_v4_terminal.get("child_pid") == 258499
        and benchmark_stage1_v4_terminal.get("child_exit_code") == 2
        and benchmark_stage1_v4_terminal.get("failure_ordinal") == 0
        and benchmark_stage1_v4_terminal.get("failure_operator") == "input_rmsnorm"
        and benchmark_stage1_v4_terminal.get("failure_category") == "command_error"
        and benchmark_stage1_v4_terminal.get("generated_token_count") == 0
        and benchmark_stage1_v4_terminal.get("planner_execution_permitted") is False
        and benchmark_stage1_v4_terminal.get("replay_permitted") is False
        and benchmark_stage1_v4_terminal.get("stage_transition_authorized") is False
        and benchmark_stage1_v4_terminal.get("rtl_correctness_conclusion") is None
        and benchmark_stage1_v4_terminal.get("product_acceptance_effect") is False
    )
    instruct_diagnostic_v3_ok = bool(
        INSTRUCT_DIAGNOSTIC_V3_PACKAGE.is_file()
        and INSTRUCT_DIAGNOSTIC_V3_PACKAGE.stat().st_size == 7251
        and sha256(INSTRUCT_DIAGNOSTIC_V3_PACKAGE)
        == INSTRUCT_DIAGNOSTIC_V3_PACKAGE_SHA256
        and INSTRUCT_DIAGNOSTIC_V3_PROVENANCE.is_file()
        and INSTRUCT_DIAGNOSTIC_V3_PROVENANCE.stat().st_size == 16185
        and sha256(INSTRUCT_DIAGNOSTIC_V3_PROVENANCE)
        == INSTRUCT_DIAGNOSTIC_V3_PROVENANCE_SHA256
        and INSTRUCT_DIAGNOSTIC_V3_CHECKSUMS.is_file()
        and INSTRUCT_DIAGNOSTIC_V3_CHECKSUMS.stat().st_size == 598
        and INSTRUCT_DIAGNOSTIC_V3_VERIFIER.is_file()
        and INSTRUCT_DIAGNOSTIC_V3_VERIFIER.stat().st_size == 25300
        and sha256(INSTRUCT_DIAGNOSTIC_V3_VERIFIER)
        == INSTRUCT_DIAGNOSTIC_V3_VERIFIER_SHA256
        and INSTRUCT_DIAGNOSTIC_V3_TEST.is_file()
        and INSTRUCT_DIAGNOSTIC_V3_TEST.stat().st_size == 2067
        and sha256(INSTRUCT_DIAGNOSTIC_V3_TEST)
        == INSTRUCT_DIAGNOSTIC_V3_TEST_SHA256
        and instruct_diagnostic_v3_report.get("status")
        == "PASS_TERMINAL_NO_EXECUTION_CONTAINMENT"
        and instruct_diagnostic_v3_report.get("package_sha256")
        == INSTRUCT_DIAGNOSTIC_V3_PACKAGE_SHA256
        and instruct_diagnostic_v3_report.get("authority_sha256")
        == INSTRUCT_DIAGNOSTIC_V3_AUTHORITY_SHA256
        and instruct_diagnostic_v3_report.get("containment_sha256")
        == INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT_SHA256
        and instruct_diagnostic_v3_report.get("supervisor_payload_invocations") == 1
        and instruct_diagnostic_v3_report.get("state_reservations") == 0
        and instruct_diagnostic_v3_report.get("process_starts") == 0
        and instruct_diagnostic_v3_report.get("runtime_absent_path_count") == 5
        and instruct_diagnostic_v3_report.get("runtime_executed") is False
        and instruct_diagnostic_v3_report.get("retry_or_revalidation_permitted") is False
        and instruct_diagnostic_v3_report.get("rtl_correctness_conclusion") is None
        and all(not path.exists() for path in INSTRUCT_DIAGNOSTIC_V3_RUNTIME_ABSENT_PATHS)
        and benchmark_stage1_v3_package.get("status")
        == "HISTORICAL_PREAUTHORITY_FREEZE_RETIRED_NON_REUSABLE"
        and benchmark_stage1_v3_package.get("package", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V3_PACKAGE_SHA256
        and benchmark_stage1_v3_package.get("provenance", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V3_PROVENANCE_SHA256
        and benchmark_stage1_v3_package.get("static_verifier", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V3_VERIFIER_SHA256
        and benchmark_stage1_v3_package.get("zero_process_regression", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V3_TEST_SHA256
        and benchmark_stage1_v3_package.get("package_reusable") is False
        and benchmark_stage1_v3_package.get("static_revalidation_permitted") is False
        and benchmark_stage1_v3_package.get("fresh_external_execution_authority_required")
        is False
        and benchmark_stage1_v3_package.get("product_acceptance_effect") is False
        and benchmark_stage1_v3_terminal.get("status")
        == "TERMINAL_NO_EXECUTION_CONTAINMENT"
        and benchmark_stage1_v3_terminal.get("authority_consumed") is True
        and benchmark_stage1_v3_terminal.get("state_reservations") == 0
        and benchmark_stage1_v3_terminal.get("runtime_process_starts") == 0
        and benchmark_stage1_v3_terminal.get("supervisor_payload_invocation_count") == 1
        and benchmark_stage1_v3_terminal.get("supervisor_exit_code") == 1
        and benchmark_stage1_v3_terminal.get("failure_taxonomy")
        == "supervisor_module_import_failure"
        and benchmark_stage1_v3_terminal.get("revalidation_permitted") is False
        and benchmark_stage1_v3_terminal.get("containment", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT_SHA256
    )

    active_route = benchmark.get("active_specification_route", {})
    benchmark_s6 = active_route.get("s6_terminal_containment", {}) if isinstance(active_route, dict) else {}
    benchmark_s7 = active_route.get("s7_specification", {}) if isinstance(active_route, dict) else {}
    benchmark_s7_control = (
        benchmark_s7.get("owned_reservation_control_plane_binding", {})
        if isinstance(benchmark_s7, dict)
        else {}
    )
    benchmark_s7_v4f = (
        benchmark_s7.get("terminal_v4f_reservation_launch", {})
        if isinstance(benchmark_s7, dict)
        else {}
    )
    benchmark_instruct_diagnostic = (
        active_route.get("diagnostic_instruct_runtime", {})
        if isinstance(active_route, dict)
        else {}
    )
    benchmark_instruct_v4_terminal = (
        active_route.get("diagnostic_instruct_v4_terminal", {})
        if isinstance(active_route, dict)
        else {}
    )
    active_route_ok = bool(
        benchmark_s6.get("terminal_audit", {}).get("sha256") == S6_TERMINAL_AUDIT_SHA256
        and benchmark_s6.get("fresh_review_binding", {}).get("sha256") == S6_LATEST_SHA256
        and benchmark_s6.get("downstream_authority_opened") is False
        and benchmark_s7.get("specification", {}).get("sha256") == S7_SPECIFICATION_SHA256
        and benchmark_s7.get("manifest", {}).get("sha256") == S7_MANIFEST_SHA256
        and benchmark_s7.get("fresh_review_binding", {}).get("sha256")
        == S7_REVIEW_BINDING_SHA256
        and benchmark_s7.get("package_status")
        == "MATERIALIZED_NONCONSUMING_PRESEAL_REPLACEMENT_FRESH_REVIEW_ACCEPTED_MODEL_ATTEMPT_ABSENT_V4F_RESERVATION_TERMINAL_BEFORE_SPAWN"
        and benchmark_s7.get("execution_package", {}).get("sha256")
        == S7_EXECUTION_MANIFEST_SHA256
        and benchmark_s7.get("execution_package", {}).get("canonical_continuation_id")
        == "c5fd93463239"
        and benchmark_s7.get("package_audit", {}).get("sha256")
        == S7_EXECUTION_AUDIT_SHA256
        and benchmark_s7.get("execution_package_fresh_review", {}).get("sha256")
        == S7_EXECUTION_FRESH_REVIEW_SHA256
        and benchmark_s7.get("execution_package_fresh_review", {}).get("status")
        == "ACCEPTED_S7_PRESEAL_COMPATIBILITY_REPAIR_NO_ATTEMPT"
        and benchmark_s7.get("data_manifest_sha256") == S7_DATA_MANIFEST_SHA256
        and benchmark_s7.get("data_or_probe_materialized") is True
        and benchmark_s7.get("package_construction_authority_granted") is True
        and benchmark_s7.get("operator_authority_granted") is True
        and benchmark_s7.get("operator_authority_consumed") is True
        and benchmark_s7.get("consuming_attempt_authority_granted") is False
        and benchmark_s7.get("gpu_preflight_completed") is False
        and benchmark_s7.get("accepted_base_execution_package", {}).get("sha256")
        == S7_ACCEPTED_BASE_MANIFEST_SHA256
        and benchmark_s7.get("accepted_base_execution_package", {}).get("runner_sha256")
        == S7_ACCEPTED_BASE_RUNNER_SHA256
        and benchmark_s7.get("opaque_s6_disjointness_custodian_attestation_present") is True
        and benchmark_s7.get("opaque_s6_disjointness_custodian_attestation", {})
        .get("sha256")
        == S7_CUSTODIAN_ATTESTATION_SHA256
        and benchmark_s7.get("opaque_s6_disjointness_custodian_attestation", {})
        .get("status")
        == "PASS"
        and benchmark_s7.get("opaque_s6_disjointness_custodian_attestation", {})
        .get("fresh_review_binding", {})
        .get("sha256")
        == S7_CUSTODIAN_REVIEW_BINDING_SHA256
        and benchmark_s7.get("opaque_s6_disjointness_custodian_attestation", {})
        .get("fresh_review_binding", {})
        .get("source_handoff_sha256")
        == S7_CUSTODIAN_REVIEW_HANDOFF_SHA256
        and benchmark_s7.get("opaque_s6_disjointness_custodian_attestation", {})
        .get("consuming_execution_authority_granted")
        is False
        and benchmark_s7.get("acceptance_effect")
        == "materialized_nonconsuming_model_package_and_custodian_pass_fresh_review_accepted_v4f_reservation_authority_consumed_terminal_before_spawn_no_reservation_success_no_downstream_authority"
        and benchmark_s7.get("quantization_or_rtl_authorized") is False
        and s7_execution_package_ok
        and s7_custodian_ok
        and benchmark_s7_v4f.get("binding", {}).get("sha256")
        == S7_V4F_TERMINAL_BINDING_SHA256
        and benchmark_s7_v4f.get("classification")
        == "TERMINAL_PRE_CANDIDATE_SPAWN_FAILURE_NO_RESERVATION_SUCCESS"
        and benchmark_s7_v4f.get("launcher_invocation_count") == 1
        and benchmark_s7_v4f.get("launcher_exit_code") == 1
        and benchmark_s7_v4f.get("allowlisted_phase_evidence")
        == ["before_spawn"]
        and benchmark_s7_v4f.get("candidate_process_invocation_count") == 0
        and benchmark_s7_v4f.get("candidate_process_started") is False
        and benchmark_s7_v4f.get("reservation_success_established") is False
        and benchmark_s7_v4f.get("backend_identity_established") is False
        and benchmark_s7_v4f.get("authority_consumed") is True
        and benchmark_s7_v4f.get("replay_allowed") is False
        and benchmark_s7_v4f.get("fresh_l2_status") == "done"
        and benchmark_s7_v4f.get("fresh_l2_handoff_sha256")
        == S7_V4F_REVIEW_HANDOFF_SHA256
        and benchmark_s7_v4f.get("exact_pre_spawn_failure_condition") is None
        and benchmark_s7_v4f.get("product_acceptance_effect") is False
        and s7_v4f_terminal_ok
        and benchmark_s7_control.get("classification")
        == "NO_SAFE_QUERY_BINDING"
        and benchmark_s7_control.get("evidence", {}).get("sha256")
        == S7_CONTROL_PLANE_EVIDENCE_SHA256
        and benchmark_s7_control.get("fresh_review_binding", {}).get("sha256")
        == S7_CONTROL_PLANE_REVIEW_BINDING_SHA256
        and benchmark_s7_control.get("fresh_review_binding", {}).get(
            "source_handoff_sha256"
        )
        == S7_CONTROL_PLANE_REVIEW_HANDOFF_SHA256
        and benchmark_s7_control.get("missing_exact_fields")
        == required_control_plane_fields
        and benchmark_s7_control.get("authoritative_control_plane_query_count")
        == 0
        and benchmark_s7_control.get("amlt_local_lookup_count") == 0
        and benchmark_s7_control.get("network_query_count") == 0
        and benchmark_s7_control.get("query_or_execution_authorized") is False
        and s7_control_plane_ok
        and benchmark_ds32.get("contract_sha256")
        == DS32_CSR_ADDRESS_CONTRACT_SHA256
        and benchmark_ds32.get("address_hex") == "0x0A0"
        and benchmark_ds32.get("v1_conflicting_address_hex") == "0x088"
        and benchmark_ds32.get("v1_conflicting_architectural_csr")
        == "DESC_SIZE"
        and benchmark_ds32.get("rtl_application_authorized") is False
        and ds32_csr_contract_ok
        and benchmark_instruct_diagnostic.get("package", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_PACKAGE_SHA256
        and benchmark_instruct_diagnostic.get("package_fresh_review_binding", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING_SHA256
        and benchmark_instruct_diagnostic.get("package_fresh_review_binding", {}).get("source_handoff_sha256")
        == INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_HANDOFF_SHA256
        and benchmark_instruct_diagnostic.get("terminal_evidence", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE_SHA256
        and benchmark_instruct_diagnostic.get("terminal_fresh_review_binding", {}).get("sha256")
        == INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING_SHA256
        and benchmark_instruct_diagnostic.get("terminal_fresh_review_binding", {}).get("source_handoff_sha256")
        == INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_HANDOFF_SHA256
        and benchmark_instruct_diagnostic.get("runtime_applicability")
        == "EXECUTED_ONCE_TERMINAL_ORCHESTRATION_FAILURE_AFTER_POPEN"
        and benchmark_instruct_diagnostic.get("execution_authority_consumed") is True
        and benchmark_instruct_diagnostic.get("runtime_process_starts") == 1
        and benchmark_instruct_diagnostic.get("generated_token_count") == 0
        and benchmark_instruct_diagnostic.get("replay_allowed") is False
        and benchmark_instruct_diagnostic.get("rtl_correctness_conclusion") is None
        and benchmark_instruct_diagnostic.get("official_output_namespace_absent") is True
        and instruct_diagnostic_ok
        and benchmark_instruct_v4_terminal.get("status")
        == "TERMINAL_FRESH_L2_CHILD_NONZERO_EXIT_NO_REPLAY"
        and benchmark_instruct_v4_terminal.get("package_sha256")
        == INSTRUCT_DIAGNOSTIC_V4_PACKAGE_SHA256
        and benchmark_instruct_v4_terminal.get("authority_sha256")
        == INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_SHA256
        and benchmark_instruct_v4_terminal.get("authority_review_binding_sha256")
        == INSTRUCT_DIAGNOSTIC_V4_REVIEW_SHA256
        and benchmark_instruct_v4_terminal.get("terminal_review_binding_sha256")
        == INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW_SHA256
        and benchmark_instruct_v4_terminal.get("terminal_evidence_manifest_sha256")
        == INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST_SHA256
        and benchmark_instruct_v4_terminal.get("authority_cardinality") == 1
        and benchmark_instruct_v4_terminal.get("authority_activated") is True
        and benchmark_instruct_v4_terminal.get("authority_consumed") is True
        and benchmark_instruct_v4_terminal.get("state_reservations") == 1
        and benchmark_instruct_v4_terminal.get("runtime_process_starts") == 1
        and benchmark_instruct_v4_terminal.get("runtime_absent_path_count") == 0
        and benchmark_instruct_v4_terminal.get("terminal_outcome") == "child_nonzero_exit"
        and benchmark_instruct_v4_terminal.get("child_exit_code") == 2
        and benchmark_instruct_v4_terminal.get("failure_ordinal") == 0
        and benchmark_instruct_v4_terminal.get("failure_operator") == "input_rmsnorm"
        and benchmark_instruct_v4_terminal.get("failure_category") == "command_error"
        and benchmark_instruct_v4_terminal.get("generated_token_count") == 0
        and benchmark_instruct_v4_terminal.get("planner_execution_permitted") is False
        and benchmark_instruct_v4_terminal.get("replay_permitted") is False
        and benchmark_instruct_v4_terminal.get("stage_transition_authorized") is False
        and benchmark_instruct_v4_terminal.get("rtl_correctness_conclusion") is None
        and benchmark_instruct_v4_terminal.get("product_completion_claim") is False
        and instruct_diagnostic_v4_ok
    )

    benchmark_closed = bool(
        benchmark.get("applies") is False
        and benchmark.get("external_contract") is None
        and benchmark.get("non_benchmark_statement")
        and isinstance(local, dict)
        and local.get("top_module") == "ace2_shell"
        and local.get("public_interface_source")
        == "design/SPEC.md#public-ace2_shell-parameter-and-port-contract"
        and v3_evidence_ok
        and v4_terminal_ok
        and v4_diagnostic_ok
        and v5_recommendation_l2_ok
        and v5_package_ok
        and v7_terminal_ok
        and v8_terminal_ok
        and s1_ok
        and s3_terminal_ok
        and s4_ok
        and s6_terminal_ok
        and s7_specification_ok
        and s7_execution_package_ok
        and s7_custodian_ok
        and s7_control_plane_ok
        and s7_v4f_terminal_ok
        and ds32_csr_contract_ok
        and instruct_diagnostic_ok
        and exact_once_supervisor_ok
        and instruct_diagnostic_v3_ok
        and instruct_diagnostic_v4_ok
        and active_route_ok
    )

    ground_truth_text = GROUND_TRUTH.read_text(encoding="utf-8")
    ground_truth_lines = ground_truth_text.splitlines()
    binding_records, unexpected_ground_truth_lines = markdown_binding_records(
        ground_truth_lines
    )
    manager_incident_attribution_fragments = (
        "daemon/latest runtime supports terminal status `superseded`",
        "there is no daemon schema incompatibility",
        "/home/argustest/argustest2/argus_skill",
        "/home/argustest/argustest2/argus-skill-latest/argus_skill",
        "Twenty-one contained stale or unauthorized items",
        "supported terminal `skipped`",
        "canonical project hashes named by the Manager remain unchanged",
    )
    manager_incident_attribution_recorded = all(
        fragment in ground_truth_text
        for fragment in manager_incident_attribution_fragments
    )
    current_stage = pipeline.get("current_stage")
    specification_status = pipeline.get("stages", {}).get("specification", {}).get("status")
    accepted_specification_statuses = {
        "in_progress",
        "closed_next_stage_approved",
        "done",
    }
    active_stage_matches = (
        current_stage == "specification"
        and specification_status in accepted_specification_statuses
    )
    manager_reconciliation_required = bool(
        current_stage == "v4_execution_package"
        and specification_status == "closed_next_stage_approved"
        and v4_terminal_ok
    )
    pipeline_checksum_matches = checksum_matches(PIPELINE, PIPELINE_CHECKSUM)
    stage_projection_claims = current_stage_projection_claims(binding_records)
    stage_projection_conflicts = [
        claim
        for claim in stage_projection_claims
        if claim["projected_stage"] != current_stage
    ]
    current_stage_projection_consistent = not stage_projection_conflicts

    u280_gate = benchmark.get("u280_environment_gate", {})
    latest_u280_path, latest_u280_matches = bound_file(
        u280_gate, "latest_inventory", "latest_inventory_sha256"
    ) if isinstance(u280_gate, dict) else (ROOT, False)
    u280_ok = bool(
        latest_u280_matches
        and u280_gate.get("vivado_present") is False
        and u280_gate.get("vitis_present") is False
        and u280_gate.get("vpp_present") is False
        and u280_gate.get("xrt_utilities_present") is False
        and u280_gate.get("xilinx_pci_vendor_10ee_function_count") == 0
        and u280_gate.get("xrt_device_node_count") == 0
        and u280_gate.get("hardware_emulation_endpoint") is None
        and u280_gate.get("u280_board_access") is False
        and u280_gate.get("toolchain_download_authorized") is False
    )

    checklist = {
        "spec.behavior-interface": {
            "status": "PASS"
            if shell.get("status") == "PASS"
            and behavior_ok
            and ds32_csr_contract_ok
            and s7_control_plane_ok
            and s7_v4f_terminal_ok
            else "FAIL",
            "shell_contract_audit": shell.get("status"),
            "parameter_count": shell.get("checks", {}).get("parameter_count_source"),
            "port_count": shell.get("checks", {}).get("port_count_source"),
            "missing_behavior_fragments": behavior_missing,
            "ds32_csr_address_reconciliation_verified": ds32_csr_contract_ok,
            "s7_owned_reservation_no_safe_query_binding_verified": s7_control_plane_ok,
            "s7_v4f_terminal_before_spawn_verified": s7_v4f_terminal_ok,
        },
        "spec.clock-reset-protocol": {
            "status": "PASS" if protocol_ok else "FAIL",
            "clock_domains": 1,
            "missing_protocol_fragments": protocol_missing,
        },
        "spec.acceptance-matrix": {
            "status": "PASS" if not missing_classes else "FAIL",
            "required_classes": list(REQUIRED_ACCEPTANCE_CLASSES),
            "observed_classes": observed_classes,
            "missing_classes": missing_classes,
        },
        "spec.benchmark-interface-closure": {
            "status": "PASS" if benchmark_closed else "FAIL",
            "external_benchmark_applies": benchmark.get("applies"),
            "external_contract": benchmark.get("external_contract"),
            "v3_terminal_evidence_verified": v3_evidence_ok,
            "v4_frozen_inputs_verified": v4_freeze_ok,
            "v4_terminal_dev_no_go_verified": v4_terminal_ok,
            "v4_aggregate_diagnostic_and_v5_recommendation_verified": v4_diagnostic_ok,
            "v5_recommendation_independent_l2_verified": v5_recommendation_l2_ok,
            "v5_package_and_aborted_attempt_containment_verified": v5_package_ok,
            "v7_terminal_lifecycle_and_fresh_review_verified": v7_terminal_ok,
            "v8_preexecution_package_verified": v8_package_freeze_ok,
            "v8_terminal_no_execution_and_fresh_review_verified": v8_terminal_ok,
            "s1_marker_free_transport_and_fresh_review_verified": s1_ok,
            "s3_terminal_containment_and_fresh_review_verified": s3_terminal_ok,
            "s4_terminal_containment_and_fresh_review_verified": s4_ok,
            "s6_terminal_probe_quality_no_go_and_fresh_review_verified": s6_terminal_ok,
            "s7_non_consuming_specification_and_fresh_review_verified": s7_specification_ok,
            "s7_non_consuming_execution_package_and_fresh_review_verified": s7_execution_package_ok,
            "s7_custodian_pass_and_fresh_review_verified": s7_custodian_ok,
            "s7_owned_reservation_no_safe_query_binding_verified": s7_control_plane_ok,
            "s7_v4f_terminal_before_spawn_verified": s7_v4f_terminal_ok,
            "ds32_csr_address_reconciliation_verified": ds32_csr_contract_ok,
            "diagnostic_instruct_package_and_fresh_review_verified": instruct_diagnostic_ok,
            "future_exactly_once_runtime_supervisor_verified": exact_once_supervisor_ok,
            "future_exactly_once_instruct_v3_package_verified": instruct_diagnostic_v3_ok,
            "terminal_instruct_v4_one_start_child_nonzero_verified": instruct_diagnostic_v4_ok,
            "active_specification_route_matches_benchmark_contract": active_route_ok,
        },
        "spec.pipeline-state-integrity": {
            "status": "PASS"
            if pipeline_checksum_matches
            and current_stage_projection_consistent
            and manager_incident_attribution_recorded
            else "FAIL",
            "checksum_companion": PIPELINE_CHECKSUM.relative_to(ROOT).as_posix(),
            "checksum_matches": pipeline_checksum_matches,
            "current_stage_projection_consistent": current_stage_projection_consistent,
            "current_stage_projection_claims": stage_projection_claims,
            "current_stage_projection_conflicts": stage_projection_conflicts,
            "manager_incident_attribution_recorded": manager_incident_attribution_recorded,
            "manager_incident_attribution_missing_fragments": [
                fragment
                for fragment in manager_incident_attribution_fragments
                if fragment not in ground_truth_text
            ],
        },
    }
    checklist_pass = all(item["status"] == "PASS" for item in checklist.values())
    selected_policy_id = internal.get("selected_policy_id") if isinstance(internal, dict) else None
    compression_selected = compression.get("selected_policy_id") if isinstance(compression, dict) else None
    stage1 = benchmark.get("stage_order", {}).get("stage_1_local_runtime", "")
    stage2 = benchmark.get("stage_order", {}).get("stage_2_u280")
    lifecycle_closed = bool(
        companion_matches(S6_CONTRACT)
        and companion_matches(S6_MANIFEST)
        and companion_matches(S6_L2_ACCEPTANCE)
        and sha256(S6_CONTRACT) == S6_CONTRACT_SHA256
        and s6_contract.get("contract_id")
        == "qwen2.5-0.5b-instruct-ace2-bf16-full-finetune-successor-s6-package-v1"
        and s6_contract.get("attempt_authority", {}).get("granted") is False
        and s6_contract.get("attempt_authority", {}).get(
            "separate_fresh_operator_authority_required"
        )
        is True
        and s6_manifest.get("contract", {}).get("sha256") == S6_CONTRACT_SHA256
        and s6_manifest.get("tree_sha256") == S6_EXECUTION_TREE
        and s6_l2_acceptance.get("accepted") is True
        and s6_l2_acceptance.get("status")
        == "ACCEPTED_BF16_FULL_FINETUNE_SUCCESSOR_S6_PACKAGE_NO_ATTEMPT"
        and s6_l2_acceptance.get("contract_sha256") == S6_CONTRACT_SHA256
        and s6_l2_acceptance.get("execution_tree_sha256") == S6_EXECUTION_TREE
        and s6_l2_acceptance.get("attempt_authority_granted") is False
        and s6_l2_acceptance.get("detached_launch_authorized") is False
        and S6_AUTHORITY.exists()
        and not S6_RUN_ROOT.exists()
        and S6_SUBMISSION_INTENT.exists()
        and S6_SUBMISSION_RESULT.exists()
        and s6_terminal_ok
        and s7_specification_ok
        and instruct_diagnostic_ok
        and exact_once_supervisor_ok
        and instruct_diagnostic_v3_ok
        and instruct_diagnostic_v4_ok
        and active_route_ok
        and v7_terminal_ok
        and v8_terminal_ok
        and s1_ok
        and s3_terminal_ok
        and s4_ok
        and s7_v4f_terminal_ok
        and selected_policy_id is None
        and compression_selected is None
        and v4_diagnostic_ok
        and v5_recommendation_l2_ok
        and v5_package_ok
        and "terminal_s6_probe_quality_no_go" in stage1
        and "fresh_l2_accepted_s7_specification" in stage1
        and "model_execution_package_custodian_pass" in stage1
        and "v4f_reservation_authority_consumed_terminal_before_spawn_zero_candidate_invocations_no_reservation_success"
        in stage1
        and "terminal_v4_one_start_child_nonzero_input_rmsnorm_no_tokens_no_replay"
        in stage1
        and stage2 == "pending_after_stage_1"
    )
    content_pass = bool(
        checklist_pass
        and not unexpected_ground_truth_lines
        and u280_ok
        and lifecycle_closed
    )
    if content_pass and active_stage_matches:
        status = "PASS"
    elif content_pass and manager_reconciliation_required:
        status = "PASS_SPECIFICATION_CHECKLIST_MANAGER_RECONCILIATION_REQUIRED"
    else:
        status = "FAIL"

    input_paths = [
        SELF,
        SHELL_AUDIT,
        SHELL,
        MAKEFILE,
        SPEC,
        BENCHMARK,
        PIPELINE,
        PIPELINE_CHECKSUM,
        GROUND_TRUTH,
        LATEST,
        U280_INVENTORY,
        EXACT_ONCE_SUPERVISOR,
        EXACT_ONCE_SUPERVISOR_TEST,
        EXACT_ONCE_SUPERVISOR_PYTHON310_VERIFIER,
        EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE,
        INSTRUCT_DIAGNOSTIC_V3_PACKAGE,
        INSTRUCT_DIAGNOSTIC_V3_PROVENANCE,
        INSTRUCT_DIAGNOSTIC_V3_CHECKSUMS,
        INSTRUCT_DIAGNOSTIC_V3_VERIFIER,
        INSTRUCT_DIAGNOSTIC_V3_TEST,
        INSTRUCT_DIAGNOSTIC_V3_CONTAINMENT,
        INSTRUCT_DIAGNOSTIC_V4_PACKAGE,
        INSTRUCT_DIAGNOSTIC_V4_PROVENANCE,
        INSTRUCT_DIAGNOSTIC_V4_CHECKSUMS,
        INSTRUCT_DIAGNOSTIC_V4_AUTHORITY,
        INSTRUCT_DIAGNOSTIC_V4_AUTHORITY_CHECKSUM,
        INSTRUCT_DIAGNOSTIC_V4_REVIEW,
        INSTRUCT_DIAGNOSTIC_V4_REVIEW_CHECKSUM,
        INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW,
        INSTRUCT_DIAGNOSTIC_V4_TERMINAL_REVIEW_CHECKSUM,
        INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST,
        INSTRUCT_DIAGNOSTIC_V4_TERMINAL_MANIFEST_CHECKSUM,
        INSTRUCT_DIAGNOSTIC_V4_TERMINAL,
        INSTRUCT_DIAGNOSTIC_V4_DURABLE_TERMINAL,
        INSTRUCT_DIAGNOSTIC_V4_PROCESS_STARTED,
        INSTRUCT_DIAGNOSTIC_V4_FIRST_FAILURE,
        INSTRUCT_DIAGNOSTIC_V4_SUMMARY,
        INSTRUCT_DIAGNOSTIC_V4_PROGRESS,
        INSTRUCT_DIAGNOSTIC_V4_COMMANDS,
        V4_ATTESTATION,
        V4_DIAGNOSTIC_BUILDER,
        V4_DIAGNOSTIC,
        V4_DIAGNOSTIC_CHECKSUM,
        V5_RECOMMENDATION_L2,
        V5_RECOMMENDATION_L2_CHECKSUM,
        V5_EXECUTION_CONTRACT,
        V5_EXECUTION_MANIFEST,
        V5_EXECUTION_SELF_TEST,
        V5_EXECUTION_L2_DECISION,
        V5_EXECUTION_L2_ACCEPTANCE,
        V5_ATTEMPT_AUTHORITY,
        V5_LAUNCH_READINESS,
        V5_ATTEMPT_MARKER,
        V5_EXIT_STATUS,
        V5_ATTEMPT_CONSOLE,
        V5_CONTAINMENT_INPUT,
        V5_CONTAINMENT_L2,
        V5_INVALID_CONTAINMENT_INPUT,
        V5_INVALID_CONTAINMENT_L2,
        V7_ATTEMPT_AUTHORITY,
        V7_TERMINAL_AUDIT,
        V7_FRESH_REVIEW,
        V8_CONTRACT,
        V8_FREEZE,
        V8_MANIFEST,
        V8_SELF_TEST,
        V8_REVIEW_REQUEST,
        V8_REVIEW_DECISION,
        V8_REVIEW_ACCEPTANCE,
        V8_AUTHORITY,
        V8_INVENTORY,
        V8_INTENT,
        V8_RESULT,
        V8_POST_INVENTORY,
        V8_TERMINAL_AUDIT,
        V8_FRESH_REVIEW,
        S1_CONTRACT,
        S1_PACKAGE_AUDIT,
        S1_REPAIR_AUDIT,
        S1_FRESH_REVIEW,
        S1_AUTHORITY,
        S1_INTENT,
        S1_RESULT,
        S1_RAW,
        S1_TERMINAL_AUDIT,
        S1_TERMINAL_REVIEW,
        S1_TERMINAL_STDOUT,
        S3_CONTRACT,
        S3_PACKAGE_AUDIT,
        S3_MANIFEST,
        S3_SELF_TEST,
        S3_L2_ACCEPTANCE,
        S3_TERMINAL_AUDIT,
        S3_TERMINAL_REVIEW,
        S4_CONTRACT,
        S4_PACKAGE_AUDIT,
        S4_MANIFEST,
        S4_SELF_TEST,
        S4_L2_ACCEPTANCE,
        S4_AUTHORITY,
        S4_TERMINAL_AUDIT,
        S4_TERMINAL_REVIEW,
        S6_CONTRACT,
        S6_MANIFEST,
        S6_L2_ACCEPTANCE,
        S6_AUTHORITY,
        S6_SUBMISSION_INTENT,
        S6_SUBMISSION_RESULT,
        S6_TERMINAL_AUDIT,
        S7_SPECIFICATION,
        S7_MANIFEST,
        S7_FRESH_REVIEW_BINDING,
        S7_EXECUTION_AUTHORITY,
        S7_EXECUTION_MANIFEST,
        S7_EXECUTION_AUDIT,
        S7_EXECUTION_FRESH_REVIEW,
        S7_DATA_MANIFEST,
        S7_CUSTODIAN_ATTESTATION,
        S7_CUSTODIAN_REVIEW_BINDING,
        S7_CONTROL_PLANE_EVIDENCE,
        S7_CONTROL_PLANE_REVIEW_BINDING,
        DS32_CSR_ADDRESS_CONTRACT,
        INSTRUCT_DIAGNOSTIC_PACKAGE,
        INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING,
        INSTRUCT_DIAGNOSTIC_PREFLIGHT,
        INSTRUCT_DIAGNOSTIC_EXECUTOR,
        INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE,
        INSTRUCT_DIAGNOSTIC_TERMINAL_SEAL,
        INSTRUCT_DIAGNOSTIC_EVIDENCE_SUMS,
        INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING,
        latest_u280_path,
        *[
            project_path(record.get("path", ""))
            for record in v5_containment_input.get("immutable_negative_evidence", [])
            if isinstance(record, dict)
        ],
        *v3_paths,
        *v4_paths,
        *v4_terminal_paths.values(),
    ]
    existing_inputs = sorted({path for path in input_paths if path.is_file()})
    return {
        "schema_version": 12,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "read-only specification-stage audit; no model execution, simulation, formal, synthesis, PPA, chat, FPGA build, emulation, or board execution",
        "status": status,
        "pipeline": {
            "current_stage": current_stage,
            "specification_status": specification_status,
            "accepted_specification_statuses": sorted(accepted_specification_statuses),
            "active_stage_matches": active_stage_matches,
            "manager_reconciliation_required": manager_reconciliation_required,
            "manager_owned_state_mutated": False,
        },
        "checklist": checklist,
        "binding_ground_truth": {
            "count": len(binding_records),
            "items": binding_records,
            "unexpected_non_unknown_lines": unexpected_ground_truth_lines,
        },
        "mission_gate": {
            "specification_checklist_complete": checklist_pass,
            "v3_terminal_evidence_verified": v3_evidence_ok,
            "v4_frozen_inputs_verified": v4_freeze_ok,
            "v4_terminal_dev_no_go_verified": v4_terminal_ok,
            "v4_aggregate_diagnostic_and_v5_recommendation_verified": v4_diagnostic_ok,
            "v5_recommendation_independent_l2_verified": v5_recommendation_l2_ok,
            "v5_package_and_aborted_attempt_containment_verified": v5_package_ok,
            "v7_terminal_lifecycle_and_fresh_review_verified": v7_terminal_ok,
            "v8_preexecution_package_verified": v8_package_freeze_ok,
            "v8_terminal_no_execution_and_fresh_review_verified": v8_terminal_ok,
            "s1_marker_free_transport_and_fresh_review_verified": s1_ok,
            "s3_terminal_containment_and_fresh_review_verified": s3_terminal_ok,
            "s3_terminal_audit_check_count": s3_terminal_audit.get("check_count"),
            "s3_failure_taxonomy": s3_terminal_audit.get("failure_taxonomy"),
            "s3_attempt_authority_present": S3_AUTHORITY.exists(),
            "s3_submission_intent_present": S3_SUBMISSION_INTENT.exists(),
            "s3_official_namespace_present": S3_RUN_ROOT.exists(),
            "s4_terminal_containment_and_fresh_review_verified": s4_ok,
            "s4_package_identity_sha256": s4_package_audit.get("package_identity_sha256"),
            "s4_execution_tree_sha256": s4_manifest.get("tree_sha256"),
            "s4_package_audit_check_count": s4_package_audit.get("check_count"),
            "s4_required_stage_file_count": s4_package_audit.get("stage_closure", {}).get("required_file_count"),
            "s4_runner_self_test_check_count": s4_self_test.get("check_count"),
            "s4_fresh_l2_status": s4_l2_acceptance.get("status"),
            "s4_attempt_authority_present": S4_AUTHORITY.exists(),
            "s4_submission_intent_present": (S4_SUBMISSION_ROOT / "backend-submission-intent.json").exists(),
            "s4_official_namespace_present": S4_RUN_ROOT.exists(),
            "s4_terminal_audit_check_count": s4_terminal_audit.get("check_count"),
            "s4_failure_taxonomy": s4_terminal_audit.get("failure_taxonomy"),
            "s4_terminal_review_status": s4_terminal_review.get("review", {}).get("status"),
            "s6_contract_sha256": sha256(S6_CONTRACT),
            "s6_execution_tree_sha256": s6_manifest.get("tree_sha256"),
            "s6_fresh_l2_status": s6_l2_acceptance.get("status"),
            "s6_attempt_authority_present": S6_AUTHORITY.exists(),
            "s6_submission_intent_present": (
                S6_SUBMISSION_ROOT / "backend-submission-intent.json"
            ).exists(),
            "s6_official_namespace_present": S6_RUN_ROOT.exists(),
            "s6_terminal_probe_quality_no_go_verified": s6_terminal_ok,
            "s6_terminal_audit_check_count": len(
                s6_terminal_audit.get("checks", {})
            ),
            "s6_terminal_audit_sha256": sha256(S6_TERMINAL_AUDIT),
            "s6_terminal_fresh_reviewer_status": latest.get(
                "independent_review", {}
            ).get("status"),
            "s7_specification_fresh_review_verified": s7_specification_ok,
            "s7_specification_sha256": sha256(S7_SPECIFICATION),
            "s7_manifest_sha256": sha256(S7_MANIFEST),
            "s7_fresh_review_binding_sha256": sha256(S7_FRESH_REVIEW_BINDING),
            "s7_execution_package_and_fresh_review_verified": s7_execution_package_ok,
            "s7_execution_package_manifest_sha256": sha256(S7_EXECUTION_MANIFEST),
            "s7_execution_package_audit_sha256": sha256(S7_EXECUTION_AUDIT),
            "s7_execution_package_fresh_review_sha256": sha256(S7_EXECUTION_FRESH_REVIEW),
            "s7_execution_package_canonical_continuation_id": s7_execution_manifest.get(
                "continuation_id"
            ),
            "s7_consuming_state_present": S7_V4F_ATTEMPT_CONSUMED.exists(),
            "s7_model_consuming_state_present": any(
                path.exists() for path in s7_consuming_paths
            ),
            "s7_custodian_attestation_present": S7_CUSTODIAN_ATTESTATION.exists(),
            "s7_custodian_attestation_and_fresh_review_verified": s7_custodian_ok,
            "s7_custodian_attestation_sha256": sha256(S7_CUSTODIAN_ATTESTATION),
            "s7_custodian_review_binding_sha256": sha256(
                S7_CUSTODIAN_REVIEW_BINDING
            ),
            "s7_owned_reservation_no_safe_query_binding_verified": s7_control_plane_ok,
            "s7_owned_reservation_evidence_sha256": sha256(
                S7_CONTROL_PLANE_EVIDENCE
            ),
            "s7_owned_reservation_review_binding_sha256": sha256(
                S7_CONTROL_PLANE_REVIEW_BINDING
            ),
            "s7_owned_reservation_missing_exact_fields": required_control_plane_fields,
            "s7_owned_reservation_control_plane_query_count": 0,
            "s7_v4f_terminal_before_spawn_verified": s7_v4f_terminal_ok,
            "s7_v4f_terminal_binding_sha256": sha256(S7_V4F_TERMINAL_BINDING),
            "s7_v4f_launcher_exit_code": s7_v4f_launcher_outcome.get(
                "launcher_exit_code"
            ),
            "s7_v4f_candidate_process_invocation_count": s7_v4f_launcher_outcome.get(
                "candidate_process_invocation_count"
            ),
            "s7_v4f_reservation_success_established": s7_v4f_launcher_outcome.get(
                "reservation_success_established"
            ),
            "s7_consuming_execution_authority_granted": True,
            "s7_consuming_execution_authority_consumed": True,
            "ds32_csr_address_reconciliation_verified": ds32_csr_contract_ok,
            "ds32_csr_address_contract_sha256": sha256(
                DS32_CSR_ADDRESS_CONTRACT
            ),
            "ds32_csr_address_hex": ds32_contract_csr.get("address_hex"),
            "ds32_v1_patch_application_authorized": False,
            "diagnostic_instruct_package_and_fresh_review_verified": instruct_diagnostic_ok,
            "diagnostic_instruct_package_sha256": sha256(
                INSTRUCT_DIAGNOSTIC_PACKAGE
            ),
            "diagnostic_instruct_review_binding_sha256": sha256(
                INSTRUCT_DIAGNOSTIC_PACKAGE_REVIEW_BINDING
            ),
            "diagnostic_instruct_terminal_evidence_sha256": sha256(INSTRUCT_DIAGNOSTIC_TERMINAL_EVIDENCE),
            "diagnostic_instruct_terminal_review_binding_sha256": sha256(INSTRUCT_DIAGNOSTIC_TERMINAL_REVIEW_BINDING),
            "diagnostic_instruct_historical_pipeline_snapshot_sha256": instruct_diagnostic_terminal_review.get(
                "protected_state", {}
            ).get("pipeline_state_sha256"),
            "diagnostic_instruct_runtime_applicability": "EXECUTED_ONCE_TERMINAL_ORCHESTRATION_FAILURE_AFTER_POPEN",
            "diagnostic_instruct_execution_authority_granted": True,
            "diagnostic_instruct_execution_authority_consumed": True,
            "diagnostic_instruct_runtime_executed": True,
            "diagnostic_instruct_output_namespace_present": INSTRUCT_DIAGNOSTIC_OUTPUT.exists(),
            "future_exactly_once_runtime_supervisor_verified": exact_once_supervisor_ok,
            "future_exactly_once_runtime_supervisor_sha256": sha256(EXACT_ONCE_SUPERVISOR),
            "future_exactly_once_runtime_supervisor_test_sha256": sha256(
                EXACT_ONCE_SUPERVISOR_TEST
            ),
            "future_exactly_once_runtime_supervisor_test_evidence_sha256": sha256(
                EXACT_ONCE_SUPERVISOR_TEST_EVIDENCE
            ),
            "future_exactly_once_runtime_supervisor_test_count": exact_once_supervisor_test_evidence.get(
                "regressions", {}
            ).get("tests_run"),
            "future_exactly_once_runtime_supervisor_external_authority_granted": False,
            "future_exactly_once_runtime_supervisor_product_acceptance_effect": False,
            "future_exactly_once_instruct_v3_package_verified": instruct_diagnostic_v3_ok,
            "future_exactly_once_instruct_v3_package_sha256": sha256(
                INSTRUCT_DIAGNOSTIC_V3_PACKAGE
            ),
            "future_exactly_once_instruct_v3_provenance_sha256": sha256(
                INSTRUCT_DIAGNOSTIC_V3_PROVENANCE
            ),
            "future_exactly_once_instruct_v3_terminal_status": instruct_diagnostic_v3_report.get(
                "status"
            ),
            "future_exactly_once_instruct_v3_supervisor_payload_invocations": instruct_diagnostic_v3_report.get(
                "supervisor_payload_invocations"
            ),
            "future_exactly_once_instruct_v3_state_reservations": instruct_diagnostic_v3_report.get(
                "state_reservations"
            ),
            "future_exactly_once_instruct_v3_process_starts": instruct_diagnostic_v3_report.get(
                "process_starts"
            ),
            "future_exactly_once_instruct_v3_runtime_executed": instruct_diagnostic_v3_report.get(
                "runtime_executed"
            ),
            "future_exactly_once_instruct_v3_required_absent_path_count": instruct_diagnostic_v3_report.get(
                "runtime_absent_path_count"
            ),
            "future_exactly_once_instruct_v3_historical_pipeline_snapshot_sha256": instruct_diagnostic_v3_report.get(
                "historical_pipeline_snapshot_sha256"
            ),
            "future_exactly_once_instruct_v3_current_pipeline_sha256": instruct_diagnostic_v3_report.get(
                "current_pipeline_sha256"
            ),
            "future_exactly_once_instruct_v3_current_pipeline_snapshot_required_to_match": instruct_diagnostic_v3_report.get(
                "current_pipeline_snapshot_required_to_match"
            ),
            "future_exactly_once_instruct_v3_product_acceptance_effect": False,
            "terminal_instruct_v4_one_start_child_nonzero_verified": instruct_diagnostic_v4_ok,
            "instruct_v4_terminal_package_sha256": instruct_diagnostic_v4_report.get(
                "package_sha256"
            ),
            "instruct_v4_terminal_authority_sha256": instruct_diagnostic_v4_report.get(
                "authority_sha256"
            ),
            "instruct_v4_terminal_review_sha256": instruct_diagnostic_v4_report.get(
                "terminal_review_sha256"
            ),
            "instruct_v4_terminal_manifest_sha256": instruct_diagnostic_v4_report.get(
                "terminal_manifest_sha256"
            ),
            "instruct_v4_terminal_authority_cardinality": instruct_diagnostic_v4_report.get(
                "authority_cardinality"
            ),
            "instruct_v4_terminal_authority_consumed": instruct_diagnostic_v4_report.get(
                "authority_consumed"
            ),
            "instruct_v4_terminal_state_reservations": instruct_diagnostic_v4_report.get(
                "state_reservations"
            ),
            "instruct_v4_terminal_process_starts": instruct_diagnostic_v4_report.get(
                "process_starts"
            ),
            "instruct_v4_terminal_runtime_present_path_count": instruct_diagnostic_v4_report.get(
                "runtime_present_path_count"
            ),
            "instruct_v4_terminal_planner_execution_permitted": instruct_diagnostic_v4_report.get(
                "planner_execution_permitted"
            ),
            "instruct_v4_terminal_rtl_correctness_conclusion": instruct_diagnostic_v4_report.get(
                "rtl_correctness_conclusion"
            ),
            "instruct_v4_terminal_historical_pipeline_snapshot_sha256": instruct_diagnostic_v4_report.get(
                "historical_pipeline_snapshot_sha256"
            ),
            "instruct_v4_terminal_current_pipeline_sha256": instruct_diagnostic_v4_report.get(
                "current_pipeline_sha256"
            ),
            "instruct_v4_terminal_current_pipeline_snapshot_required_to_match": instruct_diagnostic_v4_report.get(
                "current_pipeline_snapshot_required_to_match"
            ),
            "downstream_lifecycle_closed": lifecycle_closed,
            "s1_package_identity_sha256": s1_package_audit.get(
                "package_identity_sha256"
            ),
            "s1_package_audit_check_count": s1_package_audit.get("check_count"),
            "s1_repair_audit_check_count": s1_repair_audit.get("check_count"),
            "s1_fresh_reviewer_status": s1_fresh_review.get("review", {}).get(
                "status"
            ),
            "s1_terminal_reviewer_status": s1_terminal_review.get("review", {}).get(
                "status"
            ),
            "s1_authority_count": s1_authority_count,
            "s1_submission_intent_count": s1_intent_count,
            "s1_backend_job_count": s1_terminal_audit.get("cardinality", {}).get(
                "backend_job_count"
            ),
            "s1_backend_job_status": s1_terminal_audit.get("backend", {}).get(
                "status"
            ),
            "s1_terminal_audit_check_count": s1_terminal_audit.get("check_count"),
            "s1_downloaded_stdout_sha256": s1_terminal_audit.get("download", {})
            .get("stdout", {})
            .get("sha256"),
            "s1_model_quality_conclusion": None,
            "v8_execution_tree_sha256": v8_manifest.get("tree_sha256"),
            "v8_package_review_status": v8_review_acceptance.get("status"),
            "v8_terminal_review_status": v8_fresh_review.get("review", {}).get("status"),
            "v8_attempt_authority_present": V8_AUTHORITY.exists(),
            "v8_official_namespace_present": V8_RUN_ROOT.exists(),
            "v8_authority_count": v8_authority_count,
            "v8_submission_intent_count": v8_intent_count,
            "v8_backend_job_count": v8_result.get("job_count"),
            "v8_terminal_audit_check_count": v8_terminal_audit.get("check_count"),
            "v8_failure_taxonomy": v8_terminal_audit.get("failure_taxonomy"),
            "v8_model_quality_conclusion": v8_terminal_audit.get("model_quality_conclusion"),
            "v8_replay_or_resubmission_allowed": False,
            "v7_failure_taxonomy": v7_terminal_audit.get("failure_taxonomy"),
            "v7_model_quality_conclusion": v7_terminal_audit.get(
                "model_quality_conclusion"
            ),
            "v7_replay_or_attempt_0002_allowed": False,
            "v5_attempt_authority_present": V5_ATTEMPT_AUTHORITY.exists(),
            "v5_official_namespace_present": V5_RUN_ROOT.exists(),
            "v5_launch_readiness_present": V5_LAUNCH_READINESS.exists(),
            "v5_failure_taxonomy": v5_containment_l2.get("classification", {}).get(
                "failure_taxonomy"
            ),
            "v5_task_disposition": v5_containment_l2.get("task_disposition", {}).get(
                "disposition"
            ),
            "v5_attempt_0002_allowed": not bool(
                v5_containment_l2.get("classification", {}).get(
                    "attempt_0002_prohibited"
                )
            ),
            "v4_attempt_authority_granted": v4.get("attempt_authority_granted")
            if isinstance(v4, dict)
            else None,
            "v4_training_executed": v4.get("training_executed") if isinstance(v4, dict) else None,
            "selected_policy_id": selected_policy_id,
            "u280_inventory_verified": u280_ok,
            "stage_1_local_runtime": stage1,
            "stage_2_u280": stage2,
            "project_completion_eligible": False,
        },
        "input_bindings": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in existing_inputs
            if path.is_relative_to(ROOT)
        },
        "reproduction_command": "PYTHONDONTWRITEBYTECODE=1 python3 tools/audit_specification_stage.py",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{sha256(output)}  {output.name}\n", encoding="ascii"
        )
    return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
