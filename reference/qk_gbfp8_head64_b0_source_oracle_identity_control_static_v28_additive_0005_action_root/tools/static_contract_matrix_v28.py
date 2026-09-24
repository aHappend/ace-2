#!/usr/bin/env python3
"""Static semantic matrix for the inert V28 B0 package; never opens official payload tensors."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema

PROJECT_ROOT = Path("/home/argustest/ace-2")
ACTION_ID = 'ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control:38571d38:additive-0005'
CLAIM = 'STATIC_ONLY_NO_EXECUTION_AUTHORITY'
EXPECTED_IDENTITY = {'accepted_v27': {'acceptance_file_sha256': 'dfecaa0566932cf8c1a7dfe0df37beda714d7924eceefb31d3c99168dd0dfa02', 'acceptance_self_sha256': 'a137497b85317c1b4baef9f3270eb9f4789aebdd438a210edea623cb41b723b5', 'action_id': 'ace2:qk-gbfp8-base-v27:role-enforced-acceptance:aaa22c19:additive-0004', 'final_action_tree_sha256': 'e376075566e9235c0447b3e727c3c8aa81872ce786e8eeb790eb1e43ebda7925', 'package_content_sha256': '38571d38526d73b71c69eecc9ecb229ced1d4c5798430a09b33b9e5161584df3', 'package_file_sha256': '99e20b22047199e3f2f3724f02c1b4e8a881f1c8170c3af851fe6089284f235d', 'positive_post_report_file_sha256': 'db990f0030711ca1e561781f312924b52caf97283b73793e5e02850e58368253', 'positive_post_report_sha256': 'd993863721e5642a7860a0dfc218e7512e4ea8798995818fce01c218698e4534', 'provenance_audit_file_sha256': '6ded31e4b1ba7cccb05d637eff2e37065736c740fb6df3c1e6b3c01a25efe45f', 'provenance_identity_sha256': 'c2af08e4949622f8db8d4f64496647d43d238a1519f627cc4ac75c036ba66cac', 'acceptance_decision': 'ACCEPT_STATIC_PACKAGE', 'claim_boundary': 'STATIC_ONLY_NO_EXECUTION_AUTHORITY'}, 'frozen_contract': {'benchmark_package_sha256': '3d36df763e775bc8f3fb5d106eaf842f7b74482c236b9697906bb93647c44832', 'binding_table_sha256': '87ab3deb25f77d89b0797d72004b4436f9541987da13a42f44a2bff7f9fa9655', 'input_token_ids_sha256': '1b8c972381a2c3d7c754d1d2879b4389a13aca3f1d485f703510702c1cf2eb86', 'lane_metadata_byte_count': 20057, 'lane_metadata_sha256': '4d5904378b9ccbabd434119b6a57f8d4266abf3d8e772c4c1bca98df785e7f3a', 'model_identity_sha256': '4a3c398aa381103aadf185da47b0631bce67539700e767eef588b761dc7827a7', 'official_evaluator_path': '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py', 'official_evaluator_sha256': '8ab74c7397006c9f419059f295613ce4a743a3e0b540176ba7cf6dbb6efd7f63', 'official_parser_path': '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py', 'official_parser_sha256': 'ff9216d58bcd1584361e17d8d1854f3bcb96923c17927b1f309861546807bdb1', 'official_result_schema_sha256': '07f818190e7f97032a6bc3724914f00f36070f429f1cd549c67e4e7b026ae720', 'sealed_set_id': 'w4a8-c02-attention-substage-trace-v2', 'selected_tensor_names': ['bf16.k_rope', 'bf16.q_rope', 'bf16.qk_scaled_scores'], 'tensor_bundle_byte_count': 1305797, 'tensor_bundle_sha256': '7cd2bbdbe611fdfce4d2cae66f6f20a43685087a35ce05c26c82d96749662175', 'tensor_record_count': 25, 'v21_result_byte_count': 8767, 'v21_result_file_sha256': '4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f'}, 'historical_v21_invocation': {'argv': ['/home/argustest/miniconda3/bin/python3.13', '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/tools/qk_gbfp8_head64_granularity_sweep_evaluator_worker_v21.py', '--mode', 'production', '--package', '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json', '--acceptance', '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json', '--irreversible-action-id', 'ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001'], 'cwd': '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root', 'environment': {'LANG': 'C', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0', 'TZ': 'UTC'}, 'invocation_sha256': '132555e567d7c99402b19207de4eb178798497c0f0d5beaa746e1c1f1eae3784', 'shell': False}, 'interpreter': {'path': '/home/argustest/miniconda3/bin/python3.13', 'sha256': 'fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad', 'version': '3.13.5'}, 'source_tensor_records': [{'tensor_name': 'bf16.q_rope', 'dtype': 'torch.bfloat16', 'sha256': '285e064ffea9571b7e3ed192a7083bf831dc5d444e0139558d3d51f084995429', 'shape': [1, 14, 41, 64]}, {'tensor_name': 'bf16.k_rope', 'dtype': 'torch.bfloat16', 'sha256': '401cdb0dc4a8def3190ac424f96df272c2bcf11241874759977d692845a19c0a', 'shape': [1, 2, 41, 64]}], 'oracle_tensor_record': {'tensor_name': 'bf16.qk_scaled_scores', 'dtype': 'torch.bfloat16', 'sha256': '49627e8364e534c61f4db8208d82798e53409c3c10d5d5e28c3c1462ef617765', 'shape': [1, 14, 41, 41]}, 'v21_result': {'byte_count': 8767, 'file_sha256': '4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f', 'irreversible_action_id': 'ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001', 'invocation_sha256': '132555e567d7c99402b19207de4eb178798497c0f0d5beaa746e1c1f1eae3784', 'result_self_sha256': '17b1809ce603504f1e91082c0cd77e1e7d082e242d406ba66a49af7583587689', 'terminal_reason_code': 'HARD_THRESHOLD_FAILED', 'terminal_status': 'FAILED_TERMINAL'}, 'carried_g_aggregate_references': [{'label': 'G1', 'source_action_id': 'ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001', 'source_result_file_sha256': '4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f', 'source_result_self_sha256': '17b1809ce603504f1e91082c0cd77e1e7d082e242d406ba66a49af7583587689', 'source_json_pointer': '/candidate_results/3'}, {'label': 'G2', 'source_action_id': 'ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001', 'source_result_file_sha256': '4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f', 'source_result_self_sha256': '17b1809ce603504f1e91082c0cd77e1e7d082e242d406ba66a49af7583587689', 'source_json_pointer': '/candidate_results/2'}, {'label': 'G4', 'source_action_id': 'ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001', 'source_result_file_sha256': '4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f', 'source_result_self_sha256': '17b1809ce603504f1e91082c0cd77e1e7d082e242d406ba66a49af7583587689', 'source_json_pointer': '/candidate_results/1'}, {'label': 'G8', 'source_action_id': 'ace2:qk-gbfp8-base-v21:execute-once:289140ba:additive-0001', 'source_result_file_sha256': '4901c835dd9b8700cb3a3dd5ecac54f2d1112f7f2f6e909a7ad37ee35ab8941f', 'source_result_self_sha256': '17b1809ce603504f1e91082c0cd77e1e7d082e242d406ba66a49af7583587689', 'source_json_pointer': '/candidate_results/0'}], 'future_execution_binding': {'future_execution_action_id': 'ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control-execute-once:38571d38:additive-0005', 'interpreter': {'path': '/home/argustest/miniconda3/bin/python3.13', 'sha256': 'fd4487ad3503b1aff61919108fd6f4f63c4245169fe9275f7c2ab76a1a53d3ad', 'version': '3.13.5'}, 'argv': ['/home/argustest/miniconda3/bin/python3.13', '/home/argustest/ace-2/reference/qk_gbfp8_head64_b0_source_oracle_identity_control_execution_v28_additive_0005_action_root/tools/execute_qk_gbfp8_head64_b0_source_oracle_identity_control_once_v28.py', '--mode', 'production', '--static-package', '/home/argustest/ace-2/reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0005_action_root/QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_PACKAGE.json', '--static-acceptance', '/home/argustest/ace-2/reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0005_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json', '--authority-envelope', '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/authority/exactly-once-envelope.json', '--output-sidecar', '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/primary/result/b0/source-oracle-aggregate-sidecar.json', '--terminal', '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/primary/result/b0/first-terminal.json', '--irreversible-action-id', 'ace2:qk-gbfp8-base-v28:b0-source-oracle-identity-control-execute-once:38571d38:additive-0005'], 'cwd': '/home/argustest/ace-2/reference/qk_gbfp8_head64_b0_source_oracle_identity_control_execution_v28_additive_0005_action_root', 'environment': {'LANG': 'C', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0', 'TZ': 'UTC'}, 'runtime_namespace': '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once', 'output_paths': {'authority_envelope': '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/authority/exactly-once-envelope.json', 'authority_consumption': '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/authority/authority-consumption.json', 'credential_consumption': '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/authority/credential-consumption.json', 'ledger': '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/ledger/exactly-once-ledger.json', 'sidecar': '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/primary/result/b0/source-oracle-aggregate-sidecar.json', 'terminal': '/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once/primary/result/b0/first-terminal.json'}, 'shell': False, 'execution_binding_sha256': '06142230fd7974f22e9cf2f0f474c17ecfb0ba688689a4fc7b6c190cd2cc1f31'}}
PREDECESSOR_SNAPSHOTS = [{'version': 'V21_ACTION', 'entry_count': 30, 'file_count': 25, 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root', 'tree_sha256': 'f71d1652d9b488547386e3e476e5fd88a81e10e1a0739300e167acafb2bd28f6', 'status': 'IMMUTABLE_ACCEPTED_EXECUTION_EVIDENCE', 'hash_scope': 'ENTRIES_ONLY'}, {'version': 'V21_RUNTIME', 'entry_count': 11, 'file_count': 5, 'root': '/home/argustest/ace-2/runtime/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_289140ba', 'tree_sha256': 'ac7cfdfddec92340e63f9892230c35f263da346b986fb843517da0ddaac7dd05', 'status': 'IMMUTABLE_TERMINAL_RUNTIME_EVIDENCE', 'hash_scope': 'ENTRIES_ONLY'}, {'version': 'V22', 'entry_count': 11, 'file_count': 7, 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_source_oracle_attribution_static_v22_action_root', 'tree_sha256': '7bb7ffebf39c0a5fda8d85da24e05100e8cc6674d2f72a06be1321dd5b6f4dcb', 'status': 'IMMUTABLE_STATIC_EVIDENCE', 'hash_scope': 'ENTRIES_ONLY'}, {'version': 'V23', 'entry_count': 17, 'file_count': 11, 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_source_oracle_verifier_hardening_static_v23_action_root', 'tree_sha256': '35f705b9ae3a85eb62654dbea6fd2b624ba122cda1d4b6feee27188922ebe93a', 'status': 'IMMUTABLE_ACCEPTED_STATIC_EVIDENCE', 'hash_scope': 'ENTRIES_ONLY'}, {'version': 'V24', 'entry_count': 15, 'file_count': 10, 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_bytecode_free_verifier_static_v24_action_root', 'tree_sha256': '19e302a9cae9dc0f341253c45e3fad63a8ef5747b80ab786e32a394d27d5b672', 'status': 'IMMUTABLE_TERMINAL_REJECTION', 'hash_scope': 'ENTRIES_ONLY'}, {'version': 'V25', 'entry_count': 18, 'file_count': 13, 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_literal_b_mode_seal_static_v25_action_root', 'tree_sha256': 'ca176ab9116e7e0285dc943dbe0123a07cf728df3b6d6cb6a0485c636a24bdae', 'status': 'IMMUTABLE_TERMINAL_FAILURE', 'hash_scope': 'ENTRIES_ONLY'}, {'version': 'V26', 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_launch_harness_repair_static_v26_action_root', 'tree_sha256': '86c17c45557c92c58ada0afd252fa960184f64c1692df72e085d2e07563427ec', 'status': 'PROVENANCE_INVALID_ACCEPTANCE_PRESERVED_NOT_REINTERPRETED', 'hash_scope': 'ROOT_AND_ENTRIES'}, {'version': 'V27', 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_role_enforced_acceptance_static_v27_action_root', 'tree_sha256': 'e376075566e9235c0447b3e727c3c8aa81872ce786e8eeb790eb1e43ebda7925', 'status': 'IMMUTABLE_ACCEPTED_STATIC_PREDECESSOR', 'hash_scope': 'ROOT_AND_ENTRIES'}, {'version': 'V28_FAILED_ADDITIVE_0001', 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_action_root', 'tree_sha256': '35f42838f4d1979bc9675069a7edd0e5a90d40eb3835bc6aadc95ec6e98870d5', 'status': 'IMMUTABLE_FAILED_SEALED_V28_ADDITIVE_0001', 'failure_file_sha256': '1e32376b1ccc9b34247c426b3eda3f0ac20bc5c7983067813f5c69a061646908', 'failure_taxonomy': 'LAUNCH_NEGATIVE_HARNESS_GUARD_ORDER', 'hash_scope': 'ROOT_AND_ENTRIES'}, {'version': 'V28_FAILED_ADDITIVE_0002', 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0002_action_root', 'tree_sha256': '28cdd5ee628ea836cb968f76cfeca2a274f4cebd22ba0d4c29e2aed4d99b863f', 'status': 'IMMUTABLE_FAILED_SEALED_V28_ADDITIVE_0002', 'failure_file_sha256': '822aadaf125d274da3bb4ea20f93309c3d4b7572dfd806af23266fecb43b8b22', 'failure_taxonomy': 'LAUNCH_NEGATIVE_HARNESS_GUARD_ORDER', 'hash_scope': 'ROOT_AND_ENTRIES'}, {'version': 'V28_FAILED_ADDITIVE_0003', 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0003_action_root', 'tree_sha256': '63f1c57a690939aa161ff8978562243d80ed866c403bf3518ece9976a7ac682d', 'status': 'IMMUTABLE_FAILED_SEALED_V28_ADDITIVE_0003', 'failure_file_sha256': '5b16c9c33f165b38c9fe2796da6bbe446eac9abf96513571b759376b539d1f9c', 'failure_taxonomy': 'GENERATED_ENTRYPOINT_MISSING_AUDIT_BINDING', 'hash_scope': 'ROOT_AND_ENTRIES'}, {'version': 'V28_FAILED_ADDITIVE_0004', 'root': '/home/argustest/ace-2/reference/qk_gbfp8_head64_b0_source_oracle_identity_control_static_v28_additive_0004_action_root', 'tree_sha256': '3d8f0424c8323b5a7e1dd2663ddda692d0e95a592dcbc0aea1c0d35f93e7ec1b', 'status': 'IMMUTABLE_FAILED_SEALED_V28_ADDITIVE_0004', 'failure_file_sha256': '43eedd14cc35786f53719d11854de1915e9abeafecdb39306d33df5ea3b0460d', 'failure_taxonomy': 'PREDECESSOR_HASH_SCOPE_CONVENTION_MISMATCH', 'hash_scope': 'ROOT_AND_ENTRIES'}]
V27_PACKAGE = PROJECT_ROOT / "reference/qk_gbfp8_head64_role_enforced_acceptance_static_v27_action_root/QK_GBFP8_HEAD64_ROLE_ENFORCED_ACCEPTANCE_STATIC_V27_PACKAGE.json"
V27_ACCEPTANCE = PROJECT_ROOT / "reference/qk_gbfp8_head64_role_enforced_acceptance_static_v27_action_root/review/FRESH_L2_STATIC_ACCEPTANCE.json"
V27_POST = PROJECT_ROOT / "build/v27-role-enforced-acceptance-static-0004/fresh-l2-post-0001/QK_GBFP8_HEAD64_ROLE_ENFORCED_ACCEPTANCE_STATIC_V27_POST_ACCEPTANCE_REPORT.json"
V27_AUDIT = PROJECT_ROOT / "build/v27-role-enforced-acceptance-static-0004/fresh-l2-provenance-audit-0001/QK_GBFP8_HEAD64_ROLE_ENFORCED_ACCEPTANCE_STATIC_V27_PROVENANCE_AUDIT.json"
V21_PACKAGE = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/QK_GBFP8_HEAD64_GRANULARITY_SWEEP_EXECUTION_V21_ORDER_INDEPENDENT_BINDING_TERMINAL_COVERAGE_PACKAGE.json"
V21_BINDINGS = PROJECT_ROOT / "reference/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_action_root/bindings/C02_EXACT_BINDINGS_25.json"
V21_RESULT = PROJECT_ROOT / "runtime/qk_gbfp8_head64_granularity_sweep_execution_v21_order_independent_binding_terminal_coverage_289140ba/primary/result/base/result.json"
INTERPRETER = Path('/home/argustest/miniconda3/bin/python3.13')
OFFICIAL_EVALUATOR = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_evaluator_static_v8.py')
OFFICIAL_PARSER = Path('/home/argustest/ace-2/reference/qk_gbfp8_head64_granularity_sweep_static_v8_action_root/reference/qk_gbfp8_head64_granularity_sweep_c02_static_v8.py')
FUTURE_RUNTIME_NAMESPACE = Path('/home/argustest/ace-2/runtime/qk_gbfp8_head64_b0_source_oracle_identity_control_v28_38571d38_additive_0005_once')


class ContractError(RuntimeError):
    pass


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ContractError(detail)


def compact_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes(); value = json.loads(raw.decode("ascii", "strict"))
    require(type(value) is dict and compact_bytes(value) == raw, f"canonical JSON: {path}")
    return value, raw


def inventory(root: Path, include_root: bool = True) -> list[dict[str, Any]]:
    records = []
    if include_root:
        info = os.lstat(root); records.append({"kind": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "path": "."})
    for path in sorted(root.rglob("*")):
        info = os.lstat(path); relative = path.relative_to(root).as_posix()
        require(not stat.S_ISLNK(info.st_mode), f"symlink: {relative}")
        mode = f"{stat.S_IMODE(info.st_mode):04o}"
        if stat.S_ISDIR(info.st_mode): records.append({"kind": "directory", "mode": mode, "path": relative})
        elif stat.S_ISREG(info.st_mode): records.append({"kind": "file", "mode": mode, "path": relative, "sha256": sha256_file(path), "size": info.st_size})
        else: raise ContractError(f"unsupported entry: {relative}")
    return records


def tree_hash(root: Path, include_root: bool = True) -> str:
    return sha256_bytes(compact_bytes(inventory(root, include_root)))


def verify_self_hash(value: dict[str, Any], field: str) -> None:
    observed = value.get(field); unhashed = dict(value); unhashed.pop(field, None)
    require(observed == sha256_bytes(compact_bytes(unhashed)), f"self hash: {field}")


def normalize_preacceptance_records(records: list[dict[str, Any]], package_name: str, acceptance_relative: str) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        if record["path"] == acceptance_relative: continue
        item = dict(record)
        if item["path"] == "review": item["mode"] = "0755"
        if item["path"] == package_name:
            item.pop("sha256", None); item["sha256_field"] = "package_content_sha256"; item["sha256_scope"] = "canonical_object_without_package_content_sha256"
        normalized.append(item)
    return normalized


def validate_package(package: dict[str, Any]) -> None:
    require(package["action_identity"]["action_id"] == ACTION_ID, "action identity")
    require(package["action_identity"]["predecessor_action_id"] == 'ace2:qk-gbfp8-base-v27:role-enforced-acceptance:aaa22c19:additive-0004', "predecessor identity")
    require(package["claim_boundary"]["claim"] == CLAIM, "claim boundary")
    require(package["claim_boundary"]["execution_authorized"] is False, "execution authority absent")
    require(package["claim_boundary"]["runtime_namespace_materialized"] is False, "runtime namespace absent")
    require(package["claim_boundary"]["authority_or_credential_materialized"] is False, "authority absent")
    require(package["claim_boundary"]["consumption_or_ledger_materialized"] is False, "consumption absent")
    require(package["claim_boundary"]["result_or_terminal_materialized"] is False, "result absent")
    require(package["identity_bindings"] == EXPECTED_IDENTITY, "identity bundle")
    control = package["source_oracle_identity_control"]
    require(control["permanently_nonselecting"] is True and control["quantization_applied"] is False, "B0 inert control")
    require(control["residual_control_labels"] == ["B0", "G1", "G2", "G4", "G8"], "residual labels")
    require(control["row_class_labels"] == ["ALL_CAUSAL_ROWS", "SINGLETON_ROWS", "ORACLE_TIED_TOP_ROWS", "ORACLE_UNIQUE_POSITIVE_MARGIN_ROWS"], "row classes")
    require(package["predecessor_preservation"]["v26_provenance_status"] == "PROVENANCE_INVALID_ACCEPTANCE_PRESERVED_NOT_REINTERPRETED", "V26 provenance semantics")
    require(package["predecessor_preservation"]["failed_additive_predecessor"]["status"] == "IMMUTABLE_FAILED_SEALED_V28_ADDITIVE_0004", "failed additive predecessor semantics")
    require(package["future_exactly_once_authority_envelope"]["materialized_by_this_package"] is False, "authority absent")
    require(package["fresh_l2_review"]["status"] == "PENDING_INDEPENDENT_REVIEW", "review pending")


def verify_inventory(action_root: Path, package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    acceptance_relative = package["static_file_policy"]["acceptance_relative_path"]
    acceptance = action_root / acceptance_relative
    expected = set(package["static_file_policy"]["allowed_relative_files"])
    if not allow_acceptance:
        expected.remove(acceptance_relative); require(not os.path.lexists(acceptance), "Engineer materialized acceptance")
    else: require(acceptance.is_file(), "acceptance absent")
    records = inventory(action_root)
    observed = {item["path"] for item in records if item["kind"] == "file"}
    require(observed == expected, f"exact inventory: {sorted(observed)}")
    forbidden_suffixes = tuple(package["bytecode_free_verifier_contract"]["forbidden_inventory_suffixes"])
    for item in records:
        require("__pycache__" not in item["path"] and not item["path"].endswith(forbidden_suffixes), f"forbidden inventory: {item['path']}")
        expected_mode = "0444" if item["kind"] == "file" else ("0755" if not allow_acceptance and item["path"] == "review" else "0555")
        require(item["mode"] == expected_mode, f"mode seal: {item['path']}")
    normalized = normalize_preacceptance_records(records, package["static_file_policy"]["package_relative_path"], acceptance_relative)
    require(normalized == package["mode_seal"]["pre_acceptance_records"], "embedded pre-acceptance mode manifest")
    return {"file_count": len(observed), "mode_seal_exact": True, "pre_acceptance_tree_sha256": sha256_bytes(compact_bytes(normalized))}


def verify_generated_files(action_root: Path, package: dict[str, Any]) -> dict[str, Any]:
    for relative, expected in package["generated_files"].items():
        path = action_root / relative
        require(path.is_file() and path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"], f"generated file: {relative}")
    return {"generated_file_count": len(package["generated_files"])}


def verify_identities(package: dict[str, Any]) -> dict[str, Any]:
    accepted = EXPECTED_IDENTITY["accepted_v27"]
    require(sha256_file(V27_PACKAGE) == accepted["package_file_sha256"], "V27 package drift")
    require(sha256_file(V27_ACCEPTANCE) == accepted["acceptance_file_sha256"], "V27 acceptance drift")
    require(sha256_file(V27_POST) == accepted["positive_post_report_file_sha256"], "V27 post drift")
    require(sha256_file(V27_AUDIT) == accepted["provenance_audit_file_sha256"], "V27 audit drift")
    require(tree_hash(V27_PACKAGE.parent) == accepted["final_action_tree_sha256"], "V27 final tree drift")
    require(sha256_file(V21_BINDINGS) == EXPECTED_IDENTITY["frozen_contract"]["binding_table_sha256"], "binding drift")
    require(sha256_file(V21_RESULT) == EXPECTED_IDENTITY["v21_result"]["file_sha256"], "V21 result drift")
    require(sha256_file(INTERPRETER) == EXPECTED_IDENTITY["interpreter"]["sha256"], "interpreter drift")
    require(sha256_file(OFFICIAL_EVALUATOR) == EXPECTED_IDENTITY["frozen_contract"]["official_evaluator_sha256"], "evaluator drift")
    require(sha256_file(OFFICIAL_PARSER) == EXPECTED_IDENTITY["frozen_contract"]["official_parser_sha256"], "parser drift")
    v21, _ = canonical(V21_PACKAGE)
    require(v21["production_evaluator_invocation"] == EXPECTED_IDENTITY["historical_v21_invocation"], "historical argv/cwd/environment drift")
    for snapshot in PREDECESSOR_SNAPSHOTS:
        include_root = snapshot["hash_scope"] == "ROOT_AND_ENTRIES"
        require(tree_hash(Path(snapshot["root"]), include_root) == snapshot["tree_sha256"], f"predecessor drift: {snapshot['version']}")
    return {"identity_bundle_sha256": package["identity_bindings_sha256"], "predecessor_snapshot_count": len(PREDECESSOR_SNAPSHOTS), "v26_provenance_invalid_preserved": True}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path); require(spec is not None and spec.loader is not None, f"module spec: {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def verify_identity_classifier(action_root: Path) -> dict[str, Any]:
    core = load_module(action_root / "tools/qk_gbfp8_head64_source_oracle_identity_control_core_v28.py", "v28_identity_classifier")
    oracle = core.synthetic_rows(); exact = core.classify_rows(deepcopy(oracle), oracle)
    require(exact["classification"] == "SOURCE_ORACLE_MATCH", "match classifier")
    digests = set()
    for index in range(574):
        source = deepcopy(oracle); source[index][0] += 1; result = core.classify_rows(source, oracle)
        require(result["classification"] == "SOURCE_ORACLE_MISMATCH" and result["source_oracle_mismatch_count"] == 1, f"mismatch classifier: {index}")
        digests.add(result["mismatch_membership_sha256"])
    require(len(digests) == 574, "position discrimination")
    return {"identity_position_test_count": 575, "unique_mismatch_membership_digest_count": 574}


def _aggregate(row_class: str, count: int) -> dict[str, Any]:
    digest = sha256_bytes(compact_bytes([row_class, count]))
    empty = sha256_bytes(compact_bytes([]))
    return {
        "row_class": row_class, "row_count": count,
        "score_agreement_count": count, "score_mismatch_count": 0,
        "top_key_agreement_count": count, "top_key_mismatch_count": 0,
        "margin_sign_agreement_count": count, "margin_sign_mismatch_count": 0,
        "sum_signed_error_q12_20_lsb": 0, "sum_absolute_error_q12_20_lsb": 0,
        "sum_squared_error_q40_40_lsb2": 0, "maximum_absolute_error_q12_20_lsb": 0,
        "row_membership_sha256": digest, "score_mismatch_membership_sha256": empty,
        "top_key_mismatch_membership_sha256": empty, "margin_sign_mismatch_membership_sha256": empty,
    }


def sidecar_baseline() -> dict[str, Any]:
    refs = {item["label"]: item for item in EXPECTED_IDENTITY["carried_g_aggregate_references"]}
    value = {
        "schema_version": 1,
        "artifact_kind": "qk_gbfp8_head64_source_oracle_identity_control_v28_b0_aggregate_sidecar",
        "v28_action_id": ACTION_ID,
        "claim_boundary": "SEPARATELY_AUTHORIZED_DIAGNOSTIC_ONLY_NONSELECTING",
        "authority_envelope_sha256": "0" * 64,
        "control": {
            "base_selection_permitted": False, "candidate_selection_permitted": False,
            "checkpoint_176_or_chat_advancement_permitted": False, "comparison_only": True,
            "control_id": "B0_SOURCE_QK_IDENTITY", "may_count_as_any_quantization_group": False,
            "may_satisfy_base_selection": False, "official_result_schema_influence_permitted": False,
            "permanently_nonselecting": True, "quantization_applied": False,
            "stage2_authorization_permitted": False, "threshold_change_permitted": False,
            "tensor_values_exposed": False,
        },
        "identity_bindings": EXPECTED_IDENTITY,
        "outcome": "SOURCE_ORACLE_MATCH",
        "partition_accounting": {
            "all_causal_row_count": 574, "singleton_row_count": 14,
            "oracle_tied_top_row_count": 214, "oracle_unique_positive_margin_row_count": 346,
            "partition_sum": 574, "partitions_disjoint_and_complete": True,
        },
        "row_class_aggregates": [
            _aggregate("ALL_CAUSAL_ROWS", 574), _aggregate("SINGLETON_ROWS", 14),
            _aggregate("ORACLE_TIED_TOP_ROWS", 214), _aggregate("ORACLE_UNIQUE_POSITIVE_MARGIN_ROWS", 346),
        ],
        "residual_control_aggregates": [
            {"label": "B0", "role": "SOURCE_QK_RECONSTRUCTION_ORACLE_IDENTITY_CONTROL", "applies_quantization": False, "permanently_nonselecting": True, "outcome": "SOURCE_ORACLE_MATCH"},
            *[{"label": label, "role": "CARRIED_IMMUTABLE_G_AGGREGATE_REFERENCE_ONLY", "carried_immutable_aggregate_reference": refs[label]} for label in ("G1", "G2", "G4", "G8")],
        ],
    }
    value["sidecar_sha256"] = sha256_bytes(compact_bytes(value)); return value


def validate_sidecar(value: dict[str, Any], schema: dict[str, Any]) -> None:
    jsonschema.Draft202012Validator.check_schema(schema); jsonschema.Draft202012Validator(schema).validate(value); verify_self_hash(value, "sidecar_sha256")
    require([item["row_class"] for item in value["row_class_aggregates"]] == ["ALL_CAUSAL_ROWS", "SINGLETON_ROWS", "ORACLE_TIED_TOP_ROWS", "ORACLE_UNIQUE_POSITIVE_MARGIN_ROWS"], "row classes exact once")
    require([item["label"] for item in value["residual_control_aggregates"]] == ["B0", "G1", "G2", "G4", "G8"], "residual labels exact once")
    mismatch = 0
    for item in value["row_class_aggregates"]:
        for prefix in ("score", "top_key", "margin_sign"):
            require(item[f"{prefix}_agreement_count"] + item[f"{prefix}_mismatch_count"] == item["row_count"], f"{prefix} accounting")
            mismatch += item[f"{prefix}_mismatch_count"]
    expected = "SOURCE_ORACLE_MATCH" if mismatch == 0 else "SOURCE_ORACLE_MISMATCH"
    require(value["outcome"] == expected and value["residual_control_aggregates"][0]["outcome"] == expected, "fail-closed attribution outcome")
    if expected == "SOURCE_ORACLE_MATCH":
        for item in value["row_class_aggregates"]:
            require(item["sum_signed_error_q12_20_lsb"] == item["sum_absolute_error_q12_20_lsb"] == item["sum_squared_error_q40_40_lsb2"] == item["maximum_absolute_error_q12_20_lsb"] == 0, "match error totals")


def authority_baseline() -> dict[str, Any]:
    hashes = {
        "package_file_sha256": "1" * 64, "package_content_sha256": "2" * 64,
        "fresh_l2_acceptance_file_sha256": "3" * 64, "fresh_l2_acceptance_self_sha256": "4" * 64,
        "accepted_action_tree_sha256": "5" * 64,
    }
    binding = EXPECTED_IDENTITY["future_execution_binding"]
    value = {
        "schema_version": 1,
        "artifact_kind": "qk_gbfp8_head64_source_oracle_identity_control_v28_future_exactly_once_authority_envelope",
        "source_static_action_id": ACTION_ID,
        "future_execution_action_id": binding["future_execution_action_id"],
        "source_static_acceptance": deepcopy(hashes),
        "manager_admission": {"actor": "manager", "decision": "ADMIT_B0_ONCE", "blocked": False, "event_sha256": "6" * 64, "source_static_acceptance": deepcopy(hashes)},
        "operator_authority": {"actor": "operator", "decision": "AUTHORIZE_B0_ONCE", "affirmative": True, "provenance_event_sha256": "7" * 64, "source_static_acceptance": deepcopy(hashes), "execution_binding_sha256": binding["execution_binding_sha256"]},
        "execution_binding": binding,
        "exactly_once": {
            "attempt_ordinal": 1, "authority_consumption_budget": 1, "credential_consumption_budget": 1,
            "official_payload_open_budget": 1, "official_evaluator_invocation_budget": 1,
            "official_target_process_start_budget": 1, "retry_replay_resume_repair_permitted": False,
            "runtime_namespace_must_not_preexist": True, "terminal_sealing_required_on_every_outcome": True,
            "terminal_states": ["SOURCE_ORACLE_MATCH", "SOURCE_ORACLE_MISMATCH", "FAILED_TERMINAL"],
        },
        "identity_bundle_sha256": '6b5fb3ee49fd592519deca55897c3e2309e0d82c62040013b985f1dd8dfe965a',
        "reservation_nonce_sha256": "8" * 64,
    }
    value["authority_envelope_sha256"] = sha256_bytes(compact_bytes(value)); return value


def validate_authority(value: dict[str, Any], schema: dict[str, Any], require_runtime_absent: bool = True) -> None:
    jsonschema.Draft202012Validator.check_schema(schema); jsonschema.Draft202012Validator(schema).validate(value); verify_self_hash(value, "authority_envelope_sha256")
    require(value["source_static_acceptance"] == value["manager_admission"]["source_static_acceptance"] == value["operator_authority"]["source_static_acceptance"], "authority hash bindings")
    require(value["operator_authority"]["execution_binding_sha256"] == value["execution_binding"]["execution_binding_sha256"], "execution binding hash")
    if require_runtime_absent: require(not os.path.lexists(value["execution_binding"]["runtime_namespace"]), "runtime namespace preexists")


def apply_operations(value: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    changed = deepcopy(value)
    for operation in operations:
        path = operation["path"]; parent: Any = changed
        for key in path[:-1]: parent = parent[key]
        key = path[-1]
        if operation["operation"] == "set": parent[key] = operation["value"]
        elif operation["operation"] == "delete": del parent[key]
        elif operation["operation"] == "append": parent[key].append(operation["value"])
        else: raise ContractError(f"unknown mutation: {operation['operation']}")
    for field in ("sidecar_sha256", "authority_envelope_sha256", "package_content_sha256"):
        if field in changed:
            changed.pop(field); changed[field] = sha256_bytes(compact_bytes(changed))
    return changed


def verify_negative_catalog(action_root: Path, package: dict[str, Any], sidecar_schema: dict[str, Any], authority_schema: dict[str, Any]) -> dict[str, Any]:
    catalog, _ = canonical(action_root / "fixtures/STATIC_NEGATIVE_CASES.json"); verify_self_hash(catalog, "fixture_sha256")
    rejected = []
    for case in catalog["cases"]:
        try:
            if case["target"] == "sidecar": validate_sidecar(apply_operations(sidecar_baseline(), case["operations"]), sidecar_schema)
            elif case["target"] == "authority": validate_authority(apply_operations(authority_baseline(), case["operations"]), authority_schema)
            elif case["target"] == "package": validate_package(apply_operations(package, case["operations"]))
            else: raise ContractError("unknown target")
        except Exception:
            rejected.append(case["id"])
        else:
            raise ContractError(f"negative accepted: {case['id']}")
    require(len(rejected) == catalog["case_count"], "negative count")
    return {"static_negative_case_count": len(rejected), "rejected_case_ids": rejected}


def verify_no_live_effects(action_root: Path, package: dict[str, Any], allow_acceptance: bool) -> dict[str, Any]:
    acceptance = str(action_root / package["static_file_policy"]["acceptance_relative_path"])
    for path in package["forbidden_live_paths"]:
        if allow_acceptance and path == acceptance: continue
        require(not os.path.lexists(path), f"forbidden live path: {path}")
    return {
        "official_payload_open_count": 0,
        "official_evaluator_invocation_count": 0,
        "official_target_process_starts": 0,
        "runtime_namespace_materialized": False,
        "authority_or_credential_materialized": False,
        "result_or_terminal_materialized": False,
    }


def verify_candidate(action_root: Path, allow_acceptance: bool, launch_proof: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    package_path = action_root / "QK_GBFP8_HEAD64_SOURCE_ORACLE_IDENTITY_CONTROL_STATIC_V28_PACKAGE.json"
    package, package_raw = canonical(package_path); verify_self_hash(package, "package_content_sha256"); validate_package(package)
    sidecar_schema, _ = canonical(action_root / package["source_oracle_identity_control"]["sidecar_schema_path"])
    authority_schema, _ = canonical(action_root / package["future_exactly_once_authority_envelope"]["schema_path"])
    validate_sidecar(sidecar_baseline(), sidecar_schema); validate_authority(authority_baseline(), authority_schema)
    negative = verify_negative_catalog(action_root, package, sidecar_schema, authority_schema)
    report = {
        "artifact_kind": "qk_gbfp8_head64_source_oracle_identity_control_static_v28_candidate_verifier_report",
        "claim_boundary": CLAIM,
        "generated_files": verify_generated_files(action_root, package),
        "identity_bindings": verify_identities(package),
        "identity_classifier": verify_identity_classifier(action_root),
        "inventory": verify_inventory(action_root, package, allow_acceptance),
        "launch_proof": launch_proof,
        "live_effects": verify_no_live_effects(action_root, package, allow_acceptance),
        "row_class_count": 4,
        "residual_control_label_count": 5,
        "safe_sidecar_semantics": True,
        "future_exactly_once_schema_semantics": True,
        "static_negative_case_count": negative["static_negative_case_count"],
        "package_content_sha256": package["package_content_sha256"],
        "package_file_sha256": sha256_bytes(package_raw),
        "official_payload_open_count": 0,
        "official_evaluator_invocation_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_V28_STATIC_CANDIDATE_PENDING_FRESH_L2" if not allow_acceptance else "PASS_V28_STATIC_CANDIDATE_WITH_ACCEPTANCE",
    }
    negative_report = {
        "artifact_kind": "qk_gbfp8_head64_source_oracle_identity_control_static_v28_negative_fixture_report",
        "claim_boundary": CLAIM,
        **negative,
        "official_payload_open_count": 0,
        "official_evaluator_invocation_count": 0,
        "official_target_process_starts": 0,
        "status": "PASS_V28_IDENTITY_AND_AUTHORITY_MUTATION_REJECTION",
    }
    report["report_sha256"] = sha256_bytes(compact_bytes(report)); negative_report["report_sha256"] = sha256_bytes(compact_bytes(negative_report))
    return report, negative_report
