import hashlib
import json
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download


PROMPT_MANIFEST = Path(__file__).resolve().parents[1] / "PROMPT_MANIFEST.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(data_path: Path, labels_path: Path) -> list[dict]:
    rows = data_path.read_text(encoding="utf-8").splitlines()
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != len(labels):
        raise RuntimeError(f"PIQA rows/labels differ for {data_path.name}")
    records = []
    for row, label in zip(rows, labels, strict=True):
        record = json.loads(row)
        records.append(
            {
                "goal": record["goal"],
                "sol1": record["sol1"],
                "sol2": record["sol2"],
                "label": int(label),
            }
        )
    return records


def _record_hash(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cached_download(url: str, expected_sha256: str) -> Path:
    downloads = Path(datasets.config.DOWNLOADED_DATASETS_PATH)
    matches = []
    for metadata_path in downloads.glob("*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        path = metadata_path.with_suffix("")
        if metadata.get("url") == url and path.is_file() and _sha256(path) == expected_sha256:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"expected one verified cached PIQA archive, found {len(matches)}")
    return matches[0]


def _verified_records(
    split: str,
    data_path: Path,
    labels_path: Path,
    expected: dict,
) -> list[dict]:
    records = _records(data_path, labels_path)
    if len(records) != expected["count"]:
        raise RuntimeError(
            f"PIQA {split} record count differs from the manifest: "
            f"{len(records)} != {expected['count']}"
        )
    observed = _record_hash(records)
    if observed != expected["sha256"]:
        raise RuntimeError(
            f"PIQA {split} record hash differs from the manifest: "
            f"{observed} != {expected['sha256']}"
        )
    return records


def load_piqa(revision: str, **_: object) -> datasets.DatasetDict:
    manifest = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))
    spec = manifest["lm_eval"]["tasks"]["piqa"]
    if revision != spec["revision"]:
        raise ValueError(f"PIQA revision {revision} differs from {spec['revision']}")
    loader = Path(
        hf_hub_download(
            repo_id=spec["repository"],
            repo_type="dataset",
            filename="piqa.py",
            revision=revision,
        )
    )
    if _sha256(loader) != spec["loader_source_sha256"]:
        raise RuntimeError("pinned PIQA loader source hash differs from the manifest")
    source = spec["raw_sources"]["train_dev"]
    archive = _cached_download(source["url"], source["sha256"])
    archive_sha256 = _sha256(archive)
    if archive_sha256 != source["sha256"]:
        raise RuntimeError(
            "PIQA train/dev archive hash differs from the manifest: "
            f"{archive_sha256} != {source['sha256']}"
        )
    extracted = Path(datasets.DownloadManager().extract(str(archive)))
    roots = [
        path
        for path in extracted.rglob("physicaliqa-train-dev")
        if (path / "train.jsonl").is_file() and (path / "dev.jsonl").is_file()
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one extracted PIQA root, found {len(roots)}")
    root = roots[0]
    records = spec["records"]
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(
                _verified_records(
                    "train",
                    root / "train.jsonl",
                    root / "train-labels.lst",
                    records["train"],
                )
            ),
            "validation": datasets.Dataset.from_list(
                _verified_records(
                    "validation",
                    root / "dev.jsonl",
                    root / "dev-labels.lst",
                    records["validation"],
                )
            ),
        }
    )
