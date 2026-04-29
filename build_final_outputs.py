from __future__ import annotations

import argparse
import subprocess
import shutil
import sys
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

ALLOWED_EXTENSIONS = {".csv", ".json"}


def ensure_required_inputs(workspace_root: Path) -> None:
    """Fail fast if required source data folders are missing."""
    required_dirs = [workspace_root / "datamap", workspace_root / "bi_rawdata"]
    missing = [str(p) for p in required_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required folders are missing: " + ", ".join(missing)
        )


def run_analysis_pipeline(workspace_root: Path) -> None:
    """
    Re-generate outputs from raw data by running BI analysis scripts.
    This avoids relying on pre-shipped result files.
    """
    scripts = [
        workspace_root / "governance_bi" / "bi2_pkfk_governance.py",
        workspace_root / "governance_bi" / "bi3_health_sequential.py",
        workspace_root / "governance_bi" / "bi4_derived_features.py",
    ]

    missing_scripts = [str(p) for p in scripts if not p.exists()]
    if missing_scripts:
        raise FileNotFoundError(
            "Required analysis scripts are missing: " + ", ".join(missing_scripts)
        )

    for script in scripts:
        print(f"[run] {script.name}")
        # Use current Python interpreter so the same environment is reused.
        subprocess.run(
            [sys.executable, str(script)],
            cwd=str(workspace_root),
            check=True,
        )


def iter_source_files(source_dirs: Iterable[Path]) -> Iterable[Path]:
    for src_dir in source_dirs:
        if not src_dir.exists():
            continue
        for file_path in src_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                yield file_path


def build_destination_name(workspace_root: Path, file_path: Path) -> str:
    """
    Keep filename collision-safe by prefixing with the parent top-level folder.
    Example: governance_bi/bi3_output/bi3_summary.json -> bi3_output__bi3_summary.json
    """
    relative = file_path.relative_to(workspace_root)
    if len(relative.parts) >= 2:
        folder_hint = relative.parts[-2]
    else:
        folder_hint = relative.parts[0]
    return f"{folder_hint}__{file_path.name}"


def collect_outputs(workspace_root: Path, final_dir: Path, clean: bool) -> tuple[int, int]:
    final_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        for old in final_dir.iterdir():
            if old.is_file() and old.suffix.lower() in ALLOWED_EXTENSIONS:
                old.unlink()

    source_dirs = [
        workspace_root / "governance_bi" / "output",
        workspace_root / "governance_bi" / "bi3_output",
    ]

    copied = 0
    skipped = 0
    for src_file in iter_source_files(source_dirs):
        dest_name = build_destination_name(workspace_root, src_file)
        dest_file = final_dir / dest_name
        if dest_file.exists() and dest_file.stat().st_size == src_file.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src_file, dest_file)
        copied += 1

    return copied, skipped


def write_manifest(final_dir: Path, copied: int, skipped: int, generated_from_raw: bool) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_from_raw": generated_from_raw,
        "copied": copied,
        "skipped": skipped,
        "files": sorted(
            [p.name for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
        ),
    }
    manifest_path = final_dir / "final_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-generate BI outputs from raw data and collect CSV/JSON into final folder."
        )
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Workspace root path (default: parent of this script).",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=None,
        help="Final output folder path (default: <workspace-root>/final).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing CSV/JSON files in final folder before collecting.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip running raw-data analysis scripts and only collect currently generated outputs.",
    )

    args = parser.parse_args()
    workspace_root: Path = args.workspace_root.resolve()
    final_dir: Path = (args.final_dir or (workspace_root / "final")).resolve()

    ensure_required_inputs(workspace_root)
    if not args.skip_analysis:
        run_analysis_pipeline(workspace_root)

    copied, skipped = collect_outputs(workspace_root, final_dir, args.clean)
    write_manifest(final_dir, copied=copied, skipped=skipped, generated_from_raw=(not args.skip_analysis))

    print(f"workspace_root: {workspace_root}")
    print(f"final_dir: {final_dir}")
    print(f"analysis_rerun: {not args.skip_analysis}")
    print(f"copied: {copied}")
    print(f"skipped (same size): {skipped}")
    print("done")


if __name__ == "__main__":
    main()
