#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

def scan_pair_folder(folder: Path, min_sizes):
    miss, small = [], []
    edfs = {p.stem: p for p in folder.glob("*.edf")}
    tsvs = {p.stem: p for p in folder.glob("*.tsv")}

    for stem, edf in edfs.items():
        tsv = tsvs.get(stem)
        if (not edf.exists()) or (tsv is None) or (not tsv.exists()):
            miss.append(stem)
            continue
        if edf.stat().st_size < min_sizes[".edf"] or tsv.stat().st_size < min_sizes[".tsv"]:
            small.append(stem)

    orphans_edf = sorted(set(edfs.keys()) - set(tsvs.keys()))
    orphans_tsv = sorted(set(tsvs.keys()) - set(edfs.keys()))

    return miss, small, orphans_edf, orphans_tsv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="../../../datasets/nch-sleep",
                    help="Base directory")
    ap.add_argument("--min-edf", type=int, default=50_000_000,
                    help="Minimum size of .edf (bytes)")
    ap.add_argument("--min-tsv", type=int, default=10_000,
                    help="Minimum size of .tsv (bytes)")
    ap.add_argument("--list", action="store_true",
                    help="Save detailed list in files .txt in the base directory")
    ap.add_argument("--json", action="store_true",
                    help="Emit a summary.json with the statistics")
    args = ap.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    min_sizes = {".edf": args.min_edf, ".tsv": args.min_tsv}

    summary = {}
    for label in ("positives", "negatives"):
        folder = base / label
        if not folder.exists():
            print(f"[{label}] ⚠️ folder not found: {folder}")
            summary[label] = {"missing": 0, "small": 0, "orphans_edf": 0, "orphans_tsv": 0, "total_edf": 0}
            continue

        miss, small, orph_edf, orph_tsv = scan_pair_folder(folder, min_sizes)
        total_edf = len(list(folder.glob("*.edf")))

        print(f"\n[{label}] base={folder}")
        print(f"  total .edf: {total_edf}")
        print(f"  missing pairs: {len(miss)}")
        print(f"  too small    : {len(small)}  (min .edf={min_sizes['.edf']:,}B, .tsv={min_sizes['.tsv']:,}B)")
        print(f"  orphans: .edf sem .tsv = {len(orph_edf)} | .tsv sem .edf = {len(orph_tsv)}")

        if miss[:10]:
            print("  ex. missing :", ", ".join(miss[:10]) + (" ..." if len(miss) > 10 else ""))
        if small[:10]:
            print("  ex. small   :", ", ".join(small[:10]) + (" ..." if len(small) > 10 else ""))
        if orph_edf[:10]:
            print("  ex. orphan edf:", ", ".join(orph_edf[:10]) + (" ..." if len(orph_edf) > 10 else ""))
        if orph_tsv[:10]:
            print("  ex. orphan tsv:", ", ".join(orph_tsv[:10]) + (" ..." if len(orph_tsv) > 10 else ""))

        summary[label] = {
            "missing": len(miss),
            "small": len(small),
            "orphans_edf": len(orph_edf),
            "orphans_tsv": len(orph_tsv),
            "total_edf": total_edf,
        }

        if args.list:
            (base / f"{label}_missing.txt").write_text("\n".join(miss) + ("\n" if miss else ""))
            (base / f"{label}_small.txt").write_text("\n".join(small) + ("\n" if small else ""))
            (base / f"{label}_orphans_edf.txt").write_text("\n".join(orph_edf) + ("\n" if orph_edf else ""))
            (base / f"{label}_orphans_tsv.txt").write_text("\n".join(orph_tsv) + ("\n" if orph_tsv else ""))

    print("\nResume:")
    for label, stats in summary.items():
        total = stats["total_edf"] or 1
        pct_bad = 100.0 * (stats["missing"] + stats["small"]) / total
        print(f"  {label}: total={stats['total_edf']} | missing={stats['missing']} | small={stats['small']} "
              f"| orphans(edf)={stats['orphans_edf']} | orphans(tsv)={stats['orphans_tsv']} "
              f"| % problematic={pct_bad:.1f}%")

    if args.json:
        (base / "summary_verify.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
