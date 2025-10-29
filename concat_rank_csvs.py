import pandas as pd
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_root", required=True)
    ap.add_argument("--prefix", default="target_activate_rank")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--start_rank", type=int, default=1)
    ap.add_argument("--end_rank", type=int, default=2048)
    args = ap.parse_args()

    frames = []
    for r in range(args.start_rank, args.end_rank+1):
        p = Path(args.csv_root)/f"{args.prefix}{r}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["rank"] = r
        frames.append(df)
    if not frames:
        raise SystemExit("No CSVs found to merge.")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(out)} rows")

if __name__ == "__main__":
    main()