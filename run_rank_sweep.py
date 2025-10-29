from in_silico_perturber import InSilicoPerturber
from in_silico_perturber_stats import InSilicoPerturberStats
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--csv_root", required=True)
    ap.add_argument("--target_ensembl", required=True)
    ap.add_argument("--max_rank", type=int, default=2048)
    ap.add_argument("--perturb_type", choices=["activate","inhibit"], default="activate")
    ap.add_argument("--forward_batch_size", type=int, default=64)
    ap.add_argument("--nproc", type=int, default=4)
    args = ap.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.csv_root).mkdir(parents=True, exist_ok=True)

    for r in range(1, args.max_rank+1):
        isp = InSilicoPerturber(
            perturb_type=args.perturb_type,
            perturb_rank_direct_shift=r,
            genes_to_perturb=[args.target_ensembl],
            combos=0,
            anchor_gene=None,
            emb_mode="cell",
            forward_batch_size=args.forward_batch_size,
            nproc=args.nproc
        )
        out_dir = f"{args.out_root}/rank_{r}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        isp.perturb_data(
            model_directory=args.model_dir,
            input_data_file=args.dataset,
            output_directory=out_dir,
            output_prefix=f"target_{args.perturb_type}_rank{r}"
        )

        InSilicoPerturberStats(
            mode="aggregate_data",
            combos=0,
            anchor_gene=None
        ).get_stats(
            input_data_directory=out_dir,
            null_dist_data_directory=None,
            output_directory=args.csv_root,
            output_prefix=f"target_{args.perturb_type}_rank{r}"
        )

if __name__ == "__main__":
    main()