"""Command-line entry point for the analysis pipeline.

Examples
--------
python run_pipeline.py
python run_pipeline.py --data data/raw/loans_granting_model_dataset.csv
python run_pipeline.py --sample --max-rows 5000
python run_pipeline.py --sample --max-rows 1500 --skip-umap --skip-tsne
python run_pipeline.py --map fico=my_score_col --map state=region
"""

import argparse
from pathlib import Path

import config
from src import data_loader, pipeline


def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.
                                     RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None,
                        help="Path to the loans CSV file (real dataset)")
    parser.add_argument("--sample", action="store_true",
                        help="Generate and analyse synthetic demo data")
    parser.add_argument("--max-rows", type=int, default=config.MAX_ROWS,
                        help="Rows used for clustering (random sample)")
    parser.add_argument("--map", action="append", default=[],
                        metavar="CANONICAL=SOURCE",
                        help="Manual column mapping, repeatable")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()

    if args.sample and args.data:
        parser.error("Use either --sample or --data, not both.")

    if args.sample:
        from src.generate_sample_data import generate
        data_path = config.RAW_DATA_DIR / "sample_synthetic_loans.csv"
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            print("Generating synthetic demo data (not real borrower data)...")
            generate().to_csv(data_path, index=False)
        print("Using SYNTHETIC demo data. Results are not real-world findings.")
    elif args.data:
        data_path = args.data
    elif config.DEFAULT_DATA_FILE.exists():
        data_path = config.DEFAULT_DATA_FILE
    else:
        from src.generate_sample_data import generate
        data_path = config.RAW_DATA_DIR / "sample_synthetic_loans.csv"
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("Real Lending Club CSV not found. Generating synthetic demo data.")
        print("For real analysis, place the dataset at:", config.DEFAULT_DATA_FILE)
        generate().to_csv(data_path, index=False)

    if not Path(data_path).exists():
        parser.error(
            f"Dataset not found: {data_path}\n"
            "Add the CSV to data/raw/, or run: "
            "python run_pipeline.py --sample --max-rows 5000"
        )

    pipeline.run(data_path=data_path, max_rows=args.max_rows,
                 overrides=data_loader.parse_overrides(args.map),
                 skip_umap=args.skip_umap, skip_tsne=args.skip_tsne)


if __name__ == "__main__":
    main()
