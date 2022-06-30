from argparse import ArgumentParser
import pandas as pd
from pathlib import Path

det_csvnames = ["non-covid.csv", "covid.csv"]
sev_csvnames = ["mild.csv", "moderate.csv", "severe.csv", "critical.csv"]

def main(outdir, pred_paths):
    outdir = Path(outdir)
    for path in pred_paths:
        path = Path(path)
        df = pd.read_csv(path)
        parts = path.stem.split("_")
        detsev = parts[1]
        subnum = parts[2]

        subdir = outdir / f"{detsev}_{subnum}"
        subdir.mkdir(parents=True)

        col = "pred_inf" if detsev == "det" else "pred_sev"
        csvnames = det_csvnames if detsev == "det" else sev_csvnames
        for cls, csvname in enumerate(csvnames):
            df[df[col] == cls]["patient"].to_csv(subdir/csvname, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("prediction", nargs="+")
    args = parser.parse_args()
    main(args.output, args.prediction)
