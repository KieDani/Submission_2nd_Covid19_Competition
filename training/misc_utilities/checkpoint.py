from pathlib import Path
from argparse import ArgumentParser
import torch


def load_best_checkpoint(folder: Path, by: str = "loss", mode: str = "min", return_path: bool = False):
    "The loaded checkpoint will be loaded to the CPU."
    folder = Path(folder)
    if mode == "min":
        def is_better(new, best):
            return new[by] < best[by]
    elif mode == "max":
        def is_better(new, best):
            return new[by] > best[by]

    best_checkpoint = None
    best_path = None
    for path in folder.glob("checkpoint-*.pt"):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        if best_checkpoint is None or is_better(checkpoint["metrics"], best_checkpoint["metrics"]):
            best_checkpoint = checkpoint
            best_path = path

    if return_path:
        return best_checkpoint, best_path
    return best_checkpoint


def cli():
    parser = ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--metric", default="loss")
    parser.add_argument("--mode", choices=["min", "max"])
    args = parser.parse_args()
    ckp, best_path = load_best_checkpoint(args.folder, by=args.metric, mode=args.mode, return_path=True)
    print(str(best_path))

if __name__ == "__main__":
    cli()
