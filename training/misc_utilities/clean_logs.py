import time
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from training.misc_utilities.tensorboard import tensorboard2df


class ScanResult(Exception):
    def __init__(self, reason):
        self.reason = reason


def scandir(path: Path, stale_hours: float, min_steps: int):
    # check if cross validation
    if len(list(path.glob("cv*"))) > 0:
        for cvdir in path.glob("cv*"):
            if cvdir.is_dir():
                scandir(cvdir, stale_hours, min_steps)
        return

    event_files = list(path.glob("events.*"))
    if len(event_files) == 0:
        raise ScanResult("No Tensorboard")

    assert len(event_files) == 1, "Multiple tensorboard files exist"
    tbfile = event_files[0]
    tbdf = tensorboard2df(tbfile)
    if len(tbdf) == 0:
        raise ScanResult("Empty Tensorboard")

    last_update = tbdf["time"].max()
    is_stale = time.time() - last_update > stale_hours * 60 * 60

    max_step = tbdf["step"].max()
    if max_step < min_steps and is_stale:
        raise ScanResult(f"Too few steps ({max_step} < {min_steps})")


def cli():
    parser = ArgumentParser()
    parser.add_argument("root", help="Directory to scan")
    parser.add_argument(
        "--stale-hours",
        default=4,
        type=float,
        help="Number of hours that have to pass since the last update to mark this run as stale.",
    )
    parser.add_argument(
        "--min-steps",
        default=1000,
        type=int,
        help="If a tensorboard contains fewer steps than this number, the run will be marked.",
    )
    args = parser.parse_args()

    for rundir in tqdm(list(Path(args.root).iterdir()), unit="run"):
        if not rundir.is_dir():
            continue

        try:
            scandir(rundir, args.stale_hours, args.min_steps)
        except ScanResult as result:
            tqdm.write(f"{rundir}: {result.reason}")
        except Exception as err:
            tqdm.write(rundir)
            raise err


if __name__ == "__main__":
    cli()
