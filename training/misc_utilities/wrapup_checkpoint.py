from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence
import torch


def assert_check_subset(item, other, ignore_paths: Sequence[str] = None):
    """Check if `item` is equivalent to `other`

    @ignore_paths: Dot (.) separated paths that should not be checked (e.g.: config.modelconfig.size)
    """

    if ignore_paths is None:
        ignore_paths = []

    def check_subset(item, other, path: Sequence[str]):
        # ignore this path?
        for ign in ignore_paths:
            if ign == ".".join(path):
                return []

        mismatches = []
        if isinstance(item, dict):
            if not isinstance(other, dict):
                return [(path, item, other)]

            for key, value in item.items():
                mismatches.extend(check_subset(value, other[key], path=path + [key]))
        elif isinstance(item, list):
            for i, (child, child_other) in enumerate(zip(item, other)):
                mismatches.extend(
                    check_subset(child, child_other, path=path + [str(i)])
                )
        elif item != other:
            return [(path, item, other)]
        return mismatches

    mismatches = check_subset(item, other, path=["config"])
    if len(mismatches) > 0:
        mm_str = "\n".join(
            [
                f"({'.'.join(path)})\ncheckpoint: {item}\ncode: {other}"
                for (path, item, other) in mismatches
            ]
        )
        raise AssertionError(f"Not a subset:\n{mm_str}")


def main():
    parser = ArgumentParser(description="Prepares the given checkpoint for submission")
    parser.add_argument("output")
    parser.add_argument("checkpoint", nargs="*")
    parser.add_argument("--model-key", default="model_ema")
    args = parser.parse_args()

    model_key = args.model_key

    if Path(args.output).exists():
        print(f"Output file {args.output} already exists")
        exit(1)

    model_weights = []
    config = None
    data_transforms = None
    for model_path in args.checkpoint:
        checkpoint = torch.load(model_path, map_location="cpu")
        model_weights.append(checkpoint[model_key])

        if config is None:
            config = checkpoint["config"]
        else:
            assert_check_subset(
                config,
                checkpoint["config"],
                ignore_paths=["config._current_fold", "config.created"],
            )

        if data_transforms is None:
            data_transforms = checkpoint["data_transforms"]
        elif data_transforms != checkpoint["data_transforms"]:
            raise AssertionError(
                f"Data Transforms differ:\n{data_transforms}\nvs:\n{checkpoint['data_transforms']}"
            )

    del config["dataconfigs"]

    torch.save(
        {
            "config": config,
            "data_transforms": data_transforms,
            model_key: model_weights,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
