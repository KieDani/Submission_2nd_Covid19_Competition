from typing import Sequence


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
