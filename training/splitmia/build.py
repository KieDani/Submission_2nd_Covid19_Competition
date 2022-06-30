from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from paths import DATA_PATH
import os


def assign_folds(df, num_folds, random_state=None, group_by=None):
    fold_size = len(df) // num_folds
    orig_len = len(df)
    folds = []
    for fold_id in range(num_folds - 1):
        fold_frac = fold_size / (orig_len - fold_id * fold_size)
        if group_by is None or len(group_by) == 0:
            fold = df.sample(frac=fold_frac, random_state=random_state)
        else:
            fold = df.groupby(group_by).sample(
                frac=fold_frac, random_state=random_state
            )
        df = df.drop(index=fold.index)
        fold["cv"] = fold_id
        folds.append(fold)
    df["cv"] = num_folds - 1
    folds.append(df)
    return pd.concat(folds).sort_index()


def main(reference: str, split_dir, num_folds=5):
    split_dir = Path(split_dir)
    ref = pd.read_csv(reference).drop(columns=["Unnamed: 0"])
    ref["patient"] = ref["Path"].str.replace("/", "-")
    ref = ref.set_index("patient")
    train = ref[ref.Set == "train"]

    inf_only = train[train["Sev"] != -1]

    cv5 = assign_folds(
        inf_only, num_folds=num_folds, random_state=1055, group_by=["Inf", "Sev"]
    )
    cv5.to_csv(split_dir / "cv5_infonly.csv", index=True)

    # use official split but only Sev != -1
    fulltrain = inf_only.copy()
    fulltrain["cv"] = 1
    fullval = ref[ref.Set == "val"]
    fullval = fullval[fullval["Sev"] != -1].copy()
    fullval["cv"] = 0
    full = pd.concat([fulltrain, fullval])
    full.to_csv(split_dir / "eccv.csv", index=True)

    # split exactly like the organizers intended it (and therefore no additional test set is left)
    official = ref.copy()
    official["cv"] = [0 if s == "val" else 1 for s in official["Set"]]
    # we have skipped some files because they are faulty
    # -> don't inlcude them in the split
    # skip = [
    #     "train_cov19d/covid/ct_scan_31",
    #     "train_cov19d/covid/ct_scan_47",
    #     "train_cov19d/non-covid/ct_scan899",
    #     "train_cov19d/non-covid/ct_scan988",
    #     "train_cov19d/non-covid/ct_scan_292",
    #     "train_cov19d/non-covid/ct_scan_354",
    #     "train_cov19d/non-covid/ct_scan_537",
    #     "train_cov19d/non-covid/ct_scan_853",
    #     "validation_cov19d/covid/ct_scan_101",
    #     "validation_cov19d/covid/ct_scan_18",
    #     "validation_cov19d/covid/ct_scan_40",
    #     "validation_cov19d/covid/ct_scan_46",
    #     "validation_cov19d/covid/ct_scan_48",
    #     "validation_cov19d/non-covid/ct_scan221",
    #     "validation_cov19d/non-covid/ct_scan250",
    #     "validation_cov19d/non-covid/ct_scan_15",
    # ]
    skip = []
    len_before = len(official)
    official = official[~official["Path"].isin(skip)]
    len_after = len(official)
    assert len(skip) == len_before - len_after
    official.to_csv(split_dir / "official.csv", index=True)


    # make an split for inf/no inf prediction
    infnoinf_cv5 = assign_folds(train, num_folds=num_folds, random_state=1055, group_by=["Inf"])
    infnoinf_cv5.to_csv(split_dir/"cv5_cov_nocov.csv", index=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reference", default=os.path.join(DATA_PATH, 'my_reference.csv'))
    parser.add_argument("--splitdir", default=".")
    args = parser.parse_args()
    main(args.reference, args.splitdir)
