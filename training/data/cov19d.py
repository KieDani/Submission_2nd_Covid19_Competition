import math
import pandas as pd
import os
from paths import DATA_PATH

def create_reference_file():
    path = DATA_PATH
    save_dict = []
    for trva, trva_path in [('train', 'train_cov19d'), ('val', 'validation_cov19d')]:
        severity_reference_path = os.path.join(path, trva_path, trva + '_partition_covid_categories.csv')
        severity_reference_df = pd.read_csv(severity_reference_path, delimiter=';')
        for inf in ['covid', 'non-covid']:
            inf_path = os.path.join(path, trva_path, inf)
            for scan in sorted(os.listdir(inf_path)):
                if len(scan.split('.')) > 1: continue
                savepath = os.path.join(trva_path, inf, scan)
                sev_label = -1
                if inf == 'covid':
                    scan_key = 'ct_scan_' + scan.strip('ct_scan').strip('_')
                    try:
                        sev_label = int(severity_reference_df[severity_reference_df['Name'] == scan_key]['Category'])
                    except:
                        print('No severity label for', scan_key)
                inf_label = 1 if inf == 'non-covid' else 2
                save_dict.append({'Path': savepath, 'Set': trva, 'Inf': inf_label, 'Sev': sev_label})
    df = pd.DataFrame.from_dict(save_dict)

    df.to_csv(os.path.join(path, 'my_reference.csv'))

if __name__ == "__main__":
    create_reference_file()

