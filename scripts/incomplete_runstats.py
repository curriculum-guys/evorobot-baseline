import pandas as pd
import numpy as np
import os

def process_dataframe(stages):
    df = pd.DataFrame(stages, columns=columns)
    df.msteps = [int(mstep) for mstep in df.msteps]
    df = df.sort_values('msteps')
    df['gen'] = [i for i in range(1, len(df)+1)]
    df.index = df.gen
    return df

columns = ['msteps','bestfit','bestgfit','bestsam','avgfit','paramsize']
data_dir = "../data/xdpole/runstats"
stg_dir = f"{data_dir}/stg"
stg_list = os.listdir(stg_dir)

seeds = {}
for i in range(1, 11):
    seed_name = f"s{i}"
    seeds[seed_name] = [np.arange(6)]

for seed in seeds.keys():
    for stg_file in stg_list:
        if stg_file.startswith(seed):
            stg = f"{stg_dir}/{stg_file}"
            data = np.load(stg, allow_pickle=True)
            seeds[seed] = np.append(seeds[seed], data, axis=0)
            stg_list.remove(stg_file)
    df = process_dataframe(seeds[seed][1:])
    if not df.empty:
        df.to_csv(f'{data_dir}/{seed}_run.csv', index=False)
        print(f"[{seed} done]")
