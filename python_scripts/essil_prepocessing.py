'''
essil_preprocessing.py
Author: Nicholas Hoernle
Date: September 2018
'''

import datetime
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import re

BIOMES = ['Desert_Water', 'Plains_Water', 'Jungle_Water', 'Wetlands_Water']
BIOME_NAMES = ['Desert', 'Plains', 'Jungle', 'Wetlands']
PLANTS = ['Desert_Plants', 'Plains_Plants', 'Jungle_Plants', 'Wetlands_Plants']
# PLANT_LEVELS = ['lv1', 'lv2', 'lv3', 'lv4']
PLANT_LEVELS = ['lv3', 'lv4']

PLANT_MAPPING = {
        4: 3, # Shade Plant
		5: 3, # Shifty Cactus
		6: 2, # Early Growth
		8: 2, # Early Growth
		9: 1, # Early Growth
		11: 1, # Early Growth
		12: 4, # Shimmer Tree
		13: 3, # Fruit Spitter
		17: 1, # Early Growth
		18: 2, # Early Growth
		20: 3, # Spinner Beacon
		21: 3, # Rattle Beacon
		22: 3, # Rattle Beacon
		23: 4, # Shapeshifter
		24: 2, # Early Growth
		25: 3, # Wiggle Tree
		26: 3, # Bloom Tree
		27: 3, # Fan Bush
		30: 1, # Early Growth
		31: 1, # Early Growth
		32: 2, # Early Growth
		33: 2, # Early Growth
		34: 4, # Canopy Tree
		37: 2, # Early Growth
		38: 2, # Early Growth
		39: 1, # Early Growth
		45: 3, # Pedestal Blossom Tree
		46: 2, # Grass Walker
		47: 3, # Weeping Wisp
		49: 1, # Early Growth
		50: 4, # Shower Blossom Tree
}

CREATURE_MAPPING = {
    1 : 4, #dreamcatcher
    4 : 3, #stoic bird
    5 : 1, #grazer
    6 : 4, #water deer
    10 : 3, #stalk jumper
    12 : 3, #flower bird
    13 : 3, #flower bird baby in nest
    14 : 3, #flower bird baby flying
    18 : 3, #antepony
    19 : 4, #lamplighter
    26 : 3, #mossback
    27 : 3, #casterbird
    28 : 3, #wongu
    30 : 2, #rockroller
    31 : 3, #stoic bird baby
}

def clean_column_names(df):
    df.columns = df.columns.str.strip()
    df = df.iloc[1:].reset_index()
    return df

def forward_fill_time(df):

    # correct for the seconds that are missing / remove duplicates.
    # forward fill the values in all instances where missing

    # convert to datetime object
    time_values = df['Timestamp'].apply(lambda x: datetime.datetime(*[int(v) for v in x.split('-')]))

    df['_video_seconds'] = df.index.values
    # correct the time object in the dataframe
    df['Timestamp'] = time_values
    # create a new dataframe
    df_new = pd.DataFrame({'Timestamp': pd.date_range(time_values.min(), time_values.max(), freq='S')})

    return df_new.merge(df, how='left').ffill()

def get_bin_water_values(df, biomes=['Desert', 'Plains', 'Jungle', 'Wetlands']):

    # get the bin water value and convert to the same units
    # as the global water values
    for biome in biomes:
        df[biome+'_BinsWater'] = (df[biome+'_WaterBins'].apply(lambda x: np.sum([float(v) for v in x.split('-')])) +
                                  df[biome+'_FloodBins'].apply(lambda x: np.sum([float(v) for v in x.split('-')])))/60
    return df

def get_clouds_water(df, biomes=['Desert', 'Plains', 'Jungle', 'Wetlands']):

    for biome in biomes:
        df[biome+'_CloudsWater'] = df[biome+'_Water'] - df[biome+'_BinsWater']
        df[biome+'_Water'] = df[biome+'_BinsWater']

    df['_CloudWater'] = df[[(b+'_CloudsWater') for b in biomes]].sum(axis=1)
    return df

def filter_to_the_useful_columns(df,
                                 biomes=["", 'Desert', 'Plains', 'Waterfall', 'Floor', 'Jungle', 'Reservoir', 'Wetlands', 'MountainValley'],
                                 useful=['Water', 'CloudWater', 'video_seconds', 'Plants', 'lv1', 'lv2', 'lv3', 'lv4']):

    known_columns = df.columns

    # get the most useful columns out
    columns = []
    for biome in biomes:
        for column in useful:
            suggestion = biome + '_' + column
            if suggestion in known_columns:
                columns.append(suggestion)

    columns += [f'Plants_level{i}' for i in range(1,5)]
    columns += [f'Creatures_level{i}' for i in range(1,5)]
    columns += [f'{biome}_Raining' for biome in BIOME_NAMES]
    columns += [f'{biome}_Clouds' for biome in BIOME_NAMES]

    return df[columns]

def get_plant_data(df):

    for biome in BIOMES:

        df_ = (df[[biome.replace('_Water', '_')+p for p in PLANT_LEVELS]]).sum(axis=1)
        df[biome.replace('_Water', '_') + 'Plants'] = df_
    return df

def get_filtered_first_order(df, biomes, filter_window = 4//2):

    water = df[biomes].values
    water_ = np.zeros_like(water)
    xs = np.arange(0, water.shape[0])

    for i, water_signal in enumerate(water.T):
        # for j, item in enumerate(water_signal):
        #
        #     min_ix = np.max([0,j-filter_window])
        #     max_ix = np.min([len(water_signal)-1, j+filter_window+1])
        #
        #     Y = water_signal[min_ix:max_ix][:,None]
        #     X = xs[min_ix:max_ix,None]
        #     X = np.concatenate([np.ones_like(X), X], axis=1)
        #
        #     beta0, beta1 = np.linalg.pinv((X.T.dot(X))).dot(X.T).dot(Y)[:,0]
        #     water_[j,i] = beta1
        water_[1:,i] = np.diff(np.round(water_signal, decimals=3))
        water_[1:,i] = np.clip(water_[1:,i], np.percentile(water_[1:,i], 2.5), np.percentile(water_[1:,i],97.5))

    # # scale to be 90%
    water_ = water_/np.percentile(water_, 90)
    return water_

#-------------------------------------------------------------------------------
# Some post-processing steps to make the result a little more interpretable.

def max_convolution(arr, window=2, dtype=np.int16):

    x = np.zeros_like(arr, dtype=dtype)
    for i,_ in enumerate(x):
        p = np.max([0, i-window])
        e = np.min([len(arr), i+window+1])
        x[i] = np.max(arr[p:e])
    return x

def mode_convolution(arr, window=5, dtype=np.int16):

    x = np.zeros_like(arr, dtype=dtype)
    for i,_ in enumerate(x):
        p = np.max([0, i-window])
        e = np.min([len(arr), i+window+1])
        x[i] = stats.mode(arr[p:e])[0][0]
    return x

def group_consecutives(vals, step=0):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def mode_convolution2(arr, theta, window=5, dtype=np.int16):

    consecutives = group_consecutives(arr)
    result = []

    for i, group in enumerate(consecutives):
        if i <= 1:
            result.append(group)
            continue
        if i >= len(consecutives)-1:
            result.append(group)
            continue

        if len(group) < window:
            options = [consecutives[i-1], consecutives[i], consecutives[i+1]]
            if np.argmax([[len(o) for o in options]]) == 1:
                result.append(group)
                continue

            best_option = np.argmin([np.sum(np.abs(theta[options[0][0]]['A'] - theta[options[1][0]]['A'])), np.sum(np.abs(theta[options[2][0]]['A'] - theta[options[1][0]]['A']))])

            result.append([options[best_option*2][0] for j in range(len(group))])
        else:
            result.append(group)
    return np.array([r for sub in result for r in sub])

def get_number_animals_and_plants(row, biome_name, level=-1, mapping=PLANT_MAPPING):

    text = row[biome_name].strip()
    regex_match = r"T(\d+)Q(\d+)"
    alive_counts = np.zeros(4)
    for type_ in text.split('-'):
        if len(type_) == 0:
            continue

        groups = re.match(regex_match, type_)
        try:
            type_animal = mapping[int(groups.group(1))]
            num_alv = int(groups.group(2))
            alive_counts[type_animal-1] += num_alv
        except:
            continue

    if level != -1:
        return alive_counts[level]
    return alive_counts

def find_and_store_animals_and_plants(df):
    df['Plants_level1'] = np.sum([df.apply(get_number_animals_and_plants, args=(f'{b}_PlantIndex',0,), axis=1) for b in BIOME_NAMES], axis=0).astype(int)
    df['Plants_level2'] = np.sum([df.apply(get_number_animals_and_plants, args=(f'{b}_PlantIndex',1,), axis=1) for b in BIOME_NAMES], axis=0).astype(int)
    df['Plants_level3'] = np.sum([df.apply(get_number_animals_and_plants, args=(f'{b}_PlantIndex',2,), axis=1) for b in BIOME_NAMES], axis=0).astype(int)
    df['Plants_level4'] = np.sum([df.apply(get_number_animals_and_plants, args=(f'{b}_PlantIndex',3,), axis=1) for b in BIOME_NAMES], axis=0).astype(int)

    df['Creatures_level1'] = np.sum([df.apply(get_number_animals_and_plants, args=(f'{b}_CreatureIndex',0,CREATURE_MAPPING), axis=1) for b in BIOME_NAMES], axis=0).astype(int)
    df['Creatures_level2'] = np.sum([df.apply(get_number_animals_and_plants, args=(f'{b}_CreatureIndex',1,CREATURE_MAPPING), axis=1) for b in BIOME_NAMES], axis=0).astype(int)
    df['Creatures_level3'] = np.sum([df.apply(get_number_animals_and_plants, args=(f'{b}_CreatureIndex',2,CREATURE_MAPPING), axis=1) for b in BIOME_NAMES], axis=0).astype(int)
    df['Creatures_level4'] = np.sum([df.apply(get_number_animals_and_plants, args=(f'{b}_CreatureIndex',3,CREATURE_MAPPING), axis=1) for b in BIOME_NAMES], axis=0).astype(int)

    return df

def load_essil_file(file_name):
    '''
    Preprocessing pipeline to produce a fully pre-processed file ready for inference
    '''
    df = pd.read_csv(file_name)
    df = clean_column_names(df)
    df = forward_fill_time(df)
    df = get_bin_water_values(df)
    df = get_clouds_water(df)
    df = get_plant_data(df)
    df = find_and_store_animals_and_plants(df)
    df = filter_to_the_useful_columns(df)
    df['Source_Water'] = df.Waterfall_Water + df.Floor_Water + df.Reservoir_Water + df.MountainValley_Water
    df = df.reset_index().rename(columns={'index': 'seconds'})
    return df

# ------------------------------------------------------------------------------
# Some plant processing steps that aren't really used:
def get_plants_diff_filtered(biome, data, level_bottom=1, level_top=4, increasing=True):
    Y = data[[f'{biome}_lv{l}' for l in range(level_bottom, level_top+1)]].diff().values

    Y[np.isnan(Y)] = 0
    Y = Y[:,1:].sum(axis=1)

    if increasing:
        Y[Y < 0] = 0
    else:
        Y[Y > 0] = 0
        Y = -Y

    Y = Y.astype(int)

    Y = max_convolution(Y, window=4)
    return Y

def plants_post_process(sampler):
    res = sampler.assignments[:,-1]
    res = max_convolution(res, 3)
    mean = np.array([sampler.get_sample_theta_params()[-1][val] for val in res])
    mean[mean >= 0.1] = 1
    return res, mean


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='''
    Read a raw ESSIL file from {input_file} and save the processed file to {output_file}.
    This processing step corrects the time from the ESSIL logs and stores the correct time in
    seconds as the row number. The time that correcponds to the video updates is stored under the
    '_video_seconds' column.
    ''')

    parser.add_argument('--input_file', '-I', metavar='input_file', type=str, nargs=1,
                    help='Input file to the CW raw logged data')
    parser.add_argument('--output_file', '-O', metavar='output_file', type=str, nargs=1,
                    help='Output file to store the processed data')

    args = parser.parse_args()

    # create necessary folders and prepare the input
    input_file = args.input_file[0]
    output_file = args.output_file[0]

    print('Running ESSIL Pre-Process')
    print('Reading from raw file: {input_file}')

    df = load_essil_file(input_file)
    df.to_csv(output_file)

    print('Saved processed file to: {output_file}')
    print('Done')
