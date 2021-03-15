import os
import glob
import h5py
import numpy as np
import pandas as pd

from config import Config


def DoS_variation(filename):
    df = pd.read_csv(filename, names=[str(x) for x in range(12)], header=None)
    df['1'] = df.apply(lambda x: '00f0' if x['11'] == 'T' else x['1'], axis = 1)
    df.to_csv("dataset/DoS_variation.csv", header=None)

def Fuzzy_variation(filename):
    df = pd.read_csv(filename, names=[str(x) for x in range(12)], header=None)
    df['1'] = df.apply(lambda x: hex(~int(x['1'][2:], 16))[3:] if x['11'] == 'T' else x['1'], axis = 1)
    df.to_csv("dataset/Fuzzy_variation.csv", header=None)

def id2bit(id):
    b = np.zeros(29)
    b1 = np.right_shift(id, 8).astype(np.uint8)
    b[18:21] = np.unpackbits(b1)[-3:]
    b2 = np.array(id%256, dtype=np.uint8)
    b[21:29] = np.unpackbits(b2)

    return b

def make_dataset(filename):
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(filename, names=[str(x) for x in range(12)], header=None)
    df = df[['0', '1', '11']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    labels = []

    for i in range(int(len(df)/29)):
        if i%100 == 0:
            print(f"{i}/{int(len(df)/29)}")
        
        data = df.iloc[i*29:(i+1)*29]
        labels.append(1) if 'T' in data.values else labels.append(0)
        data = np.stack(data['1'].apply(lambda x : id2bit(x)).to_numpy())

        np.save(Config.DATAPATH+str(i), data)
    np.save(Config.DATAPATH+"labels", np.array(labels))
    

if __name__ == "__main__":
    #DoS_variation(Config.FILENAME)
    Fuzzy_variation(Config.FILENAME)
    make_dataset(f"dataset/{Config.NAME}_variation.csv")
    #make_dataset(Config.FILENAME)