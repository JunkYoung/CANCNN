import os
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

def one_step(chunk, pre_time):
    count = np.zeros((Config.NUM_ID))
    sum_IAT = np.zeros((Config.NUM_ID))
    pre_time = np.zeros((Config.NUM_ID))
    for i in range(len(chunk)):
        idx = chunk['1'].iloc[i]
        count[idx] += 1
        if pre_time[idx] != 0:
            sum_IAT[idx] += chunk['0'].iloc[i] - pre_time[idx]
        pre_time[idx] = chunk['0'].iloc[i]
    
    return count, sum_IAT, pre_time

def make_dataset(filename):
    os.makedirs(Config.DATAPATH, exist_ok=True)
    df = pd.read_csv(filename, names=[str(x) for x in range(12)], header=None)
    df = df[['0', '1', '11']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['1'] = df['1'].apply(int, base=16)

    end, start = df['0'].max(), df['0'].min()
    num_data = int((end-start)/Config.UNIT_INTVL)
    pre_time = np.zeros(Config.NUM_ID)
    counts = np.zeros((Config.NUM_INTVL, Config.NUM_ID))
    sum_IATs = np.zeros((Config.NUM_INTVL, Config.NUM_ID))
    counts2 = np.zeros((Config.NUM_INTVL, Config.NUM_ID))
    sum_IATs2 = np.zeros((Config.NUM_INTVL, Config.NUM_ID))
    labels = []
    hist = []

    for i in range(num_data):
        if i%100 == 0:
            print(f"{i}/{num_data} {int(i/num_data*100)}%")
        
        frequencys = []
        mean_IATs = []

        cur = start+i*Config.UNIT_INTVL

        big_chunk = df[(df['0'] >= cur-Config.UNIT_INTVL2*Config.NUM_INTVL) & (df['0'] < cur)]
        labels.append(1) if 'T' in big_chunk.values else labels.append(0)
        
        cur_chunk = df[(df['0'] >= cur-Config.UNIT_INTVL) & (df['0'] < cur)]
        cur_count, cur_sum_IAT, pre_time = one_step(cur_chunk, pre_time)
        hist.append((cur_count, cur_sum_IAT))

        for j in range(Config.NUM_INTVL):
            idx = i-(j+1)
            if idx >= 0:
                pre_count, pre_sum_IAT = hist[idx]
            else:
                pre_count, pre_sum_IAT = np.zeros_like(cur_count), np.zeros_like(cur_sum_IAT)
            
            counts[j] = counts[j] + cur_count - pre_count
            frequency = counts[j]
            sum_IATs[j] = sum_IATs[j] + cur_sum_IAT - pre_sum_IAT
            mean_IAT = sum_IATs[j]/(frequency+0.000001)

            frequencys.append(frequency)
            mean_IATs.append(mean_IAT)

        for j in range(Config.NUM_INTVL):
            idx = i-int(j*(Config.UNIT_INTVL2/Config.UNIT_INTVL)+1)
            if idx >= 0:
                pre_count, pre_sum_IAT = hist[idx]
            else:
                pre_count, pre_sum_IAT = np.zeros_like(cur_count), np.zeros_like(cur_sum_IAT)
            
            counts2[j] = counts2[j] + cur_count - pre_count
            frequency = counts2[j]
            sum_IATs2[j] = sum_IATs2[j] + cur_sum_IAT - pre_sum_IAT
            mean_IAT = sum_IATs2[j]/(frequency+0.000001)

            frequencys.append(frequency)
            mean_IATs.append(mean_IAT)
        
        frequencys = np.array(frequencys).transpose()
        mean_IATs = np.array(mean_IATs).transpose()
        mean_IATs = np.array(pd.DataFrame(mean_IATs).replace([0, np.nan], 1))
        data = np.concatenate([frequencys, mean_IATs], -1)
        np.save(Config.DATAPATH+str(i), data.astype(np.float32))
    np.save(Config.DATAPATH+"labels", np.array(labels))
    

if __name__ == "__main__":
    #DoS_variation(Config.FILENAME)
    #Fuzzy_variation(Config.FILENAME)
    make_dataset("dataset/DoS_variation.csv")
    #make_dataset(Config.FILENAME)