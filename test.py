import os
import shutil
import numpy as np
import tensorflow as tf
from config import Config
from preprocess import make_dataset
from model import CNN

from generator import DataGenerator


Config.NAME = "DoS"
#Config.NAME = "Fuzzy"
#Config.NAME = "gear"
#Config.NAME = "RPM"

Config.NUM_ID = 2048
Config.UNIT_INTVL = 1000/1000
Config.NUM_INTVL = 5

Config.FILENAME = f"dataset/{Config.NAME}_dataset.csv"
Config.DATAPATH = f"data/unit{int(Config.UNIT_INTVL*1000)}_num{Config.NUM_INTVL}/{Config.NAME}/"
Config.MODELNAME = f"models/{Config.NAME}unit{int(Config.UNIT_INTVL*1000)}_num{Config.NUM_INTVL}.h5"


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
  except RuntimeError as e:
    print(e)


def gen_data():
    params = {'dim': (Config.NUM_ID, 2*Config.NUM_INTVL),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}
    Config.DATAPATH = 'data/test/'
    make_dataset("DoS_variation.csv")
    data = os.listdir(Config.DATAPATH)
    data.remove('labels.npy')
    data = data[int(len(data)/10*8.5):]
    labels = np.load(Config.DATAPATH+"labels.npy")
    gen_test = DataGenerator(data, labels, **params)

    return gen_test


if __name__ == "__main__":
    make_dataset(Config.FILENAME)
    cnn = CNN(Config.MODELNAME)
    cnn.train()
    cnn.gen_test = gen_data()
    cnn.test()
