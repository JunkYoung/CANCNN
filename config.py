class Config:    
    NAME = "DoS"
    #NAME = "Fuzzy"
    #NAME = "gear"
    #NAME = "RPM"

    NUM_ID = 2048
    UNIT_INTVL = 100/1000
    NUM_INTVL = 5

    FILENAME = f"dataset/{NAME}_dataset.csv"
    DATAPATH = f"data/unit{int(UNIT_INTVL*1000)}_num{NUM_INTVL}/{NAME}/"
    MODELNAME = f"models/{NAME}unit{int(UNIT_INTVL*1000)}_num{NUM_INTVL}.h5"
