class Config:    
    NAME = "DoS"
    #NAME = "Fuzzy"
    #NAME = "gear"
    #NAME = "RPM"

    NUM_ID = 2048
    UNIT_INTVL = 50/1000
    UNIT_INTVL2 = 1000/1000
    NUM_INTVL = 10
    
    FILENAME = f"dataset/{NAME}_dataset.csv"
    #DATAPATH = "data/test/Fuzzy/"
    DATAPATH = f"data/unit{int(UNIT_INTVL*1000)}_{int(UNIT_INTVL2*1000)}_num{NUM_INTVL}/{NAME}/"
    MODELNAME = f"models/{NAME}unit{int(UNIT_INTVL*1000)}_{int(UNIT_INTVL2*1000)}_num{NUM_INTVL}.h5"