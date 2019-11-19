from raisr.raisr import RAISR

if __name__ == "__main__":
    raisr = RAISR()
    raisr.preprocess("/home/christophe/Downloads/source/data/DIV2K_train_LR_bicubic_X2/",
                     "/home/christophe/Downloads/source/data/DIV2K_train_HR/")
