DATA_JSON_PATH = "data.json"
DATASET_PATH = "dataset"
SAVED_MODEL_PATH = "model.h5"
# CNN expects that training and testing data to be the same shape
# also the default settings that librosa uses to load an audio file
SAMPLES_TO_CONSIDER = 22050  # 1 second worth of sound - same as when the model was trained
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

