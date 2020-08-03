import os
import json
import librosa

from constants import SAMPLES_TO_CONSIDER, DATASET_PATH, DATA_JSON_PATH, N_MFCC, \
    N_FFT, HOP_LENGTH


def prepare_dataset(dataset_path, json_path, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT):
    # data dictionary - store all the data extracted from the audio file
    data = {
        # map the keywords ('on', 'off', etc) to their index - should pass numbers, not words to NN
        "mappings": [],  # e.g. ["on", "off", ...]
        "labels": [],  # e.g. [0, 1, ...]
        "MFCCs": [],
        "files": []  # e.g. ["dataset/on/1.wav", ...]
    }

    # loop through all the sub-dirs
    for i, (dirpath, subdirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that not at root level (os.walk(root) starts at root)
        if dirpath is not dataset_path:
            # update mappings
            # todo \\ -> /
            category = dirpath.split("\\")[-1]  # dataset/down -> [dataset, down]
            print(f"Category is: {category}")
            data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through all the filenames and extract MFCCs
            for f in filenames:
                # reconstruct file path 
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 sec - ensure the data sample are of the same shape
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # enforce 1 sec long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract the MFCCs - np array
                    mfccs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length,
                                                 n_fft=n_fft)

                    # store data in data.json
                    data["labels"].append(i - 1) 
                    data["MFCCs"].append(mfccs.T.tolist())  # cast to python list from numpy array
                    data["files"].append(file_path)
                    print(f"{file_path}: {i - 1}")

    # store in a json file
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, DATA_JSON_PATH)
