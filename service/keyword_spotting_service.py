import librosa
import numpy as np
import tensorflow.keras as keras

from constants import SAVED_MODEL_PATH, SAMPLES_TO_CONSIDER, N_MFCC, N_FFT, \
    HOP_LENGTH


class KeywordSpottingService:
    # trained TF model
    model = None
    _mappings = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ]
    _instance = None  # python doesn't enforce singleton, unlike java

    def __init__(self):
        """ Constructor.
        """
        if KeywordSpottingService._instance is None:
            KeywordSpottingService._instance = self
            KeywordSpottingService.model = keras.models.load_model(SAVED_MODEL_PATH)
        else:
            raise Exception("You cannot create another KeywordSpottingService class")

    @staticmethod
    def get_instance():
        """ Static method to fetch the current instance.
        """
        if not KeywordSpottingService._instance:
            KeywordSpottingService()
        return KeywordSpottingService._instance

    def predict(self, file_path):
        # extract MFCCs (input to NN) from the file_path
        MFCCs = self.preprocess(file_path)  # (no. of segments, no. of coefficients)

        # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(
            MFCCs)  # [ [the probability of the input being each of the ten choices] ] because one sample
        predicted_index = int(np.argmax(predictions))

        # map that index to the keyword
        return self._mappings[predicted_index]

    @staticmethod
    def preprocess(file_path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

        mfccs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return mfccs.T


if __name__ == "__main__":
    kss = KeywordSpottingService.get_instance()
    # todo change \ to  / if deploying to a linux container
    keyword1 = kss.predict("resources\\testdata\\down.wav")
    print(f"Predicted keyword: {keyword1}")
