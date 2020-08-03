import requests

HOST = "http://localhost:5000"
PREDICT_PATH = "/predict"
URL = f"{HOST}{PREDICT_PATH}"
# todo rename if deploying in linux
TEST_AUDIO_FILE_PATH = "resources\\testdata\\down.wav"


if __name__ == "__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")
