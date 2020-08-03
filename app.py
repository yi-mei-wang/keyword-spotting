import os
from datetime import datetime

from flask import Flask, jsonify, request

from service.keyword_spotting_service import KeywordSpottingService


DATETIME_FORMAT = "%Y%m%d%H%M%S"

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # get audio file and save it
    audio_file = request.files["file"]
    file_name = datetime.now().strftime(DATETIME_FORMAT)
    audio_file.save(file_name)

    # invoke keyword spotting service
    kss = KeywordSpottingService.get_instance()

    # make a prediction
    predicted_keyword = kss.predict(file_name)

    # remove audio file
    os.remove(file_name)

    data = {"keyword": predicted_keyword}
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
