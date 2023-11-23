import numpy as np
from flask import Flask, request
from PIL import Image
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/tumor-brain", methods=['POST'])
def tumor_brain_classifier():

    # ambil gambar yang dikirim pas request
    image_request = request.files['image']

    # konversi gambar jadi array
    image_pil = Image.open(image_request).convert('RGB')

    # resize gambarnya
    expected_size = (150, 150)
    resized_image_pil = image_pil.resize(expected_size)

    image_array = np.array(resized_image_pil)
    rescale_image_array = image_array/255.
    batched_rescaled_image_array = np.array([rescale_image_array])

    # load model
    load_model = tf.keras.models.load_model('tumor-brain.h5')
    print(load_model.get_config())

    result = load_model.predict(batched_rescaled_image_array)
    return get_formated_predict_result(result)


def get_formated_predict_result(predict_result):
    class_indexes = {
        0 : "Meningioma",
        1 : "Glioma",
        2 : "Pituitari"
    }

    process_predict_result = predict_result[0]
    maxIndex = 0
    maxValues = 0

    for index in range(len(process_predict_result)):
        if process_predict_result[index] > maxValues:
            maxValues = process_predict_result[index]
            maxIndex = index
    return class_indexes[maxIndex]

if __name__ == "__main__":
    app.run()
