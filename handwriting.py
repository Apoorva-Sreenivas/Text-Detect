import os
import re
import shutil
import cv2
import pickle
import numpy as np  
import tensorflow as tf
from tensorflow import keras
# from PIL import Image as im
from word_detector import detect, prepare_img, sort_multiline  # Assuming these are custom functions
from path import Path
# from typing import List

class Handwriting_Recognition_System:
    class CTCLayer(keras.layers.Layer):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.loss_fn = keras.backend.ctc_batch_cost

        def call(self, y_true, y_pred):
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            loss = self.loss_fn(y_true, y_pred, input_length, label_length)
            self.add_loss(loss)

            return y_pred

    def __init__(self, detection_args=None, recognition_model_path="ocr_model_50_epoch.h5", characters_path="./characters"):
        self.detection_args = detection_args if detection_args is not None else {
            'data': Path('./r06-137.png'),
            'kernel_size': 25,
            'sigma': 11,
            'theta': 7,
            'min_area': 100,
            'img_height': 1000
        }
        self.recognition_model_path = recognition_model_path
        self.characters_path = characters_path
        self.list_img_names_serial = []
        self.max_len = 21  # Adjust as necessary

        # Initialize model and other resources
        self._initialize()

    def _initialize(self):
        # Initialize or load necessary resources
        self._load_characters()
        self._load_recognition_model()

    def _load_characters(self):
        with open(self.characters_path, "rb") as fp:
            self.characters = pickle.load(fp)

        self.char_to_num = keras.layers.StringLookup(vocabulary=self.characters, mask_token=None)
        self.num_to_chars = keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)

    def _load_recognition_model(self):
        custom_objects = {"CTCLayer": Handwriting_Recognition_System.CTCLayer}  # Reference CTCLayer from OCRSystem
        self.reconstructed_model = keras.models.load_model(self.recognition_model_path, custom_objects=custom_objects)
        self.prediction_model = keras.models.Model(
            self.reconstructed_model.get_layer(name="image").input,
            self.reconstructed_model.get_layer(name="dense2").output
        )

    def save_image_names_to_text_files(self,image_path):
        parsed = self.detection_args
        # for fn_img in self.get_img_files(image_path):s
        # for fn_img in image_path:
        if image_path:
            img = prepare_img(cv2.imread(image_path), parsed['img_height'])
            detections = detect(img, kernel_size=parsed['kernel_size'], sigma=parsed['sigma'], theta=parsed['theta'], min_area=parsed['min_area'])
            lines = sort_multiline(detections)

            for line_idx, line in enumerate(lines):
                for word_idx, det in enumerate(line):
                    crop_img = img[det.bbox.y:det.bbox.y + det.bbox.h, det.bbox.x:det.bbox.x + det.bbox.w]
                    path = './test_images'
                    if not os.path.exists(path):
                        os.mkdir(path)

                    cv2.imwrite(f"{path}/line{line_idx}_word{word_idx}.jpg", crop_img)
                    full_img_path = f"line{line_idx}_word{word_idx}.jpg"
                    self.list_img_names_serial.append(full_img_path)

                    with open("./examples/img_names_sequence.txt", "w") as textfile:
                        for element in self.list_img_names_serial:
                            textfile.write(element + "\n")

    def prepare_test_images(self, base_image_path):
        t_images = [os.path.join(base_image_path, f) for f in os.listdir(base_image_path)]
        t_images.sort(key=self.natural_keys)
        return t_images

    def natural_keys(self, text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def preprocess_image(self, image_path, img_size=(128, 32)):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = self.distortion_free_resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def distortion_free_resize(self, image, img_size):
        w, h = img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        pad_height_top = pad_height // 2 + pad_height % 2
        pad_height_bottom = pad_height // 2
        pad_width_left = pad_width // 2 + pad_width % 2
        pad_width_right = pad_width // 2

        image = tf.pad(image, paddings=[[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        return image

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.max_len]

        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(self.num_to_chars(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def recognize_text_from_images(self, base_image_path):
        t_images = self.prepare_test_images(base_image_path)
        inf_images = self.prepare_test_images(base_image_path)  # Changed to base_image_path

        pred_test_text = []
        for image_path in inf_images:
            image = self.preprocess_image(image_path)
            image = tf.expand_dims(image, axis=0)
            preds = self.prediction_model.predict(image)
            pred_texts = self.decode_batch_predictions(preds)
            pred_test_text.append(pred_texts)

        flat_list = [item for sublist in pred_test_text for item in sublist]
        sentence = ' '.join(flat_list)
        return sentence

    def clear_test_images(self, directory="./test_images"):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def main(image_path):
    ocr_system = Handwriting_Recognition_System()
    
    ocr_system.save_image_names_to_text_files(image_path)

    # Recognize text from saved images
    base_image_path = "./test_images"
    recognized_sentence = ocr_system.recognize_text_from_images(base_image_path)
    # print(recognized_sentence)
    ocr_system.clear_test_images()
    return recognized_sentence

# main("C:/Users/Apoorva/OneDrive/Documents/Text Recognition/hand2.jpg")