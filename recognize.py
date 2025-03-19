import cv2
import tensorflow as tf
import argparse
import numpy as np
import string
import os

IMG_SIZE = 64
labels = dict(enumerate(list(string.ascii_uppercase)))

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise Exception("Frame capture failed.")
    return frame

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = 300, 100, 200, 200  # Adjust ROI as needed
    roi = gray[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    thresh = cv2.adaptiveThreshold(roi_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh_norm = thresh.astype('float32') / 255.0
    thresh_norm = np.expand_dims(thresh_norm, axis=-1)
    input_img = np.expand_dims(thresh_norm, axis=0)
    return thresh, input_img, (x, y, w, h)

def output_frame(frame, thresh, prediction, roi_coords):
    x, y, w, h = roi_coords
    letter = labels.get(np.argmax(prediction), '?')
    cv2.putText(frame, f'Prediction: {letter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("ASL Recognition", frame)
    cv2.imshow("Processed ROI", thresh)

def recognize(model_path):
    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        return
    model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam.")
        return
    while True:
        frame = capture_frame(cap)
        thresh, input_img, roi_coords = process_frame(frame)
        prediction = model.predict(input_img)
        output_frame(frame, thresh, prediction, roi_coords)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="asl_model.h5")
    args = parser.parse_args()
    recognize(args.model_path)
