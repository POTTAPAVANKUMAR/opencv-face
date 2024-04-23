from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__, template_folder='templates')  

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    images = []
    labels = []

    for i in range(1, 6):
        image = request.files.get(f'image{i}')
        label = request.form.get(f'label{i}')

        if image:
            image_bytes = image.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(img_np)
            labels.append(label)

    train_model(images, labels)
    return jsonify({"message": "Model trained successfully"})


@app.route('/test', methods=['POST'])
def test():
    file = request.files['file']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    predicted_name = test_face(image)
    return jsonify({"predicted_name": predicted_name})

def train_model(images, labels):
    face_images = []
    face_labels = []

    for image, label in zip(images, labels):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            face_images.append(roi_gray)
            face_labels.append(label)

    face_labels = np.array(face_labels)
    recognizer.train(face_images, face_labels)
    recognizer.save("trained_model.yml")

def test_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)
        if confidence < 100:
            predicted_name = str(label)
        else:
            predicted_name = "Unknown"

    return predicted_name

if __name__ == '__main__':
    app.run(debug=True)
