import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("keras_model.h5", compile=False)

# Load labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Start webcam
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    # Resize image to model size
    image = cv2.resize(frame, (224, 224))

    # Normalize
    image = np.asarray(image, dtype=np.float32)
    image = (image / 127.5) - 1

    data = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = prediction[0][index]

    # Display prediction
    cv2.putText(
        frame,
        f"{class_name} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Emotion Detection", frame)

    # Press ESC to quit
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()