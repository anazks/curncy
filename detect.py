import cv2
import numpy as np
import time
import imutils
import pyttsx3
import threading
import tensorflow as tf

# -------------------- VOICE ENGINE --------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -------------------- LOAD YOLO --------------------
net = cv2.dnn.readNet(
    "yolov4-tiny-custom_final.weights",
    "yolov4-tiny-custom.cfg"
)

# -------------------- LOAD REAL/FAKE MODEL --------------------
rf_model = tf.keras.models.load_model("RealFake_Model.h5")

classes = []
with open("classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(100, 255, size=(len(classes), 3))

# -------------------- VARIABLES --------------------
prev_label = ""
stable_count = 0
required_stable_frames = 15   # must detect for 15 frames continuously
last_spoken_time = 0
cooldown = 3  # seconds

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0

# -------------------- UI PANEL --------------------
def draw_ui(frame, fps):

    overlay = frame.copy()

    # Top bar
    cv2.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], 70),
        (20, 20, 20),
        -1
    )

    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Title
    cv2.putText(
        frame,
        "Fake Currency Detector",
        (20, 45),
        font,
        1,
        (0, 255, 255),
        2
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {round(fps, 1)}",
        (frame.shape[1] - 150, 45),
        font,
        0.7,
        (0, 255, 0),
        2
    )

    return frame

# -------------------- MAIN LOOP --------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=900)
    frame_id += 1

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame,
        0.00392,
        (416, 416),
        (0, 0, 0),
        True,
        crop=False
    )

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # -------------------- DETECTION --------------------
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Increased confidence threshold
            if confidence > 0.95:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

    detected_label = ""
    best_roi = None

    # -------------------- DRAW BOX --------------------
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]

            detected_label = label

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )
            
            # Extract ROI for Real/Fake Classification
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(width, x + w)
            y_end = min(height, y + h)
            if y_end > y_start and x_end > x_start:
                best_roi = frame[y_start:y_end, x_start:x_end]

            cv2.putText(
                frame,
                f"{label} ({round(confidences[i], 2)})",
                (x, y - 10),
                font,
                0.7,
                color,
                2
            )

    # -------------------- STABLE VOICE LOGIC --------------------
    current_time = time.time()

    if detected_label != "":

        if detected_label == prev_label:
            stable_count += 1
        else:
            stable_count = 0

        # Speak only if stable for required frames AND cooldown passed
        if stable_count >= required_stable_frames and (current_time - last_spoken_time) > cooldown:
            try:
                value = int(detected_label)
                
                status_text = "real"
                if best_roi is not None:
                    # Preprocess for the custom model
                    roi_resized = cv2.resize(best_roi, (224, 224))
                    roi_normalized = roi_resized / 255.0
                    roi_expanded = np.expand_dims(roi_normalized, axis=0)
                    
                    # Output shape is (None, 1) where > 0.5 is Real
                    prediction = rf_model.predict(roi_expanded, verbose=0)[0][0]
                    if prediction < 0.5:
                        status_text = "fake"
                    else:
                        status_text = "real"

                threading.Thread(
                    target=speak,
                    args=(f"{value} rupees detected and it is {status_text}",)
                ).start()

                last_spoken_time = current_time
                stable_count = 0

            except Exception as e:
                print("Error in voice/classification logic:", e)
                pass
    else:
        stable_count = 0

    prev_label = detected_label

    # -------------------- FPS --------------------
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time

    # -------------------- DRAW UI --------------------
    frame = draw_ui(frame, fps)

    cv2.imshow("Currency Detector", frame)

    if cv2.waitKey(1) == 27:
        break

# -------------------- RELEASE --------------------
cap.release()
cv2.destroyAllWindows()