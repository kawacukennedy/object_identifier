import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
import pyttsx3
import time
import threading

# COCO labels (first is background)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# List of known object names to match
KNOWN_OBJECTS = [
    'person', 'bottle', 'cup', 'cell phone', 'laptop', 'book', 'chair', 'dog', 'cat', 'car'
]

def speak(text):
    engine = pyttsx3.init()
    # Set male voice
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'male' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

def get_model():
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model.eval()
    return model

transform = T.Compose([
    T.ToTensor(),
])

def detection_worker(model, transform, COCO_INSTANCE_CATEGORY_NAMES, KNOWN_OBJECTS, frame_getter, result_setter, stop_event):
    last_announced = None
    while not stop_event.is_set():
        frame = frame_getter()
        if frame is None:
            time.sleep(0.01)
            continue
        # Resize frame for detection
        small_frame = cv2.resize(frame, (320, 240))
        img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img)
        with torch.no_grad():
            prediction = model([img_tensor])[0]
        detection_result = []
        detected = set()
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score > 0.7:
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
                if label_name in KNOWN_OBJECTS:
                    detected.add(label_name)
                # Scale box coordinates back to original frame size
                x1, y1, x2, y2 = box.int().tolist()
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
                detection_result.append((x1, y1, x2, y2, label_name, score))
        # Announce the first detected known object if it's new
        if detected:
            obj = sorted(detected)[0]
            if obj != last_announced:
                speak(obj)
                last_announced = obj
        result_setter(detection_result)
        time.sleep(0.1)  # Run detection every 0.1s (adjust as needed)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    model = get_model()
    latest_frame = [None]
    detection_result = [[]]
    stop_event = threading.Event()
    def frame_getter():
        return latest_frame[0]
    def result_setter(res):
        detection_result[0] = res
    # Start detection thread
    det_thread = threading.Thread(target=detection_worker, args=(model, transform, COCO_INSTANCE_CATEGORY_NAMES, KNOWN_OBJECTS, frame_getter, result_setter, stop_event))
    det_thread.start()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            latest_frame[0] = frame.copy()
            # Draw last detection results on the current frame
            for x1, y1, x2, y2, label_name, score in detection_result[0]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            time.sleep(0.01)  # Small sleep to reduce CPU usage
    finally:
        stop_event.set()
        det_thread.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 