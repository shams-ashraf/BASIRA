import cv2
import threading
from queue import Queue
from ultralytics import YOLO
from PIL import Image
from distance_model import estimate_distance_object
from traffic_light import detect_traffic_lights
from emotions import get_emotion
from ocr_model import extract_text_with_boxes

yolo_model = YOLO("yolov8n.pt")

class Dummy:
    def process_scene(self, pil_frame):
        return "street scene"

scene_caption = Dummy()

class Alerts:
    def __init__(self, log_file="model_outputs.txt", summary_file="summary_outputs.txt"):
        self.log_file = log_file
        self.summary_file = summary_file
        open(self.log_file, "w").close()
        open(self.summary_file, "w").close()
        self.summary_data = {"YOLO": set(),"Emotion": set(),"TrafficLight": set(),"Text": set(),"Scene": set()}

    def display_alerts(self, objects, scene):
        output_lines = []
        output_lines.append("\n===== MODEL OUTPUTS =====")
        output_lines.append("\n[YOLO detections]")
        for obj in objects:
            label = obj["label"]; dist = obj.get("distance", "")
            output_lines.append(f"- {label} at {dist}")
            self.summary_data["YOLO"].add(f"{label} at {dist}")
        output_lines.append("\n[Emotion recognition]")
        for obj in objects:
            if obj["label"] == "person":
                emo = obj.get("emotion", "UNKNOWN")
                output_lines.append(f"- Person emotion: {emo}")
                self.summary_data["Emotion"].add(emo)
        output_lines.append("\n[Traffic light detection]")
        for obj in objects:
            if obj["label"] == "traffic light":
                color = obj.get("color", "UNKNOWN")
                output_lines.append(f"- Traffic light: {color}")
                self.summary_data["TrafficLight"].add(color)
        output_lines.append("\n[Text / Sign recognition]")
        for obj in objects:
            if "text" in obj:
                output_lines.append(f"- Text detected: {obj['text']}")
                self.summary_data["Text"].add(obj['text'])
        output_lines.append("\n[Scene caption]")
        output_lines.append(f"- Scene: {scene}")
        self.summary_data["Scene"].add(scene)
        output_lines.append("=========================\n")
        print("\n".join(output_lines))
        with open(self.log_file, "a", encoding="utf-8") as f: f.write("\n".join(output_lines) + "\n")
        self.save_summary()

    def save_summary(self):
        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write("===== SUMMARY OUTPUTS (Aggregated) =====\n\n")
            f.write("[YOLO detections]\n"); [f.write(f"- {i}\n") for i in self.summary_data["YOLO"]]
            f.write("\n[Emotions]\n"); [f.write(f"- {i}\n") for i in self.summary_data["Emotion"]]
            f.write("\n[Traffic Lights]\n"); [f.write(f"- {i}\n") for i in self.summary_data["TrafficLight"]]
            f.write("\n[Texts / Signs]\n"); [f.write(f"- {i}\n") for i in self.summary_data["Text"]]
            f.write("\n[Scenes]\n"); [f.write(f"- {i}\n") for i in self.summary_data["Scene"]]

alerts = Alerts()
frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)

def camera_thread():
    cap = cv2.VideoCapture("http://192.168.100.4:4747/video")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 15)
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except: pass
        frame_queue.put(frame)
    cap.release()

def yolo_thread():
    frame_count = 0
    while True:
        frame = frame_queue.get()
        frame_resized = cv2.resize(frame, (640, 480))
        results = yolo_model.predict(frame_resized, imgsz=320, verbose=False, stream=True)
        objects_info = []
        frame_count += 1
        if frame_count % 10 == 0:
            ocr_results = extract_text_with_boxes(frame_resized)
            for text, (x, y, w, h) in ocr_results:
                objects_info.append({"label": "text", "text": text})
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame_resized, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                label = yolo_model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                obj_data = {"label": label, "box": [x1, y1, x2, y2]}
                obj_data["distance"] = estimate_distance_object((x1, y1, x2, y2), label)
                if label == "person":
                    face_crop = frame_resized[y1:y2, x1:x2]
                    obj_data["emotion"] = get_emotion(face_crop)
                elif label == "traffic light":
                    obj_data["color"] = detect_traffic_lights(frame_resized, (x1, y1, x2, y2))
                objects_info.append(obj_data)
                display_text = label
                if "distance" in obj_data: display_text += f" {obj_data['distance']}"
                if "emotion" in obj_data: display_text += f" {obj_data['emotion']}"
                if "color" in obj_data: display_text += f" {obj_data['color']}"
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        scene = ""
        if frame_count % 15 == 0:
            scene = scene_caption.process_scene(Image.fromarray(frame_resized))
        if result_queue.full():
            try: result_queue.get_nowait()
            except: pass
        result_queue.put((frame_resized, objects_info, scene))

threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=yolo_thread, daemon=True).start()

while True:
    if not result_queue.empty():
        frame, objects, scene = result_queue.get()
        alerts.display_alerts(objects, scene)
        cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cv2.destroyAllWindows()
