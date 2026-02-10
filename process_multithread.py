from ultralytics import YOLO
import cv2
import numpy as np

import threading
from queue import Queue
from collections import defaultdict, deque
import time

import supervision as sv

from view_transform import ViewTransformer
from save_video import VideoWriterMP4



YOLO_MODEL = YOLO("models/pruned_yolov8n.engine", task="detect")

VIDEO_PATH = "data/vehicles.mp4"
OUTPUT_VIDEO = "outputs/vehicles_result.mp4"


video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)
thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

# TRACKER = Sort()
TRACKER = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=0.3)

BOX_ANNOTATOR = sv.BoxAnnotator(thickness=thickness)
LABEL_ANNOTATOR = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
TRACE_ANNOTATOR = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.CENTER,
    )

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

detection_zone = sv.PolygonZone(polygon=SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

START = sv.Point(0, 1500)
END = sv.Point(3840, 1500)

LINE_COUNTER = sv.LineZone(start=START, end=END)

LINE_COUNTER_ANNOTATOR = sv.LineZoneAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=2)



DISPLAY_SIZE = (1200, 600)
video_writer = VideoWriterMP4(
    output_path=OUTPUT_VIDEO,
    fps=video_info.fps,
    frame_size=DISPLAY_SIZE
)

cap = cv2.VideoCapture(VIDEO_PATH)

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    




start_time = start_time_ = time.time()
frame_count = frame_count_ = 0


def frame_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break

        # frame = cv2.resize(frame, (1200, 600))

        if not frame_queue.full():
            frame_queue.put(frame)

def yolo_worker(frame_queue, result_queue, model, stop_event):
    global yolo_fps

    start_time = time.time()
    frame_count = 0

    while True:
        if stop_event.is_set() and frame_queue.empty():
            break

        try:
            frame = frame_queue.get(timeout=0.1)
        except:
            continue

        result = model(frame, verbose=False)[0]

        result_queue.put((frame, result))

        frame_count += 1

        elapsed = time.time() - start_time
        # if elapsed >= 1.0:
        with fps_lock:
            yolo_fps = frame_count / (elapsed + 1e-6)
        frame_count = 0
        start_time = time.time()

def display_worker(result_queue, tracker, stop_event, video_writer):
    global display_fps

    start_time = time.time()
    frame_count = 0

    while True:
        if stop_event.is_set() and result_queue.empty():
            break

        try:
            image, result = result_queue.get(timeout=0.1)
        except:
            continue

        frame_count += 1
        elapsed = time.time() - start_time

        # if elapsed >= 1.0:
        with fps_lock:
            display_fps = frame_count / (elapsed + 1e-6)
        frame_count = 0
        start_time = time.time()


        # timer = cv2.getTickCount()

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detection_zone.trigger(detections)]

        vehicle_detections = detections[
            (detections.class_id[:, None] == vehicle_classes).any(axis=1)
        ]
        vehicle_detections.class_id[:] = 0
        vehicle_detections = vehicle_detections.with_nms(threshold=0.5)

        vehicle_detections = tracker.update_with_detections(vehicle_detections)


        points = vehicle_detections.get_anchors_coordinates(
                anchor=sv.Position.CENTER
            )
        points = view_transformer.transform_points(points=points).astype(int)

        for tracker_id, [x, y] in zip(vehicle_detections.tracker_id, points):
            coordinates[tracker_id].append((x, y))

        labels = []
        for tracker_id in vehicle_detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = np.linalg.norm(
                    np.array(coordinate_start) - np.array(coordinate_end)  # distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
                )
                time_ = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time_ * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")

        LINE_COUNTER.trigger(vehicle_detections)

        annotated = TRACE_ANNOTATOR.annotate(image.copy(), vehicle_detections)
        annotated = BOX_ANNOTATOR.annotate(annotated, vehicle_detections)
        annotated = LABEL_ANNOTATOR.annotate(annotated, vehicle_detections, labels=labels)
        annotated = LINE_COUNTER_ANNOTATOR.annotate(annotated, line_counter=LINE_COUNTER)

        annotated = sv.draw_polygon(annotated, polygon=SOURCE, color=sv.Color.RED, thickness=thickness)

        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        with fps_lock:
            yolo_fps_local = yolo_fps
            display_fps_local = display_fps

        cv2.putText(
            annotated, f"YOLO FPS: {int(yolo_fps_local)}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 255), 4
        )

        cv2.putText(
            annotated, f"Display FPS: {int(display_fps_local)}",
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 0), 4
        )

        display_frame = cv2.resize(
            annotated,
            DISPLAY_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        video_writer.write(display_frame)

        cv2.imshow("YOLO Realtime", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break


frame_queue = Queue(maxsize=32)    # buffer đọc
result_queue = Queue(maxsize=32)   # buffer kết quả
stop_event = threading.Event()

yolo_fps = 0.0
display_fps = 0.0

fps_lock = threading.Lock()

reader = threading.Thread(
    target=frame_reader,
    args=(cap, frame_queue, stop_event),
    daemon=True
)

yolo_thread = threading.Thread(
    target=yolo_worker,
    args=(frame_queue, result_queue, YOLO_MODEL, stop_event),
    daemon=True
)

display_thread = threading.Thread(
    target=display_worker,
    args=(result_queue, TRACKER, stop_event, video_writer),
    daemon=True
)

reader.start()
yolo_thread.start()
display_thread.start()

reader.join()
yolo_thread.join()
display_thread.join()

cap.release()
cv2.destroyAllWindows()