from ultralytics import YOLO
import cv2
import numpy as np

from collections import defaultdict, deque
import time

import supervision as sv

from view_transform import ViewTransformer



YOLO_MODEL = YOLO("models/yolo26n.pt", task="detect")
VIDEO_PATH = "data/vehicles.mp4"

# Video info
video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)
thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

# Tracker
TRACKER = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=0.3)

# Annotators
BOX_ANNOTATOR = sv.BoxAnnotator(thickness=thickness)
LABEL_ANNOTATOR = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
TRACE_ANNOTATOR = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

# Homography points (video specific)
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

# Real world target dimensions
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

DISPLAY_SIZE = (1200, 600)

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
detection_zone = sv.PolygonZone(polygon=SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

cap = cv2.VideoCapture(VIDEO_PATH)

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    timer = cv2.getTickCount()

    # image = cv2.resize(frame, (1200, 600))
    image = frame.copy()


    result = YOLO_MODEL(image, verbose=False)[0]
    

    # Detect vehicles
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detection_zone.trigger(detections)]
    vehicle_detections = detections[
        (detections.class_id[:, None] == vehicle_classes).any(axis=1)
    ]
    vehicle_detections.class_id[:] = 0

    vehicle_detections = vehicle_detections.with_nms(threshold=0.5)

    # Track vehicles (frame-by-frame)
    vehicle_detections = TRACKER.update_with_detections(vehicle_detections)

    # Get vehicle bottom center points
    points = vehicle_detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )
    # Homography transform to real world coordinates
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

    annotated = TRACE_ANNOTATOR.annotate(image, vehicle_detections)
    annotated = BOX_ANNOTATOR.annotate(annotated, vehicle_detections)
    annotated = LABEL_ANNOTATOR.annotate(annotated, vehicle_detections, labels=labels)

    annotated = sv.draw_polygon(annotated, polygon=SOURCE, color=sv.Color.RED, thickness=thickness)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(
        annotated, f"FPS: {int(fps)}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    display_frame = cv2.resize(
        annotated,
        DISPLAY_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imshow("YOLO Realtime", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()