import cv2
import numpy as np
import os

def find_images_in_video(folder_path, video_path):
    # Load the images from the folder and the video clip
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]
    video_clip = cv2.VideoCapture(video_path)

    # Load YOLOv4 model with COCO classes
    net = cv2.dnn.readNetFromDarknet('YOLOv4/yolov4-tiny.cfg', 'YOLOv4/yolov4.weights')
    classes = []
    with open('YOLOv4/coco.names', 'r') as f:
        classes = f.read().strip().split('\n')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Step 2: Object Detection with YOLOv4
    def detect_objects(frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust confidence threshold based on your requirements
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    # Step 3: Find Image1 in Video
    boxes_image1, confidences_image1, class_ids_image1 = detect_objects(image1)

    # Step 4: Find Image2 in Video
    boxes_image2, confidences_image2, class_ids_image2 = detect_objects(image2)

    # Step 5: Display Video with Indication
    while True:
        ret, frame = video_clip.read()
        if not ret:
            break

        for box, confidence, class_id in zip(boxes_image1, confidences_image1, class_ids_image1):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{classes[class_id]}: {confidence:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for box, confidence, class_id in zip(boxes_image2, confidences_image2, class_ids_image2):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{classes[class_id]}: {confidence:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Video Clip', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
            break

    video_clip.release()
    cv2.destroyAllWindows()

# Example usage:
image_folder = 'ImageSample'
video_path = 'sample.mp4'
find_images_in_video(image_paths, video_path)
