import cv2
import os

def find_matches_in_video(image_folder, video_path):
    # Load the video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Unable to open video '{video_path}'")
        return

    # Initialize AKAZE detector
    detector = cv2.AKAZE_create()

    # Lists to store timestamps and XY coordinates
    timestamps = []
    xy_coordinates = []

    # Loop through all images in the folder
    for image_filename in os.listdir(image_folder):
        if image_filename.startswith('.'):
            continue  # Skip hidden files like .DS_Store

        image_path = os.path.join(image_folder, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Unable to load image '{image_path}'")
            continue

        # Detect keypoints and descriptors in the image
        kp1, des1 = detector.detectAndCompute(image, None)

        if des1 is None:
            print(f"Error: No keypoints detected in image '{image_path}'")
            continue

        # Flag to indicate if the image is found
        image_found = False

        # Loop through the frames of the video
        while True:
            ret, frame = video.read()

            if not ret:
                break

            # Convert the frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and descriptors in the frame
            kp2, des2 = detector.detectAndCompute(frame_gray, None)

            # Create a brute-force matcher
            bf = cv2.BFMatcher()

            # Match the descriptors
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test to find good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= 8:  # Adjust this threshold as needed
                # Get coordinates of matched keypoints in the frame
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Store timestamp and XY coordinates
                timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                xy_coords = dst_pts.squeeze().tolist()

                timestamps.append(timestamp)
                xy_coordinates.append(xy_coords)

                image_found = True
                
                # Print timestamp and draw the image on the frame
                timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                print(f"Match found at timestamp: {timestamp} seconds")
                cv2.imshow("Matched Keypoints", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
                    break

        if not image_found:
            print(f"Image '{image_filename}' not found in the video.")

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning for the next image

    # Release the video capture
    video.release()
    cv2.destroyAllWindows()

    # Print timestamps and XY coordinates for each image found in the video
    for i, image_filename in enumerate(os.listdir(image_folder)):
        if image_filename.startswith('.'):
            continue  # Skip hidden files like .DS_Store

        if i < len(timestamps):
            print(f"Image '{image_filename}' found at:")
            print(f"Timestamp: {timestamps[i]} seconds")
            print(f"XY Coordinates: {xy_coordinates[i]}")
            print()
        else:
            print(f"Image '{image_filename}' not found in the video.")

if __name__ == "__main__":
    import numpy as np
    image_folder = "ImageSample"  # Replace with your image folder path
    video_path = "sample.mp4"     # Replace with your video path
    find_matches_in_video(image_folder, video_path)

"""
Match found at timestamp: 1.16 seconds
Match found at timestamp: 20.240000000000002 seconds
Match found at timestamp: 32.480000000000004 seconds
Match found at timestamp: 34.4 seconds
Match found at timestamp: 44.72 seconds

Image 'sample1.png' found at:
Timestamp: 20.240000000000002 seconds
XY Coordinates: [[488.802001953125, 579.41259765625], [50.85037612915039, 476.86322021484375], [514.5361938476562, 618.9414672851562], [514.5361938476562, 618.9414672851562], [556.0205078125, 640.3312377929688], [492.3368835449219, 649.1536254882812], [492.3368835449219, 649.1536254882812], [534.3018188476562, 673.9610595703125], [550.3889770507812, 676.545166015625], [486.55816650390625, 676.6215209960938], [505.55621337890625, 686.2584228515625], [538.6546020507812, 689.2234497070312], [514.5361938476562, 618.9414672851562], [504.348876953125, 624.367431640625], [530.4962158203125, 625.6804809570312], [513.1575927734375, 672.7561645507812], [517.4977416992188, 619.9534301757812]]

Image 'sample2.png' found at:
Timestamp: 32.480000000000004 seconds
XY Coordinates: [[621.1135864257812, 165.72877502441406], [482.8063659667969, 610.7992553710938], [665.5355834960938, 37.69124221801758], [657.9765014648438, 662.8613891601562], [235.86492919921875, 100.07275390625], [530.4633178710938, 621.0523681640625], [211.19015502929688, 237.6925048828125], [235.86492919921875, 100.07275390625]]
"""