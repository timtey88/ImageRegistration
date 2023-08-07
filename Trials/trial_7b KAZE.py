import cv2
import os

def find_matches_in_video(image_folder, video_path):
    # Load the video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Unable to open video '{video_path}'")
        return

    # Initialize KAZE detector
    detector = cv2.KAZE_create()
    
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
            
            if len(good_matches) >= 40:  #50 found only at 34.4
                # Get coordinates of matched keypoints in the frame
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Draw indicators at matched keypoints in the video
                for i in range(len(good_matches)):
                    pt2 = (int(dst_pts[i][0][0]), int(dst_pts[i][0][1]))
                    cv2.circle(frame, pt2, 5, (0, 0, 255), -1)  # Red circle

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
    
