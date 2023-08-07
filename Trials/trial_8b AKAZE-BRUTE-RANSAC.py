import cv2 #OpenCV library (Computer Vision)
import numpy as np 
import os


def find_matches_in_video(image_folder, video_path):
    # Load the video
    video = cv2.VideoCapture(video_path)
    # Error message identification
    if not video.isOpened():
        print(f"Error: Unable to open video '{video_path}'")
        return
    
    # Feature-detector: AKAZE detector (keypoint detection)
    detector = cv2.AKAZE_create()
    
    # Lists to store timestamps and XY coordinates for printing results at the end
    timestamps = []
    xy_coordinates = []
    
    # Loop through all images in image_folder
    for image_filename in os.listdir(image_folder):
        if image_filename.startswith('.'): # Identify hidden files
            continue  # Skip hidden files
        # Loads images in grayscale
        image_path = os.path.join(image_folder, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #Error message identification
        if image is None:
            print(f"Error: Unable to load image '{image_path}'")
            continue
        
        # Detect keypoints and descriptors in the image
        kp1, des1 = detector.detectAndCompute(image, None) # mask = None
        #Error message identification
        if des1 is None:
            print(f"Error: No keypoints detected in image '{image_path}'")
            continue        
        # Loop iterates through each frame of the video
        while True:
            ret, frame = video.read() # ret indicates if the frame has been read
            if not ret: # Indicates when the end of the video is reached
                break
            
            # Convert the frame to grayscale (intensity)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect keypoints and descriptors computed for grayscale video frame
            kp2, des2 = detector.detectAndCompute(frame_gray, None)
            
            # Feature-matching: brute-force matcher
            bf = cv2.BFMatcher()
            
            # Match the descriptors from image (des1) and frame (des2) while returning top 2 (k=2) matches for each descriptor
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test to find good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance: # Threshold
                    good_matches.append(m)
            # good_matches minimum requirement
            if len(good_matches) >= 8:  # Threshold
                # Extract coordinates of matched keypoints in the frame
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Outlier Rejection & homography matrix: RANSAC
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # if valid homography matrix (M) was calculated:
                if M is not None:
                    # Transform coordinates using the homography matrix
                    transformed_pts = cv2.perspectiveTransform(src_pts, M)
                    frame_with_indicators = frame.copy()
                    # Draw indicators at matched keypoints in the video
                    for i in range(len(good_matches)):
                        pt2 = (int(dst_pts[i][0][0]), int(dst_pts[i][0][1]))
                        cv2.circle(frame, pt2, 5, (0, 0, 255), -1)  # Red circle
                    frame_filename = f"matched_frame_{i}.jpg"  # Create a unique filename
                    frame_path = os.path.join("matched_keypoints", frame_filename)
                    cv2.imwrite(frame_path, frame_with_indicators)
                    # Store timestamp and transformed XY coordinates
                    timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    xy_coords = transformed_pts.squeeze().tolist()
                    timestamps.append(timestamp)
                    xy_coordinates.append(xy_coords)
                    
                    # Display frame with drawn indicators
                    cv2.imshow("Matched Keypoints", frame)
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
                        break  # Exit the while loop after a match is found
        
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning for the next image    
    # Release the video capture and closed open windows
    video.release()
    cv2.destroyAllWindows()
    
    # Loop through images
    for i, image_filename in enumerate(os.listdir(image_folder)):
        if image_filename.startswith('.'): # Identify hidden files
            continue  # Skip hidden files
        # Print timestamps and XY coordinates (keypoints) for each image found in the video
        if i < len(timestamps):
            print(f"Image '{image_filename}' found at:")
            print(f"Timestamp: {timestamps[i]} seconds")
            print(f"XY Coordinates: {xy_coordinates[i]}")
            print()
        # Error message Identification
        else:
            print(f"Image '{image_filename}' not found in the video.")

if __name__ == "__main__":
    image_folder = "ImageSample"
    video_path = "sample.mp4"
    find_matches_in_video(image_folder, video_path) # Initiate matching process

"""
Image 'sample1.png' found at:
Timestamp: 20.240000000000002 seconds
XY Coordinates: [[558.2554931640625, 626.4820556640625], [509.42620849609375, 651.30322265625], [515.2879028320312, 618.7548828125], [521.427490234375, 619.61767578125], [555.3560791015625, 639.9434204101562], [490.9373779296875, 644.7142333984375], [492.9437255859375, 649.383544921875], [533.88134765625, 674.5911254882812], [550.9039306640625, 676.439453125], [486.380615234375, 676.7236328125], [505.8358154296875, 686.338623046875], [538.3743896484375, 689.4164428710938], [511.5851745605469, 619.7493896484375], [504.424072265625, 623.6964721679688], [531.65625, 626.0196533203125], [513.028564453125, 671.7600708007812], [518.7321166992188, 619.9180908203125]]

Image 'sample2.png' found at:
Timestamp: 32.480000000000004 seconds
XY Coordinates: [[621.11962890625, 165.72216796875], [151.623046875, 157.3228302001953], [43.449161529541016, 157.83755493164062], [657.970947265625, 662.8660278320312], [235.34222412109375, 99.89921569824219], [204.8343963623047, 100.36119842529297], [211.20846557617188, 237.67628479003906], [236.36880493164062, 100.26449584960938]]
"""