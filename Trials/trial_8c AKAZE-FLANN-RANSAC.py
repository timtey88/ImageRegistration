import cv2
import numpy as np
import os

def find_matches_in_video(image_folder, video_path):
    # Load the video
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Unable to open video '{video_path}'")
        return
    
    # Initialize AKAZE detector
    detector = cv2.AKAZE_create()
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=200)
    
    # Create FLANN based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
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
        
        # Convert descriptors to float32 format
        des1 = des1.astype(np.float32)
        
        # Loop through the frames of the video
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Convert the frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and descriptors in the frame
            kp2, des2 = detector.detectAndCompute(frame_gray, None)
            
            if des2 is None:
                continue
            
            # Convert descriptors to float32 format
            des2 = des2.astype(np.float32)
            
            # FLANN based matching
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test to find good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) >= 8:  # Adjust this threshold as needed
                # Get coordinates of matched keypoints in the frame
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography using RANSAC
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # Transform coordinates using the homography matrix
                    transformed_pts = cv2.perspectiveTransform(src_pts, M)
                    
                    # Draw indicators at matched keypoints in the video
                    for i in range(len(good_matches)):
                        pt2 = (int(dst_pts[i][0][0]), int(dst_pts[i][0][1]))
                        cv2.circle(frame, pt2, 5, (0, 0, 255), -1)  # Red circle
                    
                    # Store timestamp and transformed XY coordinates
                    timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    xy_coords = transformed_pts.squeeze().tolist()
                    
                    timestamps.append(timestamp)
                    xy_coordinates.append(xy_coords)
                    
                    # Print timestamp and draw the image on the frame
                    print(f"Match found at timestamp: {timestamp} seconds")
                    cv2.imshow("Matched Keypoints", frame)
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
                        break  # Exit the while loop after a match is found
        
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
    image_folder = "ImageSample"
    video_path = "sample.mp4"
    find_matches_in_video(image_folder, video_path)

"""
FLANN nodes check set at 200 

Match found at timestamp: 1.16 seconds
Match found at timestamp: 20.240000000000002 seconds
Match found at timestamp: 30.76 seconds
Match found at timestamp: 32.480000000000004 seconds
Match found at timestamp: 34.4 seconds
Match found at timestamp: 44.72 seconds
Image 'sample1.png' found at:
Timestamp: 20.240000000000002 seconds
XY Coordinates: [[557.4749755859375, 625.9085083007812], [508.9490966796875, 651.215087890625], [514.2134399414062, 618.7390747070312], [520.309326171875, 619.509521484375], [554.80078125, 639.5478515625], [490.7090759277344, 644.745849609375], [492.7330322265625, 649.365966796875], [533.7359008789062, 674.6246337890625], [551.0117797851562, 676.583740234375], [486.71441650390625, 676.5313110351562], [505.9559326171875, 686.2738647460938], [538.5477294921875, 689.6797485351562], [510.5741882324219, 619.7800903320312], [519.9464111328125, 619.4546508789062], [503.59100341796875, 623.79248046875], [530.6143188476562, 625.7679443359375], [512.845947265625, 671.6881103515625], [517.6412963867188, 619.846923828125]]

Image 'sample2.png' found at:
Timestamp: 30.76 seconds
XY Coordinates: [[379.751220703125, 526.8460693359375], [507.8310241699219, 618.3010864257812], [281.9552307128906, 577.4815673828125], [474.56707763671875, 768.5269775390625], [94.42505645751953, 175.5227813720703], [318.57904052734375, 344.29559326171875], [507.71820068359375, 617.9428100585938], [521.09912109375, 634.9835815429688]]
"""