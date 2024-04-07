'''
input video
output is basketball homography dataset
'''
import cv2
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

WHITE = (255, 255, 255)

# not sure how important this is, but will make imgs same size as soccer just in case
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# Reading input video, setting up save directories
# overarching is dataset/ncaa_bball
input_video_path = "C:/Users/simon/OneDrive/Desktop/senior ds capstone/video_capstone/20230220_WVU_OklahomaSt.mov"
video_name = os.path.basename(input_video_path)
# If you want to remove the file extension as well
video_name_without_extension = os.path.splitext(video_name)[0]

video_directory = os.path.join('dataset', 'ncaa_bball', video_name_without_extension)
img_directory = os.path.join(video_directory, 'images')  # images will be output as jpg here
homography_directory = os.path.join(video_directory, 'annotations')  # corresponding homography matrices will be output as npy here
print(homography_directory)
if not os.path.exists(video_directory):
    os.makedirs(video_directory)
    print(f"Directory '{video_directory}' created.")
    os.makedirs(img_directory)
    print(f"Directory '{img_directory}' created.")
    os.makedirs(homography_directory)
    print(f"Directory '{homography_directory}' created.")
else:
    print(f"Directory '{video_directory}' already exists.")



cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

skip_length = 3  # we read 1 frame every <skip_length> seconds

frame_skip = int(fps * skip_length)


# Define a callback function to get mouse click coordinates
def click_event(event, x, y, flags, param):
    # TODO give option to redo last click
    # TODO give option to skip go to next frame, (might be wrong camera angle)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        src_points.append((x, y))
        cv2.circle(show_frame, (x, y), 5, WHITE, -1)

        court_point = input("Enter corresponding court diagram point: ")
        while True:
            try:
                dst_points.append(court_diagram_points[court_point])
                break
            except KeyError as e:
                court_point = input("Not valid court point. Enter corresponding court diagram point again: ")

        if len(src_points) >= 4:
            should_continue = input("Continue annotating? (y/n): ")
            if should_continue.lower() != 'y':
                param[0] = False  # Stop annotating


# Predefined court diagram points
# court is 94 by 50, (0,0) is top left
court_diagram_points = {
    # whole court
    'top left': (0, 0),
    'top middle': (47, 0),
    'top right': (94, 0),
    'bottom right': (94, 50),
    'bottom middle': (47, 50),
    'bottom left': (0, 50),

    # left key
    'lk top left': (0, 19),
    'lk top right': (19, 19),
    'lk bottom right': (19, 31),
    'lk bottom left': (0, 31),

    # right key
    'rk top left': (75, 19),
    'rk top right': (94, 19),
    'rk bottom right': (94, 31),
    'rk bottom left': (75, 31),
    # center circle
    'cc left': (41, 25),
    'cc top': (47, 19),
    'cc right': (53, 25),
    'cc bottom': (47, 31),
    # other
    # 3pt line
    # NOTE: as of ~2022 (?) NCAA mens and womens have same 3 pt line
    '3 top left': (0, 40.125/12), # where the 3 pt line intersects the baseline on the upper half of left side of court
    '3 top right': (94, 40.125/12), # where the 3 pt line intersects the baseline on the upper half of right side of court
    '3 bottom right': (94, 50 - 40.125/12), # where the 3 pt line intersects the baseline on the bottom half of right side of court
    '3 bottom left': (0, 50 - 40.125 / 12), # where the 3 pt line intersects the baseline on the bottom half of left side of court
    '3 top key left': (22 + 64.75/12, 25), # top of the key 3 on left side of court
    '3 top key right': (94 - (22 + 64.75/12), 25) # top of the key 3 on right side of court

}

### test whether the court diagram points are correct
# Separate the x and y coordinates
x_coords = [coord[0] for coord in court_diagram_points.values()]
y_coords = [coord[1] for coord in court_diagram_points.values()]

# Plot the points
plt.scatter(x_coords, y_coords)

# Optionally, add labels for each point
for label, (x, y) in court_diagram_points.items():
    plt.annotate(label, (x, y))

# Invert the y-axis
plt.gca().invert_yaxis()

# Set the same spacing for x and y axes
plt.axis('equal')
# Show the plot
plt.show()



ret_val, frame = cap.read()
frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.moveWindow('Frame', 0, 0)

while ret_val:
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    src_points = []
    dst_points = []
    annotating = [True]  # Use a mutable object to allow modification inside the callback

    show_frame = copy.copy(frame)  # draw circles on copy of frame, but save original frame

    cv2.imshow('Frame', show_frame)
    cv2.waitKey(1)
    annotate = True
    if input('do you want to annotate this frame? (y/n): ').lower() == 'y':
        annotate = True
    else:
        print(f'skipping frame {current_frame}')
        annotate = False

    if annotate:
        while annotating[0]:
            cv2.imshow('Frame', show_frame)
            cv2.setMouseCallback('Frame', click_event, annotating)
            cv2.waitKey(1)  # Wait for a key press for 1 ms

        # Now we have a list of at least 4 (src, dst) pairs
        print("Source points:", src_points)
        print("Destination points:", dst_points)

        # Compute homography here if needed
        homography_matrix, _ = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC, 10)
        print(homography_matrix)
        img_save_file = f'{img_directory}/frame_{current_frame}.png'
        homo_save_file = f'{homography_directory}/frame_{current_frame}.npy'
        print(f'saving image to: {img_save_file}')
        print(f'saving homography to: {homo_save_file}')

        cv2.imwrite(img_save_file, frame)
        np.save(homo_save_file, homography_matrix)

    if input('continue to next frame? (y/n): ').lower() == 'y':
        pass
    else:
        print('ending annotation script. you annotated x frames in this session!')
        break
    # Read next frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame + frame_skip))
    ret_val, frame = cap.read()
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

cv2.destroyAllWindows()
cap.release()
