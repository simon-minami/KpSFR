'''
checking to see if the make_dataset.py file actually creates the correct homographies

iterate through img, matrix pairs in dataset/ncaa_bball
draw court lines using given homography to check

AFTER LOOKING AT OUTPUT, MAKE_DATASET.PY SEEMS TO BE WORKING CORRECTLY
'''
import os
import os.path as osp
import cv2
import numpy as np

root = osp.join('dataset', 'ncaa_bball')
image_path = osp.join(root, 'images', '20230220_WVU_OklahomaSt')
annotation_path = osp.join(root, 'annotations', '20230220_WVU_OklahomaSt')


frames = sorted(os.listdir(image_path))
homographies = sorted(os.listdir(annotation_path))
# print(frames)
# print(homographies)

for frame, homo in zip(frames, homographies):
    img = cv2.imread(osp.join(image_path, frame))

    # dataset is video to court, we want court to video to draw lines
    court_to_video = np.linalg.inv(np.load(osp.join(annotation_path, homo)))
    court_to_video = court_to_video.astype(float)  # Convert to float32
    court_corners = np.array([
        [0,0], [94, 0], [94, 50], [0, 50]
    ], dtype=float)
    court_corners = court_corners.reshape(-1, 1, 2)  # need to reshape for transformation
    court_corners_video = cv2.perspectiveTransform(court_corners, court_to_video)
    court_corners_video = court_corners_video.astype(int).reshape(-1, 2)
    print(court_corners_video, court_corners_video.shape)
    pt1 = court_corners_video[0, :]
    print(pt1)
    pt2 = court_corners_video[1, :]
    pt3 = court_corners_video[2, :]
    pt4 = court_corners_video[3, :]

    cv2.line(img, pt1, pt2, (0, 0, 255), 3)
    cv2.line(img, pt2, pt3, (0, 0, 255), 3)
    cv2.line(img, pt3, pt4, (0, 0, 255), 3)
    cv2.line(img, pt4, pt1, (0, 0, 255), 3)

    # print(court_corners_video)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

