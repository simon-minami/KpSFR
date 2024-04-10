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
    right_key = np.array([
        (75, 19), (94, 19), (94, 31), (75, 31)
    ], dtype=float)
    half_court = np.array([
        (47, 0), (47, 50)
    ], dtype=float)

    court_corners = court_corners.reshape(-1, 1, 2)  # need to reshape for transformation
    right_key = right_key.reshape(-1, 1, 2)
    half_court = half_court.reshape(-1, 1, 2)

    court_corners_video = cv2.perspectiveTransform(court_corners, court_to_video)
    right_key_video = cv2.perspectiveTransform(right_key, court_to_video)
    half_court_video = cv2.perspectiveTransform(half_court, court_to_video)
    # print(court_corners_video, court_corners_video.shape)

    court_corners_video = court_corners_video.astype(int).reshape(-1, 2)
    right_key_video = right_key_video.astype(int).reshape(-1, 2)
    half_court_video = half_court_video.astype(int).reshape(-1, 2)

    # prinkt(court_corners_video, court_corners_video.shape)
    pt1 = court_corners_video[0, :]
    pt2 = court_corners_video[1, :]
    pt3 = court_corners_video[2, :]
    pt4 = court_corners_video[3, :]



    cv2.line(img, pt1, pt2, (0, 0, 255), 3)
    cv2.line(img, pt2, pt3, (0, 0, 255), 3)
    cv2.line(img, pt3, pt4, (0, 0, 255), 3)
    cv2.line(img, pt4, pt1, (0, 0, 255), 3)

    pt1 = right_key_video[0, :]
    pt2 = right_key_video[1, :]
    pt3 = right_key_video[2, :]
    pt4 = right_key_video[3, :]

    cv2.line(img, pt1, pt2, (0, 0, 255), 3)
    cv2.line(img, pt2, pt3, (0, 0, 255), 3)
    cv2.line(img, pt3, pt4, (0, 0, 255), 3)
    cv2.line(img, pt4, pt1, (0, 0, 255), 3)

    pt1 = half_court_video[0, :]
    pt2 = half_court_video[1, :]

    cv2.line(img, pt1, pt2, (0, 0, 255), 3)

    # print(court_corners_video)
    cv2.imshow('frame', img)
    if cv2.waitKey(0) == ord('s'):
        cv2.imwrite("C:/Users/simon/Downloads/homogpic.png", img)

cv2.destroyAllWindows()

