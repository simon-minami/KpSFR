'''
new inference file that just does inference (original inference.py does
evaluation as well which we can't do with bball)

want to have input: image -> output: predicted homography matrix
'''



import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
from models.eval_network import EvalKpSFR
import utils
from models.inference_core import InferenceCore
import cv2
import time



def class_mapping(rgb, template):
    src_pts = rgb.copy()
    cls_map_pts = []

    for ind, elem in enumerate(src_pts):
        coords = np.where(elem[2] == template[:, 2])[0]  # find correspondence
        cls_map_pts.append(template[coords[0]])
    dst_pts = np.array(cls_map_pts, dtype=np.float32)

    return src_pts[:, :2], dst_pts[:, :2]

def infer_homography(eval_model, image, device, template_grid):
    with torch.no_grad():
        processor = InferenceCore(eval_model, image, device, k=91, lookup=None)
        processor.interact(0, image.shape[1], selector=None)

        out_masks = processor.prob[:, 0]  # Assuming single frame input
        out_scores, out_masks = torch.max(out_masks, dim=0)

        out_masks = out_masks.detach().cpu().numpy()
        out_scores = out_scores.detach().cpu().numpy()

        image = np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0))  # h*w*c
        pred_keypoints = np.zeros_like(out_masks)

        pred_rgb = []
        for ind in range(1, 92):  # 1 to 91 classes
            if np.any(out_masks == ind):
                y, x = np.unravel_index(np.argmax(out_masks == ind), out_masks.shape)
                pred_keypoints[y, x] = ind
                pred_rgb.append([x * 4, y * 4, ind])

        pred_rgb = np.asarray(pred_rgb, dtype=np.float32)
        pred_homo = None

        if pred_rgb.shape[0] >= 4:
            src_pts, dst_pts = class_mapping(pred_rgb, template_grid)
            pred_homo, _ = cv2.findHomography(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 10)

        return pred_homo

def inference():
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = 0
    print('CUDA Visible Devices: %s' % 0)
    device = torch.device('cuda:0')
    print('device: %s' % device)


    num_classes = 92
    # num_objects = opt.num_objects
    num_objects = 91
    non_local = bool(1)
    model_archi = 'KC'
    # Initialize models
    eval_model = EvalKpSFR(model_archi=model_archi,
                           num_objects=num_objects, non_local=non_local).to(device)

    load_weights_path = 'checkpoint/kpsfr_finetuned.pth'
    print('Loading weights: ', load_weights_path)
    assert osp.isfile(load_weights_path), 'Error: no checkpoints found'
    checkpoint = torch.load(load_weights_path, map_location=device)
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    return

    # Load the input image
    input_image = Image.open('path_to_input_image.jpg')
    input_tensor = utils.im_to_torch(input_image).unsqueeze(0).to(device)

    # Generate the template grid
    template_grid = utils.gen_template_grid_bball()

    # Infer the homography
    predicted_homography = infer_homography(eval_model, input_tensor, device, template_grid)
    print("Predicted Homography:", predicted_homography)

def main():

    inference()
    # writer.flush()
    # writer.close()


if __name__ == '__main__':

    start_time = time.time()
    main()
    print(f'Done...Take {(time.time() - start_time):.4f} (sec)')