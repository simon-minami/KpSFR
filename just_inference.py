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
from models.network import KpSFR
import utils
from models.inference_core import InferenceCore
import cv2
import time
from torchvision import transforms


def class_mapping(rgb, template):
    src_pts = rgb.copy()
    cls_map_pts = []

    for ind, elem in enumerate(src_pts):
        coords = np.where(elem[2] == template[:, 2])[0]  # find correspondence
        cls_map_pts.append(template[coords[0]])
    dst_pts = np.array(cls_map_pts, dtype=np.float32)

    return src_pts[:, :2], dst_pts[:, :2]


def infer_homography(model, image, device, template_grid):
    with torch.no_grad():
        processor = InferenceCore(model, image, device, k=91, lookup=None)
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
    device = torch.device('cuda')
    print('device: %s' % device)

    num_classes = 92
    # num_objects = opt.num_objects
    num_objects = 91
    non_local = bool(1)
    model_archi = 'KC'
    # Initialize models
    model = KpSFR(model_archi=model_archi,
                  num_objects=num_objects, non_local=non_local).to(device)

    load_weights_path = 'checkpoint/kpsfr_finetuned.pth'
    print('Loading weights: ', load_weights_path)
    assert osp.isfile(load_weights_path), 'Error: no checkpoints found'
    checkpoint = torch.load(load_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess the image (same procedure as ts_worldcup_test_loader.py)
    preprocess = transforms.Compose([
        # transforms.Resize((720, 1280)),  # Resize to match the dimensions used in training
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize (ImageNet statistics)
    ])

    # Load and preprocess the image
    input_image_path = 'dataset/ncaa_bball/images/20230220_WVU_OklahomaSt/frame_1.jpg'
    image = Image.open(input_image_path)
    print(f'image after opening: {np.array(image).shape}')
    image_tensor = preprocess(image).to(device)
    print(f'image before unsqeeze: {image_tensor.size()}')

    image_tensor = image_tensor.unsqueeze(0).unsqeeze(0)  # Add batch and frame dimensions: [1, 1, channels, height, width]
    print(f'image after unsqeeze: {image_tensor.size()}')

    # shape should now be [1, 1, channels, height, width]
    # Predict homography
    # I should be able to use the given postprocessing function in the original inference.py

    ##TEST
    print(image_tensor[:, 0])
    print(image_tensor[:, 0].size())

    model.eval()
    # Encode key features and segment the image
    with torch.no_grad():
        f32, f16, f8, f4 = model.encode_key(image_tensor)  # Add batch dimension
        predicted_heatmaps = model.segment(f32, f16, f8, f4, k=91,
                                           qcls=None)  # qcls can be None or based on your implementation

    predicted_heatmaps(len(predicted_heatmaps), predicted_heatmaps)

    print('hi')
    return

    # # Load the input image
    # input_image = Image.open('path_to_input_image.jpg')
    # input_tensor = utils.im_to_torch(input_image).unsqueeze(0).to(device)
    #
    # # Generate the template grid
    # template_grid = utils.gen_template_grid_bball()
    #
    # # Infer the homography
    # predicted_homography = infer_homography(model, input_tensor, device, template_grid)


def main():
    inference()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Done...Take {(time.time() - start_time):.4f} (sec)')
