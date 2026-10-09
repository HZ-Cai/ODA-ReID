import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
import math
import random
import numpy as np
import cv2

from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from tensorfn import load_config as DiffConfig
from config.diffconfig import DiffusionConfig, get_model_conf
from diffusion import create_gaussian_diffusion, ddim_steps

'''
This pipeline uses a pose-conditioned diffusion model to generate images from 
an appearance image and random keypoints, with optional occlusion augmentation.
'''

class PoseGenerationDataset:

    def __init__(
        self,
        ckpt_path,

        pose_dir,

        mask_dir,

        output_img_dir,
        output_pose_dir,

        dataset_type='duke',

        image_size=(128, 64)
    ):

        self.dataset_type = dataset_type

        self.image_size = image_size

        H, W = image_size

        # ==========================================================
        # output path
        # ==========================================================
        self.output_img_dir = output_img_dir
        self.output_pose_dir = output_pose_dir

        # image
        self.output_img_nomask_dir = os.path.join(output_img_dir, "nomask")
        self.output_img_mask_dir = os.path.join(output_img_dir, "mask")

        # pose
        self.output_pose_nomask_dir = os.path.join(output_pose_dir, "nomask")
        self.output_pose_mask_dir = os.path.join(output_pose_dir, "mask")

        os.makedirs(self.output_img_nomask_dir, exist_ok=True)
        os.makedirs(self.output_img_mask_dir, exist_ok=True)

        os.makedirs(self.output_pose_nomask_dir, exist_ok=True)
        os.makedirs(self.output_pose_mask_dir, exist_ok=True)

        # ==========================================================
        # pose txt list
        # ==========================================================
        self.pose_list = sorted(
            glob.glob(os.path.join(pose_dir, '*.txt'))
        )

        print(f'Find {len(self.pose_list)} pose txt files')

        # ==========================================================
        # mask npy list
        # ==========================================================
        self.mask_list = sorted(
            glob.glob(os.path.join(mask_dir, '*.npy'))
        )

        print(f'Find {len(self.mask_list)} mask npy files')

        # ==========================================================
        # skeleton
        # ==========================================================
        self.limbSeq = [
            [2,3], [2,6], [3,4], [4,5],
            [6,7], [7,8], [2,9], [9,10],
            [10,11], [2,12], [12,13], [13,14],
            [2,1], [1,15], [15,17],
            [1,16], [16,18]
        ]

        # ==========================================================
        # skeleton colors
        # ==========================================================
        self.colors = [
            (255,0,0), (255,85,0), (255,170,0),
            (255,255,0), (170,255,0), (85,255,0),
            (0,255,0), (0,255,85), (0,255,170),
            (0,255,255), (0,170,255), (0,85,255),
            (0,0,255), (85,0,255), (170,0,255),
            (255,0,255), (255,0,170), (255,0,85)
        ]

        # ==========================================================
        # diffusion config
        # ==========================================================
        conf = DiffConfig(
            DiffusionConfig,
            './config/diffusion.conf',
            show=False
        )

        # ==========================================================
        # load model
        # ==========================================================
        self.model = get_model_conf().make_model()

        ckpt = torch.load(
            ckpt_path,
            map_location='cpu'
        )

        self.model.load_state_dict(
            ckpt["ema"]
        )

        self.model = self.model.cuda()

        self.model.eval()

        # ==========================================================
        # diffusion
        # ==========================================================
        self.betas = conf.diffusion.beta_schedule.make()

        self.diffusion = create_gaussian_diffusion(
            self.betas,
            predict_xstart=False
        )

        # ==========================================================
        # preprocess
        # ==========================================================
        self.transforms = transforms.Compose([

            transforms.Resize(
                image_size,
                interpolation=Image.BICUBIC
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

    # ==============================================================
    # filename generation
    # ==============================================================
    def build_new_name(
        self,
        original_name,
        global_index
    ):

        base_name = os.path.splitext(
            original_name
        )[0]

        # ==========================================================
        # DukeMTMC-reID / Occluded_Duke
        # ==========================================================
        if self.dataset_type in [
            'duke',
            'occluded_duke'
        ]:

            prefix = '_'.join(
                base_name.split('_')[:2]
            )

            new_name = (
                f'{prefix}_f{1000000 + global_index}.jpg'
            )

        # ==========================================================
        # market1501
        # ==========================================================
        elif self.dataset_type == 'market1501':

            prefix = '_'.join(
                base_name.split('_')[:2]
            )

            new_name = (
                f'{prefix}_{100000 + global_index:06d}_09.jpg'
            )

        # ==========================================================
        # Occluded_REID
        # ==========================================================
        elif self.dataset_type == 'occluded_reid':

            prefix = '_'.join(
                base_name.split('_')[:2]
            )

            new_name = (
                f'{prefix}_{10 + global_index}.jpg'
            )

        else:

            raise ValueError(
                f'Unknown dataset_type: {self.dataset_type}'
            )

        return new_name

    # ==============================================================
    # keypoint scaling
    # ==============================================================
    def trans_keypoins(
        self,
        keypoints,
        new_h,
        new_w,
        original_image_path
    ):

        missing_keypoint_index = keypoints == -1

        img = Image.open(original_image_path)

        orig_w, orig_h = img.size

        scale_w = new_w / orig_w

        scale_h = new_h / orig_h

        keypoints[:,0] = np.clip(
            keypoints[:,0] * scale_w,
            0,
            new_w - 1
        )

        keypoints[:,1] = np.clip(
            keypoints[:,1] * scale_h,
            0,
            new_h - 1
        )

        keypoints[missing_keypoint_index] = -1

        return keypoints

    # ==============================================================
    # build pose tensor
    # ==============================================================
    def build_pose_from_txt(
        self,
        txt_path,
        mask_np,
        use_mask=True,
        num_key=4
    ):

        H, W = self.image_size

        canvas = np.zeros(
            (H, W, 3),
            dtype=np.uint8
        )

        # ==========================================================
        # read keypoints
        # ==========================================================
        keypoint = np.loadtxt(txt_path)

        original_image_path = os.path.join(
            'XXX/Occluded_Duke/bounding_box_train',
            os.path.basename(txt_path).replace('.txt', '.jpg')
        )

        keypoint = self.trans_keypoins(
            keypoint,
            new_h=H,
            new_w=W,
            original_image_path=original_image_path
        )

        # ==========================================================
        # Block out keypoints
        # ==========================================================
        if use_mask:

            for i in range(18):

                x = int(keypoint[i, 0])

                y = int(keypoint[i, 1])

                if x == -1 or y == -1:
                    continue

                if (
                    x >= 0 and x < W and
                    y >= 0 and y < H
                ):

                    if mask_np[y, x] > 0:

                        keypoint[i] = [-1, -1]

        # ==========================================================
        # Count the valid key points
        # ==========================================================
        valid_num = np.sum(
            (keypoint[:,0] != -1) &
            (keypoint[:,1] != -1)
        )

        if valid_num < num_key:
            return None

        # ==========================================================
        # draw keypoint
        # ==========================================================
        for i in range(18):

            x, y = keypoint[i]

            if x == -1 or y == -1:
                continue

            cv2.circle(
                canvas,
                (int(x), int(y)),
                4,
                self.colors[i],
                thickness=-1
            )

        joints = []

        stickwidth = 4

        # ==========================================================
        # draw skeleton
        # ==========================================================
        for i in range(17):

            Y = keypoint[
                np.array(self.limbSeq[i]) - 1,
                0
            ]

            X = keypoint[
                np.array(self.limbSeq[i]) - 1,
                1
            ]

            cur_canvas = canvas.copy()

            if -1 in Y or -1 in X:

                joints.append(
                    np.zeros_like(cur_canvas[:, :, 0])
                )

                continue

            mX = np.mean(X)

            mY = np.mean(Y)

            length = (
                (X[0] - X[1]) ** 2 +
                (Y[0] - Y[1]) ** 2
            ) ** 0.5

            angle = math.degrees(
                math.atan2(
                    X[0] - X[1],
                    Y[0] - Y[1]
                )
            )

            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)),
                (int(length / 2), stickwidth),
                int(angle),
                0,
                360,
                1
            )

            cv2.fillConvexPoly(
                cur_canvas,
                polygon,
                self.colors[i]
            )

            canvas = cv2.addWeighted(
                canvas,
                0.4,
                cur_canvas,
                0.6,
                0
            )

            joint = np.zeros_like(
                cur_canvas[:, :, 0]
            )

            cv2.fillConvexPoly(
                joint,
                polygon,
                255
            )

            joints.append(joint)

        # ==========================================================
        # RGB skeleton
        # ==========================================================
        pose = F.to_tensor(
            Image.fromarray(
                cv2.cvtColor(
                    canvas,
                    cv2.COLOR_BGR2RGB
                )
            )
        )

        # ==========================================================
        # distance transform
        # ==========================================================
        tensors_dist = []

        for joint in joints:

            im_dist = cv2.distanceTransform(
                255 - joint,
                cv2.DIST_L1,
                3
            )

            im_dist = np.clip(
                (im_dist / 3),
                0,
                255
            ).astype(np.uint8)

            tensor_dist = F.to_tensor(
                Image.fromarray(im_dist)
            )

            tensors_dist.append(
                tensor_dist
            )

        tensors_dist = torch.cat(
            tensors_dist,
            dim=0
        )

        label_tensor = torch.cat(
            [pose, tensors_dist],
            dim=0
        )

        return label_tensor

    # ==============================================================
    # Batch generation
    # ==============================================================
    def generate_folder(
        self,
        image_dir,

        # x=1.0,

        # mask_prob=0.5,

        nomask_x=1.0,

        mask_x=1.0,

        sample_algorithm='ddim',

        nsteps=50,

        seed=1234,

        batch_size=16
    ):

        random.seed(seed)

        # ==========================================================
        # source image
        # ==========================================================
        all_image_list = sorted(
            glob.glob(
                os.path.join(image_dir, '*.jpg')
            )
        )

        total_images = len(all_image_list)

        print(f'Find {total_images} images')

        # target_num = int(total_images * x)

        # print(f'Target generate number: {target_num}')

        target_nomask = int(total_images * nomask_x)
        target_mask = int(total_images * mask_x)

        print(f'Generate nomask : {target_nomask}')
        print(f'Generate mask   : {target_mask}')

        tasks = [

            {
                "use_mask": False,
                "target_num": target_nomask,
                "img_dir": self.output_img_nomask_dir,
                "pose_dir": self.output_pose_nomask_dir,
            },

            {
                "use_mask": True,
                "target_num": target_mask,
                "img_dir": self.output_img_mask_dir,
                "pose_dir": self.output_pose_mask_dir,
            }

        ]

        global_index = 0

        for task in tasks:

            generated_count = 0

            use_mask = task["use_mask"]

            save_img_root = task["img_dir"]

            save_pose_root = task["pose_dir"]

            target_num = task["target_num"]

            while generated_count < target_num:        

                src_tensors = []

                pose_tensors = []

                save_img_paths = []

                save_pose_paths = []

                current_names = []

                # ======================================================
                # build batch
                # ======================================================
                while (
                    len(src_tensors) < batch_size and
                    generated_count < target_num
                ):

                    img_path = random.choice(all_image_list)

                    pose_txt = random.choice(
                        self.pose_list
                    )

                    mask_path = random.choice(
                        self.mask_list
                    )

                    mask_np = np.load(
                        mask_path
                    ).astype(np.float32)

                    # # ==================================================
                    # # whether to use obstruction
                    # # ==================================================
                    # use_mask = random.random() < mask_prob

                    # ==================================================
                    # no use mask
                    # ==================================================
                    if not use_mask:

                        mask_np = np.zeros_like(
                            mask_np
                        ).astype(np.float32)

                    # ==================================================
                    # build pose
                    # ==================================================
                    pose_tensor = self.build_pose_from_txt(
                        pose_txt,
                        mask_np,
                        use_mask=use_mask,
                        num_key=10 
                    )

                    if pose_tensor is None:
                        continue

                    # ==================================================
                    # appearance image
                    # ==================================================
                    src = Image.open(
                        img_path
                    ).convert('RGB')

                    src_tensor = self.transforms(src)

                    # ==================================================
                    # mask tensor
                    # ==================================================
                    mask_tensor = torch.from_numpy(
                        mask_np
                    ).unsqueeze(0).float()

                    pose_tensor = torch.cat(
                        [pose_tensor, mask_tensor],
                        dim=0
                    )

                    src_tensors.append(src_tensor)

                    pose_tensors.append(pose_tensor)

                    # ==================================================
                    # save path 
                    # ==================================================
                    original_name = os.path.basename(
                        img_path
                    )

                    new_name = self.build_new_name(
                        original_name,
                        global_index
                    )

                    save_img_paths.append(
                        os.path.join(
                            save_img_root,
                            new_name
                        )
                    )

                    save_pose_paths.append(
                        os.path.join(
                            save_pose_root,
                            new_name
                        )
                    )

                    current_names.append(
                        (original_name, new_name)
                    )

                    generated_count += 1
                    global_index += 1     # Avoid same name 

                    print(
                        f'[{generated_count}/{target_num}] '
                        f'{original_name} -> {new_name} | '
                        f'use_mask={use_mask}'
                    )

                # ======================================================
                # empty batch
                # ======================================================
                if len(src_tensors) == 0:
                    continue

                # ======================================================
                # stack batch
                # ======================================================
                src_batch = torch.stack(
                    src_tensors,
                    0
                ).cuda()

                pose_batch = torch.stack(
                    pose_tensors,
                    0
                ).cuda()

                # ======================================================
                # diffusion sampling
                # ======================================================
                with torch.no_grad():

                    if sample_algorithm == 'ddpm':

                        samples = self.diffusion.p_sample_loop(
                            self.model,
                            x_cond=[
                                src_batch,
                                pose_batch
                            ],
                            progress=False,
                            cond_scale=2
                        )

                    elif sample_algorithm == 'ddim':

                        noise = torch.randn_like(
                            src_batch
                        ).cuda()

                        seq = range(
                            0,
                            1000,
                            1000 // nsteps
                        )

                        xs, x0_preds = ddim_steps(
                            noise,
                            seq,
                            self.model,
                            self.betas.cuda(),
                            [
                                src_batch,
                                pose_batch
                            ]
                        )

                        samples = xs[-1]

                    else:

                        raise ValueError(
                            'Unknown sample_algorithm'
                        )

                # ======================================================
                # save result
                # ======================================================
                samples = torch.clamp(
                    samples,
                    -1.,
                    1.
                )

                samples = (
                    samples + 1.0
                ) / 2.0

                current_bs = len(src_tensors)

                for b in range(current_bs):

                    img = samples[b].permute(
                        1,
                        2,
                        0
                    ).detach().cpu().numpy()

                    img = (
                        img * 255
                    ).astype(np.uint8)

                    Image.fromarray(img).save(
                        save_img_paths[b]
                    )

                    pose_vis = pose_tensors[b][:3]

                    pose_vis = pose_vis.permute(
                        1,
                        2,
                        0
                    ).cpu().numpy()

                    pose_vis = (
                        pose_vis * 255
                    ).clip(0,255).astype(np.uint8)

                    Image.fromarray(
                        pose_vis
                    ).save(
                        save_pose_paths[b]
                    )

                    print('Done!')


# ==============================================================
# main
# ==============================================================
if __name__ == '__main__':

    generator = PoseGenerationDataset(

        # checkpoint
        ckpt_path='XXX/weight_idloss/checkpoints_B/last.pt',

        # pose txt
        pose_dir='XXX/occduke_128_64/bounding_box_train_pose_txt_wofail_n_2pose',

        # mask npy
        mask_dir='XXX/Occluded_Duke/data_aug_masks/masks_likequery',

        # output img 
        output_img_dir='XXX/Occluded_Duke/diffusion_B_idloss/masks/generated_images_pidm_occduke_wofail_n_2pose_npy_wofail_n_2pose_app_wofail_n_2pose',
        
        # potput visible pose 
        output_pose_dir='XXX/Occluded_Duke/diffusion_B_idloss/masks/generated_pose_pidm_occduke_wofail_n_2pose_npy_wofail_n_2pose_app_wofail_n_2pose',

        dataset_type='occluded_duke',
        
        image_size=(128, 64)
    )

    generator.generate_folder(

        # appearance image
        image_dir='XXX/Occluded_Duke/bounding_box_train_wofail_n_2pose',

        nomask_x=0.0,

        mask_x=1.5,  # 3.0

        # ddim / ddpm
        sample_algorithm='ddim',

        # DDIM steps
        nsteps=50,   # 50

        seed=1234,

        batch_size=128
    )