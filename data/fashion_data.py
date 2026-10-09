import os
import cv2 
import math
import lmdb
import numpy as np
from io import BytesIO
from PIL import Image
import random
import glob

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from data.fashion_base_function import get_random_params, get_transform

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

class Dataset(Dataset):
    def __init__(self, opt, is_inference, labels_required = False):  # PIDM-main/config/data.yaml  is_inference  labels_required==True
        self.opt = opt

        print('================== use Dataset ==================')
        self.root = opt.path
        self.semantic_path = self.root 

        self.labels_required = labels_required

        # self.env = lmdb.open(
        #     path,
        #     max_readers=32,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False,
        # )

        # if not self.env:
        #     raise IOError('Cannot open lmdb dataset', path)

        '''
        self.file_path = 'train_pairs.txt' if not is_inference else 'test_pairs.txt'
        '''
        self.file_path = opt.train_pairs if not is_inference else opt.test_pairs

        self.data = self.get_paths(self.root, self.file_path)

        # ==================================================
        # for sample Triplet Negative
        # ==================================================
        self.all_target_images = []
        self.id_to_images = {}
        for item in self.data:
            img_path = item['target_image']

            # 0001_c2_f0046182.jpg -> 0001
            person_id = os.path.basename(img_path).split('_')[0]

            self.all_target_images.append(img_path)

            if person_id not in self.id_to_images:
                self.id_to_images[person_id] = []

            self.id_to_images[person_id].append(img_path)
        '''
        self.id_to_images
        {
            '0001':[...],
            '0002':[...],
            '0003':[...]
        }
        '''

        self.all_ids = list(self.id_to_images.keys())
        
        self.is_inference = is_inference
        self.preprocess_mode = opt.preprocess_mode  # resize_and_crop  
        self.scale_param = opt.scale_param if not is_inference else 0 
        # 19 bone connections (only the first 17 are used during training)
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        

        self.occ_pairs = []
        for subdir in ["left-down", "left-right-down"]: 
            occ_dir = os.path.join(
                self.root,
                self.opt.occ_object_path,
                subdir
            )
            mask_dir = os.path.join(
                self.root,
                self.opt.occ_mask_path,
                subdir
            )
            for fname in os.listdir(occ_dir):
                if not fname.lower().endswith(
                    (".jpg", ".png", ".jpeg")
                ):
                    continue
                occ_path = os.path.join(
                    occ_dir,
                    fname
                )
                mask_path = os.path.join(
                    mask_dir,
                    os.path.splitext(fname)[0] + ".npy"
                )
                self.occ_pairs.append({
                    "occ": occ_path,
                    "mask": mask_path,
                    "type": subdir
                })

        print('occlusion mode: ', self.opt.occ_mode)

    def sample_negative_image(self, exclude_id):

        candidate_ids = [
            pid for pid in self.all_ids
            if pid != exclude_id
        ]

        neg_id = random.choice(candidate_ids)

        neg_img = random.choice(
            self.id_to_images[neg_id]
        )

        return neg_img, neg_id


    def load_occ_pair(self):

        item = random.choice(self.occ_pairs)

        occ_img = Image.open(
            item["occ"]
        ).convert("RGB")

        occ_mask = np.load(
            item["mask"]
        ).astype(np.uint8)

        return occ_img, occ_mask, item["type"]

    def __len__(self):
        return len(self.data)


    def get_paths(self, root, path):
        fd = open(os.path.join(root, path))
        lines = fd.readlines()
        fd.close()

        image_paths = []
        for item in lines:
            dict_item={}
            item = item.strip().split(',')  # target, source1, source2, ...
            # source
            # dict_item['source_image'] = [path.replace('.jpg', '.png') for path in item[1:]]
            dict_item['source_image'] = item[1:]  # jpg
            dict_item['source_label'] = [os.path.join(self.semantic_path, self.img_to_label(path)) for path in dict_item['source_image']]
            # target
            # dict_item['target_image'] = item[0].replace('.jpg', '.png')
            dict_item['target_image'] = item[0]  # jpg
            # target pose（img/xxx.png → pose/xxx.txt）
            dict_item['target_label'] = os.path.join(self.semantic_path, self.img_to_label(dict_item['target_image']))
            image_paths.append(dict_item)
        return image_paths


    def open_lmdb(self):
        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(buffers=True)

    
    def __getitem__(self, index):

        mode = self.opt.occ_mode

        if mode == "modeX":
            target_use_mask = False
            source_use_mask = False

        elif mode == "modeA":
            # 1:0:1:0
            target_use_mask = random.random() < 0.5
            source_use_mask = False

        elif mode == "modeB":
            # 1:0:1:1
            r = random.random()
            if r < 1/3:
                target_use_mask = False
                source_use_mask = False       
            elif r < 2/3:
                target_use_mask = True
                source_use_mask = False       
            else:
                target_use_mask = True
                source_use_mask = True        

        elif mode == "modeC":
            # 1:1:1:0
            r = random.random()
            if r < 1/3:
                target_use_mask = False
                source_use_mask = False       
            elif r < 2/3:
                target_use_mask = False
                source_use_mask = True        
            else:
                target_use_mask = True
                source_use_mask = False       

        elif mode == "modeD":
            # 1:1:1:1
            r = random.random()
            if r < 0.25:
                target_use_mask = False
                source_use_mask = False
            elif r < 0.5:
                target_use_mask = False
                source_use_mask = True
            elif r < 0.75:
                target_use_mask = True
                source_use_mask = False
            else:
                target_use_mask = True
                source_use_mask = True

        path_item = self.data[index] 
        '''
        {
          'target_image': 'A.jpg',
          'target_label': 'pose/A.txt',
          'source_image': ['B.jpg', 'C.jpg'],
          'source_label': ['pose/B.txt', 'pose/C.txt']
        }
        '''

        # ID
        current_id = os.path.basename(path_item['target_image']).split('_')[0]
        negative_target_path, negative_target_id = self.sample_negative_image(current_id)
        negative_source_path, negative_source_id = self.sample_negative_image(current_id)
        negative_target_tensor, _, _, _ = self.get_image_tensor(negative_target_path)
        negative_source_tensor, _, _, _ = self.get_image_tensor(negative_source_path)


        # target images 
        if target_use_mask:
            occ_img, occ_mask, occ_type = self.load_occ_pair()
            target_image_tensor, param, original_size, mask = self.get_image_tensor(path_item['target_image'], occ_img, occ_mask, occ_type)
            
            if self.labels_required:
                target_label_tensor, target_face_center = self.get_label_tensor(path_item['target_label'], target_image_tensor, param, original_size, mask)
                '''
                target_label_tensor: [C_pose, H, W]  C_pose == 20
                target_face_center: [4]
                '''
                target_mask = mask.unsqueeze(0)

        else:
            target_image_tensor, param, original_size, mask = self.get_image_tensor(path_item['target_image'])
            if self.labels_required:
                target_label_tensor, target_face_center = self.get_label_tensor(path_item['target_label'], target_image_tensor, param, original_size, mask)
                '''
                target_label_tensor: [C_pose, H, W]  为 RGB skeleton + distance maps
                target_face_center: [4]
                '''
                target_mask = mask.unsqueeze(0)
            
        # source images 
        i = np.random.choice(list(range(0, len(path_item['source_image']))))
        source_image_path = path_item['source_image'][i]
        source_label_path = path_item['source_label'][i]

        if source_use_mask:
            occ_img, occ_mask, occ_type = self.load_occ_pair()
            ref_tensor, param, original_size_ref, mask = self.get_image_tensor(
                source_image_path,
                occ_img,
                occ_mask,
                occ_type,
            )
        else:
            ref_tensor, param, original_size_ref, mask = self.get_image_tensor(
                source_image_path
            )
        '''
        ref_tensor: [3, H, W]
        param: crop/resize
        '''

        if self.labels_required:
            label_ref_tensor, ref_face_center = self.get_label_tensor(source_label_path, ref_tensor, param, original_size_ref, mask)
            '''
            label_ref_tensor: [C_pose, H, W]
            '''
            source_mask = mask.unsqueeze(0)
        
        image_path = self.get_image_path(source_image_path, path_item['target_image'])

        if not self.is_inference:
            if torch.rand(1) < 0.5:
                # 随机水平翻转 source image、target image、source pose、target pose
                target_image_tensor = F.hflip(target_image_tensor)
                ref_tensor = F.hflip(ref_tensor)
                if self.labels_required:
                    target_label_tensor = F.hflip(target_label_tensor)
                    label_ref_tensor = F.hflip(label_ref_tensor)
                    # mask 也要进行水平翻转
                    source_mask = F.hflip(source_mask)
                    target_mask = F.hflip(target_mask)

        if self.opt.if_concat_mask == True:
            # print(source_mask.shape)
            target_label_tensor = torch.concat([target_label_tensor, target_mask], dim=0)
            label_ref_tensor = torch.concat([label_ref_tensor, source_mask], dim=0)

        if self.labels_required:
            input_dict = { 
                        'target_image': target_image_tensor,  # [3, H, W]
                        'target_skeleton': target_label_tensor,  # [C_pose, H, W]  C_pose = 3（RGB骨架图） + 17（distance maps） = 20  # if_concat_mask == True 时 为 21，因为concat了mask
                        'target_face_center': target_face_center,  # [4]

                        'source_image': ref_tensor,  # [3, H, W]
                        'source_skeleton': label_ref_tensor,  # [C_pose, H, W]
                        'source_face_center': ref_face_center,  # [4]

                        'path': image_path,

                        'negative_target_image': negative_target_tensor,
                        'negative_source_image': negative_source_tensor,
                        }
        else:
            input_dict = {'target_image': target_image_tensor,
                          'source_image': ref_tensor,
                         }

        return input_dict

    '''
    def get_image_path(self, source_name, target_name):
        source_name = self.path_to_fashion_name(source_name)
        target_name = self.path_to_fashion_name(target_name)
        image_path = os.path.splitext(source_name)[0] + '_2_' + os.path.splitext(target_name)[0]+'_vis.png'
        return image_path
    '''
    def get_image_path(self, source_name, target_name):
        source_base = os.path.splitext(os.path.basename(source_name))[0]
        target_base = os.path.splitext(os.path.basename(target_name))[0]
        return f"{source_base}_2_{target_base}_vis.png"

    def path_to_fashion_name(self, path_in):
        path_in = path_in.split('img/')[-1]
        path_in = os.path.join('fashion', path_in)
        path_names = path_in.split('/') 
        path_names[3] = path_names[3].replace('_', '')
        path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
        path_names = "".join(path_names)
        return path_names

    def img_to_label(self, path): 
        # return path.replace('img/', 'pose/').replace('.png', '.txt')
        return path.replace('img/', 'pose/').replace('.jpg', '.txt')

    '''
    def get_image_tensor(self, path):
        with self.env.begin(write=False) as txn:
            key = f'{path}'.encode('utf-8')
            img_bytes = txn.get(key) 
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        
        param = get_random_params(img.size, self.scale_param)  # self.scale_param = 0.05   {'crop_param': (x, y, w, h), 'scale_size':(new_h, new_w)}  
        # transform（归一化 + tensor）
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param
    '''
    def get_image_tensor(self, path, occ=None, mask=None, occ_type=None):

        img_path = os.path.join(self.root, path)

        img = Image.open(img_path).convert("RGB")

        original_size = img.size

        base_h, base_w = self.opt.load_size  # 128 64
        img = img.resize((base_w, base_h), Image.BICUBIC)  # PIL  w, h 

        if occ is not None:
            # resize occ
            occ = occ.resize((base_w, base_h), Image.BICUBIC)
            # resize mask
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img = mask_img.resize((base_w, base_h), Image.NEAREST)

            img_np = np.array(img)
            occ_np = np.array(occ)
            mask_np = np.array(mask_img)

            h, w = mask_np.shape
            if occ_type == 'left-down':
                dx = -random.randint(0, int(w * 0.1))
                dy = random.randint(0, int(h * 0.1))
            if occ_type == 'left-right-down':
                dx = random.randint(-int(w * 0.1), int(w * 0.1))
                dy = random.randint(0, int(h * 0.1))
            M = np.float32([ [1,0,dx], [0,1,dy] ]) 
            occ_np = cv2.warpAffine(occ_np, M, (w,h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
            mask_np = cv2.warpAffine(mask_np, M, (w,h), flags=cv2.INTER_NEAREST, borderValue=0) 

            img_np[mask_np == 1] = occ_np[mask_np == 1]

            img = Image.fromarray(img_np)

        else:
            mask_np = np.zeros((base_h, base_w), dtype=np.uint8)

        '''
        param = get_random_params(img.size, self.scale_param)
        '''
        param = get_random_params(img.size, self.scale_param, load_size=self.opt.load_size)
        # transform
        img_trans = get_transform(param, normalize=True, toTensor=True)
        img = img_trans(img)
        mask_trans = get_transform(param, normalize=False, toTensor=False, is_mask=True)
        mask_occ = Image.fromarray(mask_np)
        mask_occ = mask_trans(mask_occ)
        mask_occ = np.array(mask_occ).astype(np.uint8)

        assert np.all(np.isin(mask_occ, [0, 1]))
        # print(mask_occ.shape)

        mask_tensor = torch.from_numpy(mask_occ).float()
        
        return img, param, original_size, mask_tensor

    def get_label_tensor(self, path, img, param, original_size, mask): 
        '''
        path: 'pose/A.txt' 
        output:
        label_tensor = [20, H, W] 
        face_center = [4]
        '''
        canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)

        keypoint = np.loadtxt(path)   # [18, 2]

        keypoint = self.trans_keypoins(keypoint, param, img.shape[1:], original_size)  # ！！！

        mask_np = mask.numpy()
        for i in range(18):
            x = int(keypoint[i,0])
            y = int(keypoint[i,1])
            if x == -1 or y == -1:
                continue
            if (
                x >= 0 and
                x < mask_np.shape[1] and
                y >= 0 and
                y < mask_np.shape[0]
            ):
                if mask_np[y,x] > 0:
                    keypoint[i] = [-1,-1]


        for i in range(18):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
        joints = []
        
        stickwidth = 4  

        for i in range(17):

            Y = keypoint[np.array(self.limbSeq[i])-1, 0]
            X = keypoint[np.array(self.limbSeq[i])-1, 1]            
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)

            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])

            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)  # RGB skeleton

            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)

        # pose RGB tensor
        pose = F.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

        tensors_dist = 0
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3) 
            im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8) 
            tensor_dist = F.to_tensor(Image.fromarray(im_dist)) 
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])  # [17, H, W]
            e += 1
        
        label_tensor = torch.cat((pose, tensors_dist), dim=0)   # [20, H, W]


        if int(keypoint[14, 0]) != -1 and int(keypoint[15, 0]) != -1:
            y0, x0 = keypoint[14, 0:2]
            y1, x1 = keypoint[15, 0:2]
            face_center = torch.tensor([y0, x0, y1, x1]).float()
        else:
            face_center = torch.tensor([-1, -1, -1, -1]).float()               
        return label_tensor, face_center


    def trans_keypoins(self, keypoints, param, img_size, original_size):
        '''
        param:
            'crop_param': (x, y, w, h), 
            'scale_size': (new_h, new_w),
            'base_size': (h, w) 
        img_size: ref_imagec size
        original_size: w h
        '''

        missing_keypoint_index = keypoints == -1

        img_h, img_w = img_size  # 128 64 

        orig_w, orig_h = original_size

        base_h, base_w = param['base_size']  # 128 64

        # =========================
        # -> base_size（128 64）
        # =========================
        scale_w = base_w / orig_w
        scale_h = base_h / orig_h

        if 'scale_size' in param and param['scale_size'] is not None:  # this
            # print('================== scale_size ==================')
            new_h, new_w = param['scale_size']
            # print(f'New size: {new_w} x {new_h}')
            scale_w = scale_w / img_w * new_w
            scale_h = scale_h / img_h * new_h  

        if 'crop_param' in param and param['crop_param'] is not None:  # this
            w, h, _, _ = param['crop_param'] 
            # print('================== crop_param ==================')
            # print(f'Crop param: {w}, {h}')
        else:
            w, h = 0, 0

        keypoints[:,0] = np.clip(keypoints[:,0] * scale_w - w, 0, img_w - 1)
        keypoints[:,1] = np.clip(keypoints[:,1] * scale_h - h, 0, img_h - 1)

        keypoints[missing_keypoint_index] = -1
        return keypoints



class Dataset_guide(Dataset):   # 即 Disentangled classifier-free guidance，仅 sample 过程中使用 —— 并非，classifier-free guidance的概率由 diffusion.training_losses的prob参数控制
    def __init__(self, opt, is_inference, labels_required = False):
        self.opt = opt

        print('================== use Dataset_guide ==================')
        self.root = opt.path 
        self.semantic_path = self.root 

        '''
        path = os.path.join(self.root, str(opt.sub_path))
        self.path = path
        '''

        self.labels_required = labels_required
        

        # self.env = lmdb.open(
        #     path,
        #     max_readers=32,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False,
        # )

        # if not self.env:
        #     raise IOError('Cannot open lmdb dataset', path)

        '''
        self.file_path = 'train_pairs.txt' if not is_inference else 'test_pairs.txt'
        '''
        self.file_path = opt.train_pairs if not is_inference else opt.test_pairs

        self.data = self.get_paths(self.root, self.file_path)

        self.len_ = len(self.data) 
        
        self.is_inference = is_inference
        self.preprocess_mode = opt.preprocess_mode  # resize_and_crop
        self.scale_param = opt.scale_param if not is_inference else 0 

        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def __len__(self):
        return len(self.data)

    def get_paths(self, root, path):
        fd = open(os.path.join(root, path))
        lines = fd.readlines()
        fd.close()

        image_paths = []
        for item in lines:
            dict_item={}
            item = item.strip().split(',')  # target, source1, source2, ...
            # dict_item['source_image'] = [path.replace('.jpg', '.png') for path in item[1:]]
            dict_item['source_image'] = item[1:]
            dict_item['source_label'] = [os.path.join(self.semantic_path, self.img_to_label(path)) for path in dict_item['source_image']]
            # dict_item['target_image'] = item[0].replace('.jpg', '.png')
            dict_item['target_image'] = item[0]
            # target pose（img/xxx.jpg → pose/xxx.txt）
            dict_item['target_label'] = os.path.join(self.semantic_path, self.img_to_label(dict_item['target_image']))
            image_paths.append(dict_item)
        return image_paths


    def open_lmdb(self):

        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(buffers=True)


    def __getitem__(self, index):
        '''
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        '''

        path_item = self.data[index]
        '''
        {
          'target_image': 'A.jpg',
          'target_label': 'pose/A.txt',
          'source_image': ['B.jpg', 'C.jpg'],
          'source_label': ['pose/B.txt', 'pose/C.txt']
        }
        '''

        guide_labels = (torch.rand(2) < 0.5)

        # =========================
        # choose reference
        # =========================
        if guide_labels[0]:
            path_item_ref = path_item
        else:
            path_item_ref = self.data[np.random.choice(list(set(list(np.arange(self.len_)))-{index}))]

        if guide_labels[1]:
            path_item_pose = path_item
        else:
            path_item_pose = self.data[np.random.choice(list(set(list(np.arange(self.len_)))-{index}))]     

        target_image_path = path_item['target_image']
        '''
        target_image_tensor, param = self.get_image_tensor(path_item['target_image'])
        '''
        target_image_tensor, param, original_size = self.get_image_tensor(path_item['target_image'])

        pose_label_path = path_item_pose['target_label']

        target_label_tensor, target_face_center = self.get_label_tensor(pose_label_path, target_image_tensor, param)
        '''
        target_label_tensor: [C_pose, H, W]
        target_face_center: [4]
        '''

        i = np.random.choice(list(range(0, len(path_item_ref['source_image']))))
        source_image_path = path_item_ref['source_image'][i]

        ref_image_tensor, param, original_size = self.get_image_tensor(source_image_path)


        if torch.rand(1) < 0.5:
            ref_image_tensor = F.hflip(ref_image_tensor)
            target_image_tensor = F.hflip(target_image_tensor)
            target_label_tensor = F.hflip(target_label_tensor)


        return {'pose_skeleton': target_label_tensor,  #  [C_pose, H, W] 'target_skeleton'
                'target_image': target_image_tensor,   # [3, H, W]
                'ref_image_tensor': ref_image_tensor,  #  [3, H, W] 'source_image' xs
                'guide_label' : bin2dec(guide_labels, 2)  # （00/01/10/11）
                }

    '''
    def get_image_path(self, source_name, target_name): 
        source_name = self.path_to_fashion_name(source_name)
        target_name = self.path_to_fashion_name(target_name)
        image_path = os.path.splitext(source_name)[0] + '_2_' + os.path.splitext(target_name)[0]+'_vis.png'
        return image_path
    '''
    def get_image_path(self, source_name, target_name):
        source_base = os.path.splitext(os.path.basename(source_name))[0]
        target_base = os.path.splitext(os.path.basename(target_name))[0]
        return f"{source_base}_2_{target_base}_vis.png"

    def path_to_fashion_name(self, path_in): 
        path_in = path_in.split('img/')[-1]
        path_in = os.path.join('fashion', path_in)
        path_names = path_in.split('/') 
        path_names[3] = path_names[3].replace('_', '')
        path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
        path_names = "".join(path_names)
        return path_names

    def img_to_label(self, path): 
        # return path.replace('img/', 'pose/').replace('.png', '.txt')
        return path.replace('img/', 'pose/').replace('.jpg', '.txt')

    '''
    def get_image_tensor(self, path):
        with self.env.begin(write=False) as txn:
            key = f'{path}'.encode('utf-8')
            img_bytes = txn.get(key) 
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        param = get_random_params(img.size, self.scale_param)  # self.scale_param = 0.05   {'crop_param': (x, y, w, h), 'scale_size':(new_h, new_w)}  
        # transform（归一化 + tensor）
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param
    '''
    def get_image_tensor(self, path): 
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        '''
        param = get_random_params(img.size, self.scale_param)
        '''
        param = get_random_params(img.size, self.scale_param, load_size=self.opt.load_size)
        # transform
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param, original_size

    def get_label_tensor(self, path, img, param):

        canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)

        keypoint = np.loadtxt(path)   # [18, 2]   not exist: (-1, -1)

        keypoint = self.trans_keypoins(keypoint, param, img.shape[1:], original_size)

        for i in range(18):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
        joints = []

        stickwidth = 4 

        for i in range(17):
            Y = keypoint[np.array(self.limbSeq[i])-1, 0]
            X = keypoint[np.array(self.limbSeq[i])-1, 1]            
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)
        pose = F.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

        tensors_dist = 0
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
            im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
            tensor_dist = F.to_tensor(Image.fromarray(im_dist))
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
            e += 1

        label_tensor = torch.cat((pose, tensors_dist), dim=0)
        if int(keypoint[14, 0]) != -1 and int(keypoint[15, 0]) != -1:
            y0, x0 = keypoint[14, 0:2]
            y1, x1 = keypoint[15, 0:2]
            face_center = torch.tensor([y0, x0, y1, x1]).float()
        else:
            face_center = torch.tensor([-1, -1, -1, -1]).float()               
        return label_tensor, face_center

def trans_keypoins(self, keypoints, param, img_size, original_size):

        missing_keypoint_index = keypoints == -1
        
        # crop the white line in the original dataset
        keypoints[:,0] = (keypoints[:,0]-40)

        # resize the dataset
        img_h, img_w = img_size
        scale_w = 1.0/176.0 * img_w
        scale_h = 1.0/256.0 * img_h

        if 'scale_size' in param and param['scale_size'] is not None:
            new_h, new_w = param['scale_size']
            scale_w = scale_w / img_w * new_w
            scale_h = scale_h / img_h * new_h
        

        if 'crop_param' in param and param['crop_param'] is not None:
            w, h, _, _ = param['crop_param']
        else:
            w, h = 0, 0

        keypoints[:,0] = keypoints[:,0]*scale_w - w
        keypoints[:,1] = keypoints[:,1]*scale_h - h
        keypoints[missing_keypoint_index] = -1
        return keypoints
