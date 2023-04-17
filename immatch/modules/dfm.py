from argparse import Namespace
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys
dfm_path = Path(__file__).parent / '../../third_party/DFM'
sys.path.append(str(dfm_path))

from .base import Matching
from third_party.DFM.python.DeepFeatureMatcher import DeepFeatureMatcher
import immatch
from immatch.utils.data_io import read_im, resize_im

class DFM(Matching):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = Namespace(**args)
        if args.match_threshold == 0.0:
            args.match_threshold = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0]
        self.imsize = args.imsize
        # Load model
        self.model = DeepFeatureMatcher(enable_two_stage = args.enable_two_stage, model = args.pretrained_ckpt, 
                    ratio_th = args.match_threshold, bidirectional = args.bidirectional, )
        self.name = 'DFM'        
        print(f'Initialize {self.name}')

    def match_pairs(self, im1_path, im2_path):  
        im1, scale = read_im(im1_path, self.imsize)
        # 与其他模型同等比较
        # im2, scale = read_im(im2_path, self.imsize)
        im2, _ = read_im(im2_path)
        # DFM等比例缩放图像效果更好
        w_im2, h_im2, _ = resize_im(im2.width, im2.height, imsize = int(max(im2.width,im2.height)/scale[0]))
        im2 = im2.resize((w_im2, h_im2), Image.BICUBIC)
        img_A = np.array(im1)
        img_B = np.array(im2)
        
        # 修改DFM源代码，将score参数传出来
        _, _, points_A, points_B, scores = self.model.match(img_A, img_B)
        # 按比例缩放匹配点对
        kpts1 = (np.dot(points_A, scale[0])).T
        kpts2 = (np.dot(points_B, scale[0])).T
        # 原比例
        # kpts1 = points_A.T
        # kpts2 = points_B.T
        matches = np.concatenate([kpts1, kpts2], axis=1)
        # Fake scores as not output by the model(匹配数量)
        # scores = np.vstack([np.arange(0,kpts1.shape[0])]*2).T
        return matches, kpts1, kpts2, scores