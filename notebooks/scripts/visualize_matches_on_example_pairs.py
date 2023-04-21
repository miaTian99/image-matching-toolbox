#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   visualize.py
@Time    :   2023/04/16 12:52:37
@Author  :   Xubo Luo 
@Version :   1.0
@Contact :   luoxubo@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   可视化展示某一对图像的匹配结果
'''

# here put the import lib
import os
import yaml
import immatch
from immatch.utils import plot_matches

for method in ['patch2pix', 'loftr', 'superglue', 'caps_superpoint', 'd2net', 'r2d2', 'dogaffnethardnet','dfm']:
    print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Visualize matches of {method}')
    config_file = f'../../configs/{method}.yml'
    with open(config_file, 'r') as f:
        
        args = yaml.load(f, Loader=yaml.FullLoader)['example']
        if 'ckpt' in args:
            args['ckpt'] = os.path.join('../..', args['ckpt'])
        class_name = args['class']
        args['imsize'] = 512

    # Init model
    model = immatch.__dict__[class_name](args)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)
    
    # Put some example image pairs
    im1 = '../third_party/patch2pix/examples/images/pair_4/optical.jpg'
    im2 = '../third_party/patch2pix/examples/images/pair_4/infrared.jpg'

    matches, _, _, _ = matcher(im1, im2)    
    plot_matches(im1, im2, matches, radius=2, lines=True)

