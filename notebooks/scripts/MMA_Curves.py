#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2023/04/16 13:18:23
@Author  :   Xubo Luo 
@Version :   1.0
@Contact :   luoxubo@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   量化评估
'''

# here put the import lib
import numpy as np
import os
from immatch.utils.hpatches_helper import eval_summary_matching
import matplotlib
import matplotlib.pyplot as plt
# # https://www.somersault1824.com/wp-content/uploads/2015/02/color-blindness-palette-e1423327633855.png
# remove agg occupation to show image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 定义要测试的匹配方法
cbcolors={'cgreen':(26, 110, 53),
          'cgrass':(123, 252, 3),
          'cbrown':(161, 100, 56),
          'cyellow':(237, 187, 36),
          'cpurple':(177, 87, 250),
          'cpurblue' : (194, 207, 242),          
          'cgrey':(157, 163, 163),        
          'cblue':(66, 120, 245),
          'csky': (148, 250, 255),
          'corange':(245, 130, 69),
          'ccyan': (8, 189, 171),
          'crose' : (235, 101, 157),
          'cpink' : (255, 212, 212),
          'cred': (219, 15, 15),          
          'cdark': (0, 0, 0),
          'cake':(204,84,58),
          'cusuao':(145,180,147),
          'csuoh':(171,59,58),
         }
cbcolors={k:(v[0]/255,v[1]/255,v[2]/255) for k,v in cbcolors.items()}

# Methods correspond to the saved cache files
methods = ['hesaff', 
           'hesaffnet',
           'DoG1024-AffNet-HardNet.m0.95',
           'delf', 
           'D2Net', 'R2D2', 'aslfeat',
           'CAPS_SuperPoint_r4', 'CAPS_SIFT',  
           'SuperPoint_r4', 'SuperGlue_r4.m0.2', 'SuperGlue_r4.m0.9', 
           'SparseNCNet_N2000.im3200',
           'NCNet.im1024.m0.9',
           'Patch2Pix.im1024.m0.5',
           'Patch2Pix.im1024.m0.9',
           'DFM',
           'SURF1024',
           'SIFT1024',
          ]

# Legend names in the plot
names = ['HesAff+RootSIFT + NN', 
         'HAN+HN++ + NN',
         'DoGAffHardNet + SMNN',
         'DELF + NN', 
         'D2-Net + NN', 'R2D2 + NN', 'ASLFeat + NN',
         'Superpoint+CAPS + NN', 'Sift+CAPS + NN',
         'SuperPoint + NN', 
         'Superpoint + SuperGlue', 'Superpoint+SuperGlue c0.9',     
         'Sparse-NCNet im3200 top2k', 
         'NCNet_adapted c0.9',
         'Patch2Pix c0.5',
         'Patch2Pix c0.9',
         'DFM',
         'SURF1024',
         'SIFT1024'
        ]

colors = [cbcolors['cdark'], 
          cbcolors['cpurblue'], cbcolors['cpurblue'], 
          cbcolors['cgrey'],  
          cbcolors['cgrass'], cbcolors['ccyan'], cbcolors['cbrown'],  
          cbcolors['cyellow'], cbcolors['corange'],  
          cbcolors['cblue'],  cbcolors['cgreen'], cbcolors['csky'],  
          cbcolors['crose'],  
          cbcolors['cpink'],  
          cbcolors['cred'], cbcolors['cpurple'], 
          cbcolors['cake'], 
          cbcolors['cusuao'], cbcolors['csuoh'], ]

linestyles = ['-', 
              '-', '-.', 
              '--', 
              '-', '-', '-', 
              '-', '--', 
              '-', '-', '-', 
              '--', 
              '--',               
              '--', '--', 
              '-',
              '--',':',
             ]

print(len(methods), len(linestyles), len(colors), len(names))

# 加载计算好的结果
cache_dir = '../../outputs/hpatches/cache'
errors = {}
for method in methods:
    output_file = os.path.join(cache_dir, method + '.npy')
    print('\n>>>{}'.format(method))
    if not os.path.exists(output_file):
        print(f'Error: can not load precomputed results!!')
        break
        
    print('Loading precomputed errors...')
    errors[method] = np.load(output_file, allow_pickle=True)
    print(eval_summary_matching(errors[method]))


# 绘制MMA曲线


n_i = 52
n_v = 56
plt_lim = [1, 10]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)

plt.figure(figsize=(25, 10))

plt.subplot(1, 3, 1)
ymin = 0.0
 
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
plt.title('Overall')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylabel('MMA')
plt.ylim([ymin, 1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)

plt.subplot(1, 3, 2)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
plt.title('Illumination')
plt.xlabel('Threshold [px]')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([ymin, 1])
plt.gca().axes.set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)

plt.subplot(1, 3, 3)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    print (method)
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
plt.title('Viewpoint')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([ymin, 1])
plt.gca().axes.set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(fontsize=17, loc='center right', bbox_to_anchor=(0, 0, 1.75, 1))
plt.subplots_adjust(wspace=0.05)

# 量化结果曲线保存路径
plt.savefig('../../outputs/hpatches/hseq.png', bbox_inches='tight', dpi=300)
plt.show()
