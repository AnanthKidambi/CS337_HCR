import imageio
import numpy as np

global debug 
debug = False

global high_val
high_val = np.inf

global seed
seed = 69

global device
device = 'cuda:3'

def combine_as_gif(base : str, ext, in_dir, out_dir, num_images, step, gifname):
    images = [imageio.imread(f'{in_dir}/{base}{i}.{ext}') for i in range(num_images)]
    # print(len(images))
    imageio.mimsave(f'{out_dir}/{gifname}', images)

def make_gif(base, n_frames, gifname=None, step=1):
    mid_dir = 'output_frames'
    out_dir = 'output'
    if gifname is None:
        combine_as_gif(f'{base}_processed_frame_', 'jpg', mid_dir, out_dir, 1 + (n_frames - 1)//step, step, f'{base}.gif')
    else:
        combine_as_gif(f'{base}_processed_frame_', 'jpg', mid_dir, out_dir, 1 + (n_frames - 1)//step, step, gifname)