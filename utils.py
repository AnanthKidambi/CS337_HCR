import imageio
import numpy as np

global debug 
debug = False

global high_val
high_val = 1e9

global seed
seed = 69

def combine_as_gif(base : str, ext, in_dir, out_dir, num_images, step, gifname):
    images = [imageio.imread(f'{in_dir}/{base}{i}.{ext}') for i in range(num_images) for _ in range(5*step)]
    print(len(images))
    imageio.mimsave(f'{out_dir}/{gifname}', images)