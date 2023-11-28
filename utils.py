import imageio
import numpy as np

global debug 
debug = False

global high_val
high_val = np.inf

global seed
seed = 42

global device
device = 'cuda:5'

global abbrev_to_full
abbrev_to_full = {
        'pexel': 'pexelscom_pavel_danilyuk_basketball_hd',
        'aframov': 'rain-princess-aframov',
        'vangogh': 'vangogh_starry_night',
        'oil': 'oil-crop',
        'dragon': 'dragon',
        'tom' : 'tom_and_jerry',
        'kandinsky' : 'kandinsky',
        'picasso' : 'picasso_selfport1907',
}

global full_to_abbrev
full_to_abbrev = {v:k for k,v in abbrev_to_full.items()}

def combine_as_gif(base : str, ext, in_dir, out_dir, num_images, step, gifname):
    images = [imageio.imread(f'{in_dir}/{base}{i}.{ext}') for i in range(0, num_images, step)]
    imageio.mimsave(f'{out_dir}/{gifname}', images)

def make_gif(base, n_frames, gifname=None, step=1):
    mid_dir = 'output_frames'
    out_dir = 'output'
    if gifname is None:
        combine_as_gif(f'{base}_processed_frame_', 'jpg', mid_dir, out_dir, 1 + (n_frames - 1)//step, step, f'{base}.gif')
    else:
        combine_as_gif(f'{base}_processed_frame_', 'jpg', mid_dir, out_dir, 1 + (n_frames - 1)//step, step, gifname)

def set_seed(seed: int):
    import torch, os, random
    torch.cuda.empty_cache()
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False