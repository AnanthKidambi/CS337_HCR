import imageio

def combine_as_gif(base : str, ext, in_dir, out_dir, num_images, step, gifname):
    images = [imageio.imread(f'{in_dir}/{base}{i*step}.{ext}') for i in range(0, num_images)]
    imageio.mimsave(f'{out_dir}/{gifname}', images)