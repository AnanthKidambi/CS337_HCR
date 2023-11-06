import imageio

global debug 
debug = False

def combine_as_gif(base : str, ext, in_dir, out_dir, num_images, step, gifname):
    images = [imageio.imread(f'{in_dir}/{base}{i}.{ext}') for i in range(num_images) for _ in range(5*step)]
    print(len(images))
    imageio.mimsave(f'{out_dir}/{gifname}', images)