import torch
from torchvision.io import read_video, write_jpeg
import os
import imageio

from neural_style import ImageStyleTransfer

def combine_as_gif(base : str, ext, num_images, step, gifname):
    flow_dir = 'output_flows/'
    out_dir = 'output/'
    images = [imageio.imread(f'{flow_dir}{base}{i*step}.{ext}') for i in range(0, num_images)]
    imageio.mimsave(f'{out_dir}{gifname}', images)
    
if __name__ == "__main__":
    video_path = "input/pexelscom_pavel_danilyuk_basketball_hd.mp4"
    style_img = "input/rain-princess-aframov.jpg"
    input_frames, _, _ = read_video(video_path, output_format="TCHW")
    print(input_frames.device)
    step = 5
    for i in range(input_frames.shape[0]):
        if i%step == 0:
            write_jpeg(input_frames[i], f"output/frame_{i}.jpg")
    for i in range(input_frames.shape[0]):
        if i%step != 0:
            continue
        print("========iter: ", i//step, "============")
        torch.cuda.empty_cache()
        image_style_transfer = ImageStyleTransfer()
        image_style_transfer(f"output/frame_{i}.jpg", style_img, save_path=f"output/processed_frame_{i}.jpg", init_img=(f"output/frame_{i-step}.jpg" if i!=0 else None), num_steps=(500 if i==0 else 100))

    combine_as_gif('processed_frame_', 330//step + 1, step, 'princess_opt.gif')
    
