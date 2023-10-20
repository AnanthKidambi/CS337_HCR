import torch
from torchvision.io import read_video, write_jpeg
from neural_style import ImageStyleTransfer
from makegif import combine_as_gif
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    video_name = "pexelscom_pavel_danilyuk_basketball_hd"
    video_ext = "mp4"
    style_name = "rain-princess-aframov"
    style_ext = "jpg"
    in_dir = 'input'
    mid_dir= "output_frames"
    out_dir = "output"
    step = 5
    
    if args.force:
        input_frames, _, _ = read_video(f'{in_dir}/{video_name}.{video_ext}', output_format="TCHW")
        
        for i in range(input_frames.shape[0]):
            if i%step == 0:
                write_jpeg(input_frames[i], f"{mid_dir}/{video_name[:6]}_frame_{i}.jpg")
        for i in range(input_frames.shape[0]):
            if i % step != 0:
                continue
            print("========iter: ", i // step, "============")
            torch.cuda.empty_cache()
            image_style_transfer = ImageStyleTransfer()
            image_style_transfer(f"{mid_dir}/{video_name[:6]}_frame_{i}.jpg", f'{in_dir}/{style_name}.{style_ext}', save_path=f"{mid_dir}/{video_name[:6]}_processed_frame_{i}.jpg", init_img=(f"{mid_dir}/{video_name[:6]}_frame_{i-step}.jpg" if i!=0 else None), num_steps=500)

    combine_as_gif(f'{video_name[:6]}_processed_frame_', 'jpg', mid_dir, out_dir, 330//step + 1, step, f'{video_name[:6]}_{style_name[:6]}.gif')
    
