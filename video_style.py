import torch
from torchvision.io import read_video, write_jpeg
import os

from neural_style import ImageStyleTransfer

if __name__ == "__main__":
    video_path = "input/dragon.gif"
    style_img = "input/vangogh_starry_night.jpg"
    input_frames, _, _ = read_video(video_path, output_format="TCHW")
    print(input_frames.device)
    for i in range(input_frames.shape[0]):
        write_jpeg(input_frames[i], f"output/frame_{i}.jpg")
    for i in range(input_frames.shape[0]):
        print("========iter: ", i, "============")
        torch.cuda.empty_cache()
        image_style_transfer = ImageStyleTransfer()
        image_style_transfer(f"output/frame_{i}.jpg", style_img, save_path=f"output/processed_frame_{i}.jpg")
        # import time
        # time.sleep(1)
        # os.system("python neural_style.py")
    os.system("wsl -e ffmpeg -f image2 -framerate 30 -i output/processed_frame_%d.jpg -loop -1 output/final_vid.gif")
