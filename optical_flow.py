import torch
from torchvision.io import read_video, write_jpeg
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms import functional
from torchvision.utils import flow_to_image
from torchvision.transforms import transforms
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#calculate optical flow of all frames in a video
def generate_optical_flow(video_path : str):
    input_frames, _, _ = read_video(video_path, output_format="TCHW")
    input_frames = input_frames.to(device)
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    img1_batch = input_frames[:-1]
    img2_batch = input_frames[1:]
    def prerpocess(batch1, batch2):
        batch1 = functional.resize(batch1, size=((batch1.shape[2]//8)*8, (batch1.shape[3]//8)*8), antialias=False)
        batch2 = functional.resize(batch2, size=((batch2.shape[2]//8)*8, (batch2.shape[3]//8)*8), antialias=False)
        return transforms(batch1, batch2)
    img1_batch, img2_batch = prerpocess(img1_batch, img2_batch)

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device)
    model = model.eval().requires_grad_(False)
    list_of_flows = model(img1_batch, img2_batch)
    return list_of_flows[-1]

if __name__  == "__main__":
    video_file = "input/pexelscom_pavel_danilyuk_basketball_hd.mp4"
    optical_flow = generate_optical_flow(video_file)
    flow_frames = flow_to_image(optical_flow).to("cpu")
    for i in range(flow_frames.shape[0]):
        # transforms.ToPILImage()(flow_frames[i]).save(f"output/flow_{i}.jpg")
        write_jpeg(flow_frames[i], f"output/flow_{i}.jpg")
    os.system("wsl -e ffmpeg -f image2 -framerate 30 -i output/flow_%d.jpg -loop -1 output/flow.gif")