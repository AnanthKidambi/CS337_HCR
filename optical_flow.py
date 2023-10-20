import torch
from torchvision.io import read_video, write_jpeg
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms import functional
from torchvision.utils import flow_to_image
from torchvision.transforms import transforms
import os
from PIL import Image
import warnings
import imageio
from tqdm import tqdm
# warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combine_as_gif(base : str, ext, num_images, step, gifname):
    flow_dir = 'output_flows/'
    out_dir = 'output/'
    images = [imageio.imread(f'{flow_dir}{base}{i*step}.{ext}') for i in range(0, num_images)]
    imageio.mimsave(f'{out_dir}{gifname}', images)

#calculate optical flow of all frames in a video
def generate_optical_flow(video_path : str):
    input_frames, _, _ = read_video(video_path, output_format="TCHW")
    input_frames = input_frames.to(device)
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    img1_batch = input_frames[:-1]
    img2_batch = input_frames[1:]

    def preprocess(batch1, batch2):
        batch1 = functional.resize(batch1, size=((batch1.shape[2]//8)*8, (batch1.shape[3]//8)*8), antialias=False)
        batch2 = functional.resize(batch2, size=((batch2.shape[2]//8)*8, (batch2.shape[3]//8)*8), antialias=False)
        return transforms(batch1, batch2)
    
    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device)
    model = model.eval()

    list_of_flows = []
    with torch.no_grad():
        for i in tqdm(range(len(img1_batch))):
            list_of_flows.append(model(img1_batch[i].unsqueeze(0).to(device), img2_batch[i].unsqueeze(0).to(device))[-1].squeeze(0))
            
    return torch.stack(list_of_flows)

if __name__  == "__main__":
    video_name = "pexelscom_pavel_danilyuk_basketball_hd"
    video_ext = "mp4"
    in_dir = 'input/'
    optical_flow = generate_optical_flow(f'{in_dir}{video_name}.{video_ext}')
    for i in range(len(optical_flow)):
        flow_img = flow_to_image(optical_flow[i]).to("cpu")
        write_jpeg(flow_img, f"output_flows/{video_name}_flow_{i}.jpg")

    combine_as_gif(f"{video_name}_flow_", 'jpg', 331, 1, "bb_flow.gif")
    
    