import torch
from torchvision.io import read_video, write_jpeg
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms import functional
from torchvision.utils import flow_to_image
import warnings
from tqdm import tqdm
# warnings.filterwarnings("ignore")

from utils import combine_as_gif

device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using {device}, optical flows')

#calculate optical flow of all frames in a video
def generate_optical_flow(input_frames : torch.Tensor, reverse : bool = False):
    _input_frames = input_frames.to(device)
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    img1_batch = _input_frames[:-1] # frames corresponding to indices 0, 1, 2, .... n-1
    img2_batch = _input_frames[1:]  # frames corresponding to indices 1, 2, .... n
    def preprocess(batch1, batch2):
        batch1 = functional.resize(batch1, size=((batch1.shape[2]//8)*8, (batch1.shape[3]//8)*8), antialias=False)  # resize the images to nearest multiple of 8 for RAFT to work
        batch2 = functional.resize(batch2, size=((batch2.shape[2]//8)*8, (batch2.shape[3]//8)*8), antialias=False)
        return transforms(batch1, batch2)
    
    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device)
    model = model.eval()
    list_of_flows = []
    with torch.no_grad():
        for i in tqdm(range(len(img1_batch))):
            if reverse:
                list_of_flows.append(model(img2_batch[i].unsqueeze(0).to(device), img1_batch[i].unsqueeze(0).to(device))[-1].squeeze(0))
            else:
                list_of_flows.append(model(img1_batch[i].unsqueeze(0).to(device), img2_batch[i].unsqueeze(0).to(device))[-1].squeeze(0))
            
    return torch.stack(list_of_flows)

if __name__  == "__main__":
    video_name = "pexelscom_pavel_danilyuk_basketball_hd"
    video_ext = "mp4"
    in_dir = 'input'
    mid_dir = "output_flows"
    out_dir = "output"

    input_frames, _, _ = read_video(f'{in_dir}/{video_name}.{video_ext}', output_format="TCHW", pts_unit='sec')

    optical_flow = generate_optical_flow(input_frames, reverse=False)
    reverse_optical_flow = generate_optical_flow(input_frames, reverse=True)
    for i in range(len(optical_flow)):
        flow_img = flow_to_image(optical_flow[i]).to("cpu")
        write_jpeg(flow_img, f"{mid_dir}/{video_name[:6]}_flow_{i}.jpg")
        rev_flow_img = flow_to_image(reverse_optical_flow[i]).to("cpu")
        write_jpeg(rev_flow_img, f"{mid_dir}/{video_name[:6]}_rev_flow_{i}.jpg")
        
    combine_as_gif(f"{video_name[:6]}_flow_", 'jpg', mid_dir, out_dir, 331, 1, "bb_flow.gif")
    combine_as_gif(f"{video_name[:6]}_rev_flow_", 'jpg', mid_dir, out_dir, 331, 1, "bb_rev_flow.gif")
    
    