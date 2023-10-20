import torch
import torchvision.models  as models
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from torchvision.io import read_video, write_jpeg
import cv2
import numpy as np
from makegif import combine_as_gif
import argparse
from neural_style import ImageStyleTransfer, processor, extractor
from optical_flow import generate_optical_flow

class VideoStyleTransfer:
    def __init__(self) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device being used:", self.device)
        # pre_means = [0.485, 0.456, 0.406]
        self.pre_means = [0.48501961, 0.45795686, 0.40760392]
        self.pre_stds = [1, 1, 1]#[0.229, 0.224, 0.225]
        self.img_size = (512, 904)

        style_layer_nums = [1, 6, 11, 20, 29] # taken from https://www.mathworks.com/help/deeplearning/ref/vgg19.html
        content_layer_num = 22

        self.content_layers = {f"features.{content_layer_num}": "relu4_2"}
        self.style_layers = {f"features.{i}": f"relu{j+1}_1" for j, i in enumerate(style_layer_nums)}

        _style_wt_list = [1e3/n**2 for n in [64,128,256,512,512]]
        self.style_weights = {f"relu{j+1}_1":_style_wt_list[j] for j, _ in enumerate(style_layer_nums)}
        self.content_weights = {"relu4_2" : 1e0}
        self.stt_weight = 1e3 # short term temporal consistency weight

        self.model = self.get_model()
        self.content_img = None
        self.style_img = None

        self.proc = processor(self.img_size, self.pre_means, self.pre_stds)
        self.ext = extractor(self.model, self.style_layers, self.content_layers)

    def get_model(self):
        weights = models.VGG19_Weights.DEFAULT
        model = models.vgg19(weights=weights, progress=True)
        model = model.to(self.device).eval().requires_grad_(False)
        return model
    
    def get_images(self, content, style, prev_out):
        content_img = Image.open(content).convert("RGB")
        style_img = Image.open(style).convert("RGB")
        prev_out_img = Image.open(prev_out).convert("RGB")
        return style_img, content_img, prev_out_img

    # expects p_image and flow to be on the cpu
    # expects the flow from the destination image to the source image, i.e. the reverse flow
    def warp(self, p_image : torch.Tensor, flow : torch.Tensor):
        _, h, w = flow.shape
        flow_map = torch.zeros(flow.shape)
        flow_map[1] = flow[1] + torch.arange(flow.shape[1])[:, None]
        # print(torch.arange(flow.shape[1])[:, None].shape)
        # print(torch.arange(flow.shape[2])[None, :].shape)
        flow_map[0] = flow[0] + torch.arange(flow.shape[2])[None, :]
        dst = []
        for chan in range(p_image.shape[0]):
            temp = cv2.remap(p_image[chan].numpy(), flow_map[0].numpy(), flow_map[1].numpy(), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
            dst.append(torch.tensor(temp))
        return torch.stack(dst).cpu()
    
    # everything is on the gpu
    def get_mask_disoccluded(self, flow : torch.Tensor, rev_flow : torch.Tensor):
        w_bar = self.warp(flow.cpu(), rev_flow.cpu()).to(self.device)
        return (torch.square(w_bar + rev_flow).sum(dim=0) <= 0.01*(torch.square(flow).sum(dim=0) + torch.square(rev_flow).sum(dim=0)) + 0.5).to(self.device)

    def get_mask_edge(self, rev_flow : torch.Tensor):
        x_grad = torch.gradient(rev_flow[1])
        y_grad = torch.gradient(rev_flow[0])
        return torch.square(torch.stack(x_grad)).sum(dim=0) + torch.square(torch.stack(y_grad)).sum(dim=0) <= 0.01*torch.square(rev_flow).sum(dim=0) + 0.002

    def __call__(self, content : str, style : str, flow : torch.Tensor,  prev_out : str, rev_flow : torch.Tensor, save_path = None, num_steps = 500):
        # flow is the optical flow between the ith frame(this frame) and the (i-1)th frame, which is the entry at index (i-1) of the tensor returned from generate_optical_flow.
        # rev_flow is the reverse optical flow, i.e. the flow between the (i-1)th frame and the ith frame, which is the entry at index (i-1) of the tensor returned from generate_optical_flow(reverse=True).

        # ========= sanity checks ============
        assert flow.shape[0] == 2
        assert rev_flow.shape[0] == 2
        # ========= end of sanity checks ============

        self.style_img, self.content_img, self.prev_frame_img = self.get_images(content, style, prev_out)
        p_content, p_style, p_prev = self.proc.preprocess(self.content_img).to(self.device), self.proc.preprocess(self.style_img).to(self.device), self.proc.preprocess(self.prev_frame_img).cpu()

        actual_gram_matrices, _ = self.ext(p_style)
        _, actual_content_outputs = self.ext(p_content)

        prev_warped = self.warp(p_prev.squeeze(0), rev_flow).unsqueeze(0).to(self.device) # note that we are using the reverse flow because of the semantics of cv2.remap
        flow = flow.to(self.device)
        rev_flow = rev_flow.to(self.device)

        stt_mask = self.get_mask_disoccluded(flow, rev_flow) & self.get_mask_edge(rev_flow) # short term temporal consistency mask

        # ========= sanity checks ============
        print(prev_warped.shape, p_prev.shape)
        assert prev_warped.shape == p_prev.shape 
        # ========= end of sanity checks ============    

        noise_img = prev_warped.clone()
        noise_img.requires_grad = True
        num_iter = [0]
        iter_range = tqdm(range(num_steps))
        lr = 1
        optimizer = torch.optim.LBFGS([noise_img], max_iter=1, lr=lr)
        def closure():
            iter_range.update()
            style_outputs, content_outputs = self.ext(noise_img)
            loss = 0.
            num_iter[0] += 1
            for key, val in style_outputs.items():
                loss += self.style_weights[key] * nn.functional.mse_loss(style_outputs[key], actual_gram_matrices[key])
            for key, val in content_outputs.items():
                loss += self.content_weights[key]*nn.functional.mse_loss(val, actual_content_outputs[key])

            ## add the short term temporal consistency loss
            stt_loss = self.stt_weight * torch.mean(torch.square(noise_img - prev_warped)[:, :, stt_mask[:, :]])
            print("stt_loss =", stt_loss)
            # loss += stt_loss
  
            optimizer.zero_grad()
            loss.backward()
            return loss
        for _ in range(num_steps):
            optimizer.step(closure)

        corr_img = noise_img.clone()
        corr_img = self.proc.postprocess(corr_img)
        if save_path is not None:
            corr_img.save(save_path)
        return corr_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--load_cached_flows", action="store_true")
    args = parser.parse_args()
    video_name = "pexelscom_pavel_danilyuk_basketball_hd"
    video_ext = "mp4"
    style_name = "rain-princess-aframov"
    style_ext = "jpg"
    in_dir = 'input'
    mid_dir= "output_frames"
    out_dir = "output"
    step = 5
    img_size = (512, 904)
    
    if args.force:
        input_frames, _, _ = read_video(f'{in_dir}/{video_name}.{video_ext}', output_format="TCHW", pts_unit='sec')
        input_frames = transforms.Resize(img_size)(input_frames)

        for i in range(input_frames.shape[0]):
            if i%step == 0:
                write_jpeg(input_frames[i], f"{mid_dir}/{video_name[:6]}_frame_{i}.jpg")

        if not args.load_cached_flows:
            print("Computing forward optical flows...")
            optical_flow = generate_optical_flow(input_frames, reverse=True).cpu()
            print("Computing backward optical flows...")
            reverse_optical_flow = generate_optical_flow(input_frames, reverse=False).cpu()
            
            # store the optical flows
            torch.save(optical_flow, 'optical_flow.pt')
            torch.save(reverse_optical_flow, 'reverse_optical_flow.pt')
        
        else:
            optical_flow = torch.load('optical_flow.pt').cpu()
            reverse_optical_flow = torch.load('reverse_optical_flow.pt').cpu()

        for i in range(input_frames.shape[0]):
            if i % step != 0:
                continue
            print("========iter: ", i // step, "============")
            torch.cuda.empty_cache()
            if i == 0:
                if args.load_cached_flows:
                    continue
                image_style_transfer = ImageStyleTransfer()
                image_style_transfer.img_size = img_size
                image_style_transfer(
                    f"{mid_dir}/{video_name[:6]}_frame_{i}.jpg", 
                    f'{in_dir}/{style_name}.{style_ext}', 
                    save_path=f"{mid_dir}/{video_name[:6]}_processed_frame_{i}.jpg", 
                    init_img=None, 
                    num_steps=500
                )
            else:
                video_style_transfer = VideoStyleTransfer()
                video_style_transfer.img_size = img_size
                video_style_transfer(
                    f'{mid_dir}/{video_name[:6]}_frame_{i}.jpg', 
                    f'{in_dir}/{style_name}.{style_ext}', 
                    optical_flow[i-1],
                    f'{mid_dir}/{video_name[:6]}_processed_frame_{i-step}.jpg',
                    reverse_optical_flow[i-1],
                    save_path=f"{mid_dir}/{video_name[:6]}_processed_frame_{i}.jpg", 
                    num_steps=500
                )
            

    combine_as_gif(f'{video_name[:6]}_processed_frame_', 'jpg', mid_dir, out_dir, 330//step + 1, step, f'{video_name[:6]}_{style_name[:6]}.gif')
    
