import torch
import torchvision.models  as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from torchvision.io import read_video, write_jpeg
import cv2
import numpy as np
import random
from makegif import combine_as_gif
import argparse
from neural_style import ImageStyleTransfer, processor, extractor
from optical_flow import generate_optical_flow
import os
import shutil
# torch.set_printoptions(threshold=99999)

abbrev_to_full = {
        'pexel': 'pexelscom_pavel_danilyuk_basketball_hd',
        'aframov': 'rain-princess-aframov',
        'vangogh': 'vangogh-starry-night',
        'oil': 'oil-crop',
        'dragon': 'dragon',
}

full_to_abbrev = {v:k for k,v in abbrev_to_full.items()}

class VideoStyleTransfer:
    def __init__(self, img_size) -> None:
        self.device = 'cpu'
        #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(torch.cuda.is_available())
        # print(torch.cuda.device_count())
        print("Device being used:", self.device)
        # pre_means = [0.485, 0.456, 0.406]
        self.pre_means = [0.48501961, 0.45795686, 0.40760392]
        self.pre_stds = [1, 1, 1] #[0.229, 0.224, 0.225]
        self.img_size = img_size

        style_layer_nums = [1, 6, 11, 20, 29] # taken from https://www.mathworks.com/help/deeplearning/ref/vgg19.html
        content_layer_num = 22

        self.content_layers = {f"features.{content_layer_num}": "relu4_2"}
        self.style_layers = {f"features.{i}": f"relu{j+1}_1" for j, i in enumerate(style_layer_nums)} # hardcoded layer names

        # _style_wt_list = [1e3/n**2 for n in [64,128,256,512,512]]
        _style_wt_list = [0.2]*5
        self.style_weights = {val:_style_wt_list[j] for j, val in enumerate(self.style_layers.values())}
        _content_wt_list = [1.0]
        self.content_weights = {val:_content_wt_list[j] for j, val in enumerate(self.content_layers.values())}
        self.stt_weight = 2e2 # short term temporal consistency weight

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

    def warp(self, p_image : torch.Tensor, flow : torch.Tensor):
        '''
            :param p_image: The image to warp. Shape should be (C, H, W) and device should be CPU
            :param flow: The *inverse* function of the flow that is supposed to be warped over the image. Shape should be (2, H, W) and device should be CPU

            Outputs the warped image with shape (C, H, W) on CPU
        '''
        assert p_image.device == torch.device("cpu"), 'need cpu in warp'
        assert flow.device == torch.device("cpu"), 'need cpu in warp'
        
        flow_map = flow.clone()
        flow_map[1] += torch.arange(flow.shape[1])[:, None]
        flow_map[0] += torch.arange(flow.shape[2])[None, :]
        dst = []

        for chan in range(p_image.shape[0]):
            temp = cv2.remap(p_image[chan].numpy(), flow_map[0].numpy(), flow_map[1].numpy(), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
            dst.append(torch.tensor(temp))
        return torch.stack(dst).cpu()
    
    def get_mask_disoccluded(self, flow : torch.Tensor, rev_flow : torch.Tensor):
        '''
            :param flow: Forward flow from frame[i-1]. Shape should be (2, H, W) and device should be CPU
            :param rev_flow: Reverse flow from frame[i] to frame[i-1]. Shape should be (2, H, W) and device should be CPU

            Outputs the disoccluded region mask with shape (H, W) on CPU
        '''
        assert flow.device == torch.device("cpu"), 'need cpu in warp (get_mask_disoccluded)'
        assert rev_flow.device == torch.device("cpu"), 'need cpu in warp (get_mask_disoccluded)'

        w_tilde = self.warp(flow, rev_flow)
        return (self.squared_norm(w_tilde + rev_flow) <= 
                (0.01*(self.squared_norm(w_tilde) + self.squared_norm(rev_flow)) 
                 + 0.5))

    def get_mask_edge(self, rev_flow : torch.Tensor):
        '''
            :param rev_flow: Reverse flow from frame[i] to frame[i-1]. Shape should be (2, H, W) and device should be CPU

            Outputs the disoccluded region mask with shape (H, W) on CPU
        '''
        # no need to be in cpu?
        # assert rev_flow.device == torch.device("cpu")

        x_grad = torch.gradient(rev_flow[1])
        y_grad = torch.gradient(rev_flow[0])
        return (self.squared_norm(torch.stack(x_grad)) + self.squared_norm(torch.stack(y_grad)) <= (0.01*self.squared_norm(rev_flow) + 0.002))

    def __call__(self, content : str, style : str, flow : torch.Tensor,  prev_out : str, rev_flow : torch.Tensor, save_path = None, num_steps = 500):
        '''
            :param content: Path to content image. Will be converted to (1, C, H, W)
            :param style: Path to style image. Will be converted to (1, C, H, W)
            :param flow: The forward optical flow. For frame[i], this is the flow form frame[i-1] to frame[i]. Shape should be (2, H, W)
            :param prev_out: Path to the most recently computed stylized frame. Will be converted to (1, C, H, W)
            :param rev_flow: The reverse optical flow. For frame[i], this is the reverse flow from frame[i] to frame[i-1]. Shape should be (2, H, W)
            :param num_steps: Number of iterations of the optimizer. Default is 500
        '''
        # ========= sanity checks ============
        assert flow.shape[0] == 2, f"Shape is {flow.shape[0]}"
        assert rev_flow.shape[0] == 2, f"Shape is {rev_flow.shape[0]}"
        print(flow.device, rev_flow.device)
        # assert flow.device == 'cpu', 'flow needs to be on cpu'
        # assert rev_flow.device == 'cpu', 'rev_flow needs to be on cpu'
        # ========= end of sanity checks ============

        self.style_img, self.content_img, self.prev_frame_img = self.get_images(content, style, prev_out)

        p_content = self.proc.preprocess(self.content_img).to(self.device)
        p_style = self.proc.preprocess(self.style_img).to(self.device)
        p_prev_frame = self.proc.preprocess(self.prev_frame_img).cpu()
        print('check-input', p_style.isnan().nonzero(), p_content.isnan().nonzero(), p_prev_frame.isnan().nonzero())
        p_style[p_style.isnan()] = 0.
        p_content[p_content.isnan()] = 0.
        p_prev_frame[p_prev_frame.isnan()] = 0.

        # compute gram matrices and feature maps to plug into the loss
        actual_gram_matrices, _ = self.ext(p_style)
        _, actual_content_outputs = self.ext(p_content)

        ##### CPU computation #############

        prev_warped = self.warp(p_prev_frame.squeeze(0), rev_flow).unsqueeze(0) # note that we are using the reverse flow because of the semantics of cv2.remap
        print('checking...', prev_warped.isnan().nonzero())
        prev_warped[prev_warped.isnan()] = p_style[prev_warped.isnan()]
        print('oh hey, is this true:', prev_warped.isinf().nonzero())
        prev_warped[prev_warped.isinf()] = 1<<16

        disocc_mask = self.get_mask_disoccluded(flow, rev_flow)
        edge_mask = self.get_mask_edge(rev_flow)
        stt_mask = disocc_mask & edge_mask # short term temporal consistency mask

        ##### CPU computation done #############

        flow = flow.to(self.device)
        rev_flow = rev_flow.to(self.device)
        p_prev_frame = p_prev_frame.to(self.device)
        prev_warped = prev_warped.to(self.device)
        stt_mask= stt_mask.to(self.device)

        # ========= sanity checks ============
        assert prev_warped.shape == p_prev_frame.shape, f"Shapes {prev_warped.shape} and {p_prev_frame.shape} don't match" 
        # ========= end of sanity checks ============   

        # w_img = self.proc.postprocess(stt_mask.clone())
        # w_img = transforms.ToPILImage()(stt_mask.clone().cpu().float())
        # w_img.save('output_frames/pmask.jpg') 

        ### Saving masks and warped images for debugging ##########

        self.proc.postprocess(prev_warped.clone()).save(f'output_flows/warped_{i}.jpg')
        mask_img = torch.ones_like(prev_warped, dtype=torch.uint8, device='cpu')*255
        mask_img[:, :, ~edge_mask] = 0 # @masked regions the image is black
        write_jpeg(mask_img.squeeze(0), f'output_flows/edge_mask_{i}.jpg')
        mask_img[:, :, :] = 255
        mask_img[:, :, ~disocc_mask] = 0
        write_jpeg(mask_img.squeeze(0), f'output_flows/disocc_mask_{i}.jpg')
        mask_img[:, :, :] = 255
        mask_img[:, :, ~stt_mask] = 0
        write_jpeg(mask_img.cpu().squeeze(0), f'output_flows/mask_{i}.jpg')

        ### End of Saving masks and warped images for debugging ##########

        # global noise_img
        # prev_warped = prev_warped.clip(0, 255)
        # print('ok, here goes', noise_img.device, prev_warped.device)
        
        noise_img = prev_warped.clone()
        print('have set the thing to equal, duh!!')
        print('ok, here goes', noise_img.device, prev_warped.device)
        print(torch.allclose(noise_img, prev_warped), '0')
        if not torch.allclose(noise_img, prev_warped):
            print(torch.isnan(noise_img).nonzero())
            print(torch.isnan(prev_warped).nonzero())
            print('duhhh, it happened')
            import matplotlib.pyplot as plt

            abs_diff = torch.abs(noise_img - prev_warped)
            plt.imshow(abs_diff[0].permute(1, 2, 0).numpy())
            plt.savefig('random/OKOKOKOK.png')
            # print(noise_img[0][2])
            # print(prev_warped[0][2])
            print(noise_img - prev_warped)
            idxs = (noise_img - prev_warped).nonzero()
            print(idxs)
            print(noise_img[idxs], prev_warped[idxs])
            print('start')
            print(np.linalg.norm((noise_img.detach() - prev_warped)/255))
            print(torch.mean(stt_mask * torch.square(noise_img.detach()-prev_warped)).item())
            # print(np.allclose(noise_img.detach(), prev_warped), '5')
            # exit()
            # print(stt_mask.shape, torch.square(noise_img-prev_warped).shape)
            print('end')
            prev_warped[prev_warped.isnan()] = p_style[prev_warped.isnan()]
            noise_img = prev_warped.clone()
            print('now?\n', torch.allclose(noise_img, prev_warped))
            exit()
        # noise_img = p_content.clone()
        noise_img.requires_grad = True
        num_iter = [0]
        iter_range = tqdm(range(num_steps))
        lr = 1
        # optimizer = torch.optim.LBFGS([noise_img], max_iter=1, lr=lr)
        optimizer = torch.optim.Adam([noise_img], lr=lr)
        # print(torch.sum(prev_warped == prev_warped.clip(0, 255)))
        # print(prev_warped.shape[1] * prev_warped.shape[2] * prev_warped.shape[3])

        def closure():
            iter_range.update()
            self.proc.postprocess(noise_img.detach().clone()).save(f'random/noise_{i}_{num_iter[0]}.jpg')
            style_outputs, content_outputs = self.ext(noise_img)
            loss = 0.
            num_iter[0] += 1
            content_loss = style_loss = 0.
            for key, val in style_outputs.items():
                # print(actual_gram_matrices[key].isnan().nonzero())
                # print(val.isnan().nonzero())
                style_loss += self.style_weights[key] * F.mse_loss(val, actual_gram_matrices[key])
            for key, val in content_outputs.items():
                # print(actual_content_outputs[key].isnan().nonzero())
                # print(val.isnan().nonzero())
                content_loss += self.content_weights[key] * F.mse_loss(val, actual_content_outputs[key])
            h, w = prev_warped.shape[2], prev_warped.shape[3]
            style_loss /= ((h * w))
            loss = style_loss.clip(0, 1e5) + content_loss.clip(0, 1e5)
            ## add the short term temporal consistency loss
            # print('start')
            # print(np.linalg.norm((noise_img.detach() - prev_warped)/255))
            # print(torch.mean(stt_mask * torch.square(noise_img.detach()-prev_warped)).item())
            # print(np.allclose(noise_img.detach(), prev_warped), '5')
            # exit()
            # print(stt_mask.shape, torch.square(noise_img-prev_warped).shape)
            # print('end')
            # noise_img2 = torch.clip(noise_img, 0, 255)
            stt_loss = self.stt_weight * torch.mean(stt_mask * torch.square(noise_img - prev_warped))
            print('style_loss =', style_loss.item())
            print('content_loss =', content_loss.item())
            print("stt_loss =", stt_loss.item())
            loss += stt_loss

            # a total variation loss to de-blur (make it smooth)
            tv_y = F.mse_loss(noise_img[:, :, 1:, :], noise_img[:, :, :-1, :])
            tv_x = F.mse_loss(noise_img[:, :, :, 1:], noise_img[:, :, :, :-1])
            tv_loss = 2 * (tv_y + tv_x)
            tv_loss *= 1e-1 # weight
            print('tv_loss=', tv_loss.item())
            loss += tv_loss

            optimizer.zero_grad()
            loss.backward()
            print(torch.max(noise_img.grad))
            torch.nn.utils.clip_grad_value_(noise_img, clip_value=1)
            print(torch.max(noise_img.grad))
            return loss
        
        for _ in range(num_steps):
            optimizer.step(closure)
        # exit()

        corr_img = noise_img.detach().clone()
        corr_img = self.proc.postprocess(corr_img)
        if save_path is not None:
            corr_img.save(save_path)
        return corr_img
    
    def squared_norm(self, obj: torch.Tensor) -> torch.Tensor:
        """
            computes the element wise squared norm of tensor with shape (2, H, W). Norm is taken over the first dimension.
        """
        return (obj * obj).sum(dim=0)

def get_frames(in_dir, video_name, video_ext):
    """
        :param in_dir: Input directory name
        :param video_name: Video name inside in_dir
        :param video_ext: Video extension (mp4 or gif)
    """
    if video_ext == 'gif':
        input_frames = []
        with Image.open(f'{in_dir}/{video_name}.{video_ext}') as f:
            for i in range(f.n_frames):
                f.seek(i)
                input_frames.append(transforms.PILToTensor()(f.convert("RGB")))
        input_frames = torch.stack(input_frames)
    elif video_ext == 'mp4':
        input_frames, _, _ = read_video(f'{in_dir}/{video_name}.{video_ext}', output_format="TCHW", pts_unit='sec')
    else:
        raise NotImplementedError("Skill Issue")
    return input_frames

def num_frames(in_dir, video_name, video_ext):
    n_frames = None
    if video_ext == 'gif':
        input_frames = []
        with Image.open(f'{in_dir}/{video_name}.{video_ext}') as f:
            n_frames = f.n_frames
    elif video_ext == 'mp4':
        input_frames, _, _ = read_video(f'{in_dir}/{video_name}.{video_ext}', output_format="TCHW", pts_unit='sec')
        n_frames = len(input_frames)
    else:
        raise NotImplementedError("Skill Issue")
    return n_frames

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--load_cached_flows", action="store_true")
    # parser.add_argument('--video', action='store_true')
    parser.add_argument('-videoname', type=str, default='dragon.gif')
    parser.add_argument('-stylename', type=str, default='aframov.jpg')
    parser.add_argument('-indir', type=str, default='input')
    parser.add_argument('-middir', type=str, default='output_frames')
    parser.add_argument('-outdir', type=str, default='output')
    parser.add_argument('-step', type=int, default=1)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.videoname, videoext = args.videoname.split('.')
    args.stylename, styleext = args.stylename.split('.')
    if args.videoname in abbrev_to_full:
        args.videoname = abbrev_to_full[args.videoname]
    if args.stylename in abbrev_to_full:
        args.stylename = abbrev_to_full[args.stylename]
    # prepare directory structure
    assert os.path.isdir(args.indir), 'invalid input directory'
    if not args.force:
        assert os.path.isdir(args.middir), 'invalid mid directory'
        assert os.path.isdir(args.outdir), 'invalid output directory'
        return args, videoext, styleext
    
    for dirname in ['mid', 'out']:
        dirname = getattr(args, dirname+'dir')
        if os.path.isdir(dirname): 
            shutil.rmtree(dirname)
        os.mkdir(dirname) 
    if not os.path.isdir('output_flows'):
        os.mkdir('output_flows')
        
    return args, videoext, styleext

if __name__ == "__main__":
    args, video_ext, style_ext = prepare()
    video_name = args.videoname
    style_name = args.stylename
    in_dir = args.indir
    mid_dir = args.middir
    out_dir = args.outdir
    step = args.step
    img_size = None
    n_frames = num_frames(in_dir, video_name, video_ext)
    
    if not args.force:
        combine_as_gif(
            f'{full_to_abbrev[video_name]}_processed_frame_', 'jpg', 
            mid_dir, out_dir, 1+(n_frames-1)//step, step, 
            f'{full_to_abbrev[video_name]}_{full_to_abbrev[style_name]}.gif')
        exit()
    
    input_frames = get_frames(in_dir, video_name, video_ext)
    img_size = tuple([i-i%8 for i in input_frames.shape[2:]]) # optical flow needs multiple of 8
    # img_size = [128,320]
    input_frames = transforms.Resize(img_size)(input_frames)
    list_frames = []
    for i in range(input_frames.shape[0]):
        if i % step == 0:
            list_frames.append(input_frames[i])
            write_jpeg(input_frames[i], f"{mid_dir}/{full_to_abbrev[video_name]}_frame_{i//step}.jpg")

    input_frames = torch.stack(list_frames) # input frames reduced using step

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

    for i in range(len(input_frames)):
        print("========iter: ", i, "============")
        torch.cuda.empty_cache()
        if i == 0:
            # if args.load_cached_flows:
            #     continue
            image_style_transfer = ImageStyleTransfer(img_size)
            image_style_transfer(
                f"{mid_dir}/{full_to_abbrev[video_name]}_frame_{i}.jpg", 
                f'{in_dir}/{style_name}.{style_ext}', 
                save_path=f"{mid_dir}/{full_to_abbrev[video_name]}_processed_frame_{i}.jpg", 
                init_img=None, 
                num_steps=100
            )
        else:
            video_style_transfer = VideoStyleTransfer(img_size)
            video_style_transfer(
                f'{mid_dir}/{full_to_abbrev[video_name]}_frame_{i}.jpg', 
                f'{in_dir}/{style_name}.{style_ext}', 
                optical_flow[i-1].to('cpu'),
                f'{mid_dir}/{full_to_abbrev[video_name]}_processed_frame_{i-1}.jpg',
                reverse_optical_flow[i-1].to('cpu'),
                save_path=f"{mid_dir}/{full_to_abbrev[video_name]}_processed_frame_{i}.jpg", 
                num_steps=100+50*i
            )
    combine_as_gif(f'{full_to_abbrev[video_name]}_processed_frame_', 'jpg', mid_dir, out_dir, 1+(n_frames-1)//step, 1, f'{full_to_abbrev[video_name]}_{full_to_abbrev[style_name]}.gif')
            

