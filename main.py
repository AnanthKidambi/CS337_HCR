import torch
import torchvision.models  as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision.io import read_video, write_jpeg
import numpy as np
import random
from utils import combine_as_gif, abbrev_to_full, full_to_abbrev
import utils
import argparse
from neural_style import ImageStyleTransfer, processor, extractor
from optical_flow import generate_optical_flow
import os
import inspect
import wandb
import skimage

debug = utils.debug
high_val = utils.high_val
seed = utils.seed
use_wandb = None

class VideoStyleTransfer:
    def __init__(self, img_size) -> None:

        self.device = torch.device(utils.device if torch.cuda.is_available() else "cpu")
        self.pre_means = [0.48501961, 0.45795686, 0.40760392]
        self.pre_stds = [1, 1, 1]
        self.img_size = img_size

        style_layer_nums = [1, 6, 11, 20, 29] # taken from https://www.mathworks.com/help/deeplearning/ref/vgg19.html
        content_layer_num = 22

        self.content_layers = {f"features.{content_layer_num}": "conv4_2"}
        self.style_layers = {f"features.{i}": f"relu{j+1}_1" for j, i in enumerate(style_layer_nums)} # hardcoded layer names

        _style_wt_list = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.style_weights = {val:_style_wt_list[j]*10 for j, val in enumerate(self.style_layers.values())}
        
        _content_wt_list = [0.1]
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
    
    def get_images(self, content, style, prev_out, img_style_res):
        content_img = Image.open(content).convert("RGB")
        style_img = Image.open(style).convert("RGB")
        prev_out_img = Image.open(prev_out).convert("RGB")
        img_style_res = Image.open(img_style_res).convert("RGB")
        return style_img, content_img, prev_out_img, img_style_res

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

        def transform_function(coords):
            coords_ = np.flip(np.asarray(coords, dtype=int), axis=1)
            coords_ = coords_[:, 0]*flow_map.shape[2] + coords_[:, 1]

            flat_flow0 = flow_map[0].flatten()
            flat_flow1 = flow_map[1].flatten()

            return np.concatenate([flat_flow0[coords_][:, np.newaxis], flat_flow1[coords_][:, np.newaxis]], axis=1)

        for chan in range(p_image.shape[0]):
            temp_merged = skimage.transform.warp(p_image[chan].numpy(), transform_function, preserve_range=True) 

            if debug:
                if np.max(np.abs(temp_merged)) >= 1e30:
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                    import IPython
                    IPython.embed()
                    exit()

            dst.append(torch.tensor(temp_merged))
        return torch.stack(dst).cpu()
    
    def get_mask_disoccluded(self, flow : torch.Tensor, rev_flow : torch.Tensor, frame_num : int):
        '''
            :param flow: Forward flow from frame[i-1]. Shape should be (2, H, W) and device should be CPU
            :param rev_flow: Reverse flow from frame[i] to frame[i-1]. Shape should be (2, H, W) and device should be CPU

            Outputs the disoccluded region mask with shape (H, W) on CPU
        '''
        assert flow.device == torch.device("cpu"), 'need cpu in warp (get_mask_disoccluded)'
        assert rev_flow.device == torch.device("cpu"), 'need cpu in warp (get_mask_disoccluded)'

        # get w_tilde
        # To detect disocclusions, we perform a forward-backward consistency check of the optical flow [11]. Let w = (u,v) be the optical flow in forward direction and wˆ = (uˆ,vˆ) the flow in backward direction. Denote by we the forward flow warped to the second image:
        w_tilde = self.warp(flow, rev_flow)

        return (self.squared_norm(w_tilde + rev_flow) <= 
                (0.03*(self.squared_norm(w_tilde) + self.squared_norm(rev_flow))) + 1)

    def get_mask_edge(self, rev_flow : torch.Tensor):
        '''
            :param rev_flow: Reverse flow from frame[i] to frame[i-1]. Shape should be (2, H, W) and device should be CPU

            Outputs the disoccluded region mask with shape (H, W) on CPU
        '''
        # no need to be in cpu?
        # assert rev_flow.device == torch.device("cpu")

        x_grad = torch.gradient(rev_flow[1])
        y_grad = torch.gradient(rev_flow[0])
        return (self.squared_norm(torch.stack(x_grad)) + self.squared_norm(torch.stack(y_grad)) <= (0.03*self.squared_norm(rev_flow) + 0.002))

    def __call__(self, content : str, style : str, flow : torch.Tensor,  prev_out : str, rev_flow : torch.Tensor, frame_num : int, img_style_res : str, save_path = None, num_steps = 500):
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
        # ========= end of sanity checks ============

        self.style_img, self.content_img, self.prev_frame_img, self.img_style_res = self.get_images(content, style, prev_out, img_style_res)

        p_content = self.proc.preprocess(self.content_img).to(self.device)
        p_style = self.proc.preprocess(self.style_img).to(self.device)
        p_prev_frame = self.proc.preprocess(self.prev_frame_img).cpu()
        p_img_style = self.proc.preprocess(self.img_style_res).to(self.device)

        p_style[p_style.isnan()] = p_style.mean()
        p_content[p_content.isnan()] = p_content.mean()
        p_prev_frame[p_prev_frame.isnan()] = p_prev_frame.mean()

        # compute gram matrices and feature maps to plug into the loss
        if debug:
            if (torch.max(torch.abs(p_style)) > high_val) or (torch.max(torch.abs(p_content)) > high_val):
                print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                import IPython
                IPython.embed()
                exit()

        actual_gram_matrices, _ = self.ext(p_style, normalize=True)
        _, actual_content_outputs = self.ext(p_content, normalize=True)

        ##### CPU computation #############
        try:
            partial_transform_1 = transforms.Compose([
                transforms.Resize(self.proc.img_size),
                transforms.ToTensor()
            ])
            partial_transform_2 = transforms.Compose([
                transforms.Normalize(mean=self.proc.means, std=self.proc.std),
                transforms.Lambda(lambda x: x.mul_(255))
            ])
            temp1 = partial_transform_1(self.prev_frame_img).squeeze(0).cpu()
            temp2 = self.warp(temp1, rev_flow).unsqueeze(0).to(self.device)
            prev_warped = partial_transform_2(temp2)
        except Exception as e:
            print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
            import IPython
            IPython.embed()

        if debug: 
            if prev_warped.isnan().any() or (torch.max(torch.abs(prev_warped)) >= high_val):
                print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                import IPython
                IPython.embed()
                exit()

        disocc_mask = self.get_mask_disoccluded(flow, rev_flow, frame_num)
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

        ### Saving masks and warped images for debugging ##########
        if debug:
            self.proc.postprocess(prev_warped.clone()).save(f'output_flows/warped_{frame_num}.jpg')
            mask_img = torch.ones_like(prev_warped, dtype=torch.uint8, device='cpu')*255
            mask_img[:, :, ~edge_mask] = 0 # @masked regions the image is black
            write_jpeg(mask_img.squeeze(0), f'output_flows/edge_mask_{frame_num}.jpg')
            mask_img[:, :, :] = 255
            mask_img[:, :, ~disocc_mask] = 0
            write_jpeg(mask_img.squeeze(0), f'output_flows/disocc_mask_{frame_num}.jpg')
            mask_img[:, :, :] = 255
            mask_img[:, :, ~stt_mask] = 0
            write_jpeg(mask_img.cpu().squeeze(0), f'output_flows/mask_{frame_num-1}.jpg')
        ### End of Saving masks and warped images for debugging ##########

        prev_warped[:, :, ~disocc_mask] = p_content[:, :, ~disocc_mask]

        # prev_warped = prev_warped.clip(0, 255)

        # noise = torch.randn_like(prev_warped, device=self.device, requires_grad=False) * 10
        # noise_img = prev_warped.clone()
        # noise_img[:, :, ~stt_mask] = p_img_style[:, :, ~stt_mask]
        # noise_img = p_content.clone()
        noise_img = p_img_style.clone()

        noise_img.requires_grad = True
        num_iter = [0]
        iter_range = tqdm(range(num_steps)) if debug else range(num_steps)
        optimizer = torch.optim.LBFGS([noise_img], max_iter=1)

        def closure():
            if debug:
                iter_range.update()
                if noise_img.isnan().any():
                    print(noise_img.isnan().any())
                    print(prev_warped.isnan().any())
                    print(torch.allclose(noise_img, prev_warped))
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                    import IPython
                    IPython.embed()
                    exit()
            style_outputs, content_outputs = self.ext(noise_img, normalize=True)
            if debug:
                for val in style_outputs.values():
                    if val.isnan().any() or (torch.max(torch.abs(val)) >= high_val):
                        print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                        import IPython
                        IPython.embed()
                        exit()
                for val in content_outputs.values():
                    if val.isnan().any() or (torch.max(torch.abs(val)) >= high_val):
                        print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                        import IPython
                        IPython.embed()
                        exit()
                if len(noise_img.isnan().nonzero()) == 0 and any(len(val.isnan().nonzero()) > 0 for val in style_outputs.values()):
                    print('impossible')
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                    import IPython
                    IPython.embed()
                    exit()    
                    
            loss = 0.
            num_iter[0] += 1
            content_loss = style_loss = 0.
            for key, val in style_outputs.items():
                if debug and val.isnan().any(): 
                    print(val.isnan().nonzero(), val.isinf().nonzero())
                style_loss += self.style_weights[key] * F.mse_loss(val, actual_gram_matrices[key]) 
            for key, val in content_outputs.items():
                content_loss += self.content_weights[key] * F.mse_loss(val, actual_content_outputs[key]) * 0.5
            loss = style_loss + content_loss

            ## add the short term temporal consistency loss
            stt_loss = self.stt_weight * torch.mean(stt_mask * torch.square(noise_img - prev_warped)) * 0.5
            if debug:
                print('style_loss =', style_loss.item())
                print('content_loss =', content_loss.item())
                print("stt_loss =", stt_loss.item())

            loss += stt_loss
    
            # a total variation loss to de-blur (make it smooth)
            tv_y = F.mse_loss(noise_img[:, :, 1:, :], noise_img[:, :, :-1, :])
            tv_x = F.mse_loss(noise_img[:, :, :, 1:], noise_img[:, :, :, :-1])
            tv_loss = tv_y + tv_x
            tv_loss *= 1e-2 # weight
            loss += tv_loss

            if debug: 
                print('tv_loss=', tv_loss.item())

            if use_wandb:
                wandb.log({'loss' : loss.item(), 'content_loss': content_loss.item(), 'style_loss' : style_loss.item(), 'stt_loss' : stt_loss.item(), 'tv_loss' : tv_loss.item()})

            if debug:
                if np.abs(style_loss.item()) >= high_val:
                    print("Style loss is inf")
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                    import IPython
                    IPython.embed()
                    exit()    

                if np.abs(content_loss.item()) > high_val:
                    print("Content loss is inf")
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                    import IPython
                    IPython.embed()
                    exit()

                if np.abs(stt_loss.item()) >= high_val:
                    print("STT loss is inf")
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                    import IPython
                    IPython.embed()
                    exit()

                if np.abs(tv_loss.item()) >= high_val:
                    print("TV loss is inf")
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of video_style.py")
                    import IPython
                    IPython.embed()
                    exit()

            optimizer.zero_grad()
            loss.backward()
            if debug: 
                print(torch.max(noise_img.grad))
            if debug: 
                print(torch.max(noise_img.grad))
            return loss
        
        for _ in range(num_steps):
            optimizer.step(closure)

            if use_wandb:
                wandb.log({'max-noise-img' : torch.max(torch.abs(noise_img)).item()})

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
        raise NotImplementedError("Not implemented support for the given extension")
    return n_frames

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--load_cache", action="store_true")
    parser.add_argument('--videoname', type=str, default='dragon.gif')
    parser.add_argument('--stylename', type=str, default='kandinsky.jpg')
    parser.add_argument('--styledir', type=str, default='input/style')
    parser.add_argument('--contentdir', type=str, default='input/content')
    parser.add_argument('--middir', type=str, default='output_frames')
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--imgstyledir', type=str, default='no_stt_style_transfer')
    parser.add_argument('--flowdir', type=str, default='output_flows')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_stt', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=-1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()
    
    torch.cuda.empty_cache()
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    utils.device='cpu' if args.cpu else f'cuda:{args.cuda}'

    args.videoname, videoext = args.videoname.split('.')
    args.stylename, styleext = args.stylename.split('.')
    if args.videoname in abbrev_to_full:
        args.videoname = abbrev_to_full[args.videoname]
    if args.stylename in abbrev_to_full:
        args.stylename = abbrev_to_full[args.stylename]

    # prepare directory structure
    assert os.path.isdir(args.styledir), 'invalid style directory'
    assert os.path.isdir(args.contentdir), 'invalid content directory'
    if not args.force:
        assert os.path.isdir(args.middir), 'invalid mid directory'
        assert os.path.isdir(args.outdir), 'invalid output directory'
        return args, videoext, styleext
    
    for dirname in ['mid']:
        dirname = getattr(args, dirname+'dir')
        if not os.path.isdir(dirname): os.mkdir(dirname) 
    if not os.path.isdir('output_flows'):
        os.mkdir('output_flows')
        
    return args, videoext, styleext

if __name__ == "__main__":
    args, video_ext, style_ext = prepare()
    video_name = args.videoname
    style_name = args.stylename
    style_dir = args.styledir
    content_dir = args.contentdir
    mid_dir = args.middir
    out_dir = args.outdir
    debug = args.debug
    utils.debug = args.debug
    step = args.step
    img_size = None
    use_wandb = args.wandb

    if use_wandb:
        import time
        wandb.init(project='video-style-transfer', name=f"expt_{int(time.time())}")

    n_frames = num_frames(content_dir, video_name, video_ext)
    
    if not args.force:
        n_frames = 50   
        combine_as_gif(
            f'{full_to_abbrev[video_name]}_processed_frame_', 'jpg', 
            mid_dir, out_dir, 1 + (n_frames - 1)//step, step, 
            f'{full_to_abbrev[video_name]}_{full_to_abbrev[style_name]}.gif')
        exit()
    input_frames = get_frames(content_dir, video_name, video_ext)[:args.n_frames if args.n_frames != -1 else None]
    img_size = tuple([i - i%8 for i in input_frames.shape[2:]]) # optical flow needs multiple of 8

    input_frames = transforms.Resize(img_size)(input_frames)
    list_frames = []
    for i in range(input_frames.shape[0]):
        if i % step == 0:
            list_frames.append(input_frames[i])
            write_jpeg(input_frames[i], f"{mid_dir}/{full_to_abbrev[video_name]}_frame_{i//step}.jpg")

    input_frames = torch.stack(list_frames) # input frames reduced step

    if not args.load_cache and not args.no_stt:
        print("Computing forward optical flows...")        
        optical_flow = generate_optical_flow(input_frames, reverse=False).cpu()
        torch.save(optical_flow, f'{args.flowdir}/optical_flow.pt')

        print("Computing backward optical flows...")
        reverse_optical_flow = generate_optical_flow(input_frames, reverse=True).cpu()
        torch.save(reverse_optical_flow, f'{args.flowdir}/reverse_optical_flow.pt')
    
    elif not args.no_stt:
        optical_flow = torch.load(f'{args.flowdir}/optical_flow.pt').cpu()
        reverse_optical_flow = torch.load(f'{args.flowdir}/reverse_optical_flow.pt').cpu()

    print("Device being used:", utils.device)
    img_range = None
    if args.no_stt:
        img_range = range(args.start, args.end if args.end != -1 else len(input_frames))
    else:
        img_range = range(len(input_frames))
    if not args.debug and not args.no_stt:
        img_range = tqdm(img_range)
    for i in img_range:
        if debug or args.no_stt:    
            print("========iter: ", i, "============")
        torch.cuda.empty_cache()
        if args.no_stt:
            image_style_transfer = ImageStyleTransfer(img_size)
            image_style_transfer(
                f"{mid_dir}/{full_to_abbrev[video_name]}_frame_{i}.jpg", 
                f'{style_dir}/{style_name}.{style_ext}', 
                save_path=f"{args.imgstyledir}/{full_to_abbrev[video_name]}_processed_frame_{i}.jpg", 
                init_img=None, 
                num_steps=500
            )
        else:
            if i == 0:
                os.system(f"cp {args.imgstyledir}/{full_to_abbrev[video_name]}_processed_frame_{i}.jpg {mid_dir}/{full_to_abbrev[video_name]}_processed_frame_{i}.jpg")
            else:
                video_style_transfer = VideoStyleTransfer(img_size)
                video_style_transfer(
                    f'{mid_dir}/{full_to_abbrev[video_name]}_frame_{i}.jpg', 
                    f'{style_dir}/{style_name}.{style_ext}', 
                    optical_flow[i-1],
                    f'{mid_dir}/{full_to_abbrev[video_name]}_processed_frame_{i-1}.jpg',
                    reverse_optical_flow[i-1],
                    frame_num=i,
                    img_style_res=f'{args.imgstyledir}/{full_to_abbrev[video_name]}_processed_frame_{i}.jpg',
                    save_path=f"{mid_dir}/{full_to_abbrev[video_name]}_processed_frame_{i}.jpg", 
                    num_steps=10
                )

