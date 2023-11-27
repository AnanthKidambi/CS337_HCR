import subprocess
import threading
import argparse
import utils
import torch
import torchvision.io as io
import PIL
import re
import os

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument('--videoname', type=str, default='dragon.gif')
    parser.add_argument('--stylename', type=str, default='aframov.jpg')
    parser.add_argument('--indir', type=str, default='input')
    parser.add_argument('--middir', type=str, default='output_frames')
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--imgstyledir', type=str, default='no_stt_style_transfer')
    parser.add_argument('--flowdir', type=str, default='output_flows')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    utils.set_seed(args.seed)
    # utils.device = f'cuda:{args.cuda}' if torch.cuda.is_available() and not args.cpu else 'cpu'

    # prepare directory structure
    assert os.path.isdir(args.indir), 'invalid input directory'
    if not args.force:
        assert os.path.isdir(args.middir), 'invalid mid directory'
        assert os.path.isdir(args.outdir), 'invalid output directory'
        assert os.path.isdir(args.imgstyledir), 'invalid image style transfer directory'
        assert os.path.isdir(args.flowdir), 'invalid flow directory'
    else:
        for dir in [args.middir, args.outdir, args.imgstyledir, args.flowdir]:
            if not os.path.isdir(dir):
                os.mkdir(dir)
    return args

def get_free_devices(required: int):
    cmd = "nvidia-smi --query-gpu=memory.free --format=csv"
    gpu_free_memory = subprocess.check_output(cmd.split(' '))
    gpu_free_memory = gpu_free_memory.decode("utf-8")
    gpu_free_memory = gpu_free_memory.split('\n')
    gpu_free_memory = gpu_free_memory[1:-1]
    gpu_free_memory = [int(re.findall(r'\d+', x)[0]) for x in gpu_free_memory]
    available_sorted = sorted(enumerate(gpu_free_memory), key=lambda x: x[1])
    return [f'{x[0]}' for x in available_sorted[:required]]

def get_num_frames(video_path, step):
    video_ext = video_path.split('.')[-1]
    if video_ext == 'gif':
        input_frames = []
        with PIL.Image.open(video_path) as f:
            return f.n_frames//step + 1
    elif video_ext == 'mp4':
        input_frames, _, _ = io.read_video(video_path, pts_unit='sec')
        return len(input_frames)//step + 1
    else:
        raise NotImplementedError("other types not supported")
    
def run_range(arg_string: str, device: str, start: int, end: int):
    os.system(f'python video_style.py {arg_string} --no_stt --start {start} --end {end} --cuda {device} --load_cache')

def run_flows(arg_string: str, device: str):
    os.system(f'python video_style.py {arg_string} --cuda {device}')

def main():
    # parse arguments
    args = parse()
    # arg_string = f'--videoname {args.videoname} --stylename {args.stylename} --indir {args.indir} --middir {args.middir} --outdir {args.outdir} --imgstyledir {args.imgstyledir} --flowdir {args.flowdir} --step {args.step} --debug {args.debug} --wandb {args.wandb}'
    arg_string = f'--force --videoname {args.videoname} --stylename {args.stylename} --indir "{args.indir}" --middir "{args.middir}" --outdir "{args.outdir}" --step {args.step}'
    if args.wandb:
        arg_string += ' --wandb'
    if args.debug:
        arg_string += ' --debug'
    print(arg_string)
    
    # get free devices
    num_devices = 4
    free_devices = get_free_devices(num_devices)
    print(f'free devices: {free_devices}')

    # compute flows and image style transfer for each frame
    run_flows(arg_string, free_devices[0])

    name, ext = args.videoname.split('.')
    if name in utils.abbrev_to_full:
        name = utils.abbrev_to_full[name]
    video_path = f'{args.indir}/{name}.{ext}'
    num_frames = get_num_frames(video_path, args.step)
    print(f'num_frames: {num_frames}')
    num_frames_per_device = num_frames // num_devices + 1
    for i in range(num_devices):
        start = i*num_frames_per_device
        end = (i+1)*num_frames_per_device if i != num_devices - 1 else num_frames
        print(f'start: {start}, end: {end}')
        t = threading.Thread(target=run_range, args=(arg_string, free_devices[i], start, end))
        t.start()
    
    for i in range(num_devices):
        t.join()

    # combine frames into gif
    try:
        utils.make_gif(name, num_frames, step=args.step)
    except:
        print("could not make gif")

if __name__ == "__main__":
    main()
