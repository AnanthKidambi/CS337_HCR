import subprocess
import threading
import argparse

import IPython
import utils
import torchvision.io as io
import PIL
import re
import os

def parse():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--n_threads', type=int, default=1)
    args = parser.parse_args()
    
    utils.set_seed(args.seed)

    # prepare directory structure
    assert os.path.isdir(args.styledir), 'invalid style directory'
    assert os.path.isdir(args.contentdir), 'invalid content directory'
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
    available_sorted = sorted(enumerate(gpu_free_memory), key=lambda x: -x[1])
    return [f'{x[0]}' for x in available_sorted[:required]]

def get_num_frames(video_path, step):
    video_ext = video_path.split('.')[-1]
    if video_ext == 'gif':
        input_frames = []
        with PIL.Image.open(video_path) as f:
            return f.n_frames//step
    elif video_ext == 'mp4':
        input_frames, _, _ = io.read_video(video_path, pts_unit='sec')
        return len(input_frames)//step
    else:
        raise NotImplementedError("other types not supported")

def run_video_style_transfer(arg_string: str, device: str):
    os.system(f'python main.py {arg_string} --cuda {device}')

def run_range(arg_string: str, device: str, start: int, end: int):
    print(f'python main.py {arg_string} --no_stt --start {start} --end {end} --cuda {device}')
    os.system(f'python main.py {arg_string} --no_stt --start {start} --end {end} --cuda {device}')

def main():
    # parse arguments
    args = parse()
    arg_string = f'--force --videoname {args.videoname} --stylename {args.stylename} --styledir {args.styledir} --contentdir {args.contentdir} --middir {args.middir} --outdir {args.outdir} --imgstyledir {args.imgstyledir} --flowdir {args.flowdir} --step {args.step}'
    if args.wandb:
        arg_string += ' --wandb'
    if args.debug:
        arg_string += ' --debug'
    print('Args:', arg_string)
    
    # get free devices
    num_devices = args.n_threads
    free_devices = get_free_devices(num_devices)
    print(f'free devices: {free_devices}')

    name, ext = args.videoname.split('.')
    if name in utils.abbrev_to_full:
        name = utils.abbrev_to_full[name]
    video_path = f'{args.contentdir}/{name}.{ext}'
    num_frames = get_num_frames(video_path, args.step)
    print(f'num_frames: {num_frames}')
    
    num_frames_per_device = num_frames // num_devices + 1
    
    threads = []

    for i in range(num_devices):
        start = i*num_frames_per_device
        end = (i+1)*num_frames_per_device 
        if end > num_frames: end = -1
        print(f'start: {start}, end: {end}')
        t = threading.Thread(target=run_range, args=(arg_string, free_devices[i], start, end))
        threads.append(t)
        try:
            t.start()
        except:
            print("could not run range", start, end)
            exit(1)
    
    for i in range(num_devices):
        threads[i].join()

    print("finished range")

    free_dev = get_free_devices(1)[0]
    print(f'video style transfer on device: cuda {free_dev}')
    try:
        run_video_style_transfer(arg_string, free_dev)
    except:
        print("could not run video style transfer")
        exit(1)

    # combine frames into gif
    try:
        utils.combine_as_gif(f'{name}_processed_frame_', 'jpg', args.middir, args.outdir, num_frames, args.step, f'{name}_{args.stylename.split(".")[0]}.gif')
    except:
        print("could not make gif")
        __import__('IPython').embed()
        exit(1)

if __name__ == "__main__":
    main()
