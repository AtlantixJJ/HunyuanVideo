"""Submit mutilple jobs."""
import threading, subprocess, argparse, glob, json, os
import concurrent.futures


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="-1")
parser.add_argument("--func", default="all")
args = parser.parse_args()

devices = args.gpu.split("/")
n_slots = len(devices)
functions = {}


def register(func):
    """Decorator to register a function in a dictionary."""
    global functions
    functions[func.__name__] = func
    return func


@register
def debug():
    cmd = 'python3 scripts/debug_sample.py --video-size 960 960 --video-length 17 --infer-steps 50 --prompt "A cat walks on the grass, realistic style." --flow-reverse --seed 0 --rank {rank} --n-rank {n_rank}'

    for rank in [0, 1]:
        for n_rank in [0, 1]:
            yield cmd.format(rank=rank, n_rank=n_rank)


@register
def sample():
    cmd = 'python3 scripts/sample_diffusers.py --pretrained_lora_path \"{lora_path}\" --first_image_path data/hunyuan_distillation_cfg6/{file_name}.mp4 --pretrained_path {pretrained_path} --vpt_mode \"{vpt_mode}\" --mode i2v --prompt \"{prompt}\" --height 720 --width 720 --num_frames 17 --num_infer_steps 50 --output_path expr/{expr_name}/{file_name}_{iteration}.mp4'

    args = [
        ('00000_seed1981', 'A close-up portrait of a young man with short black hair, wearing a black hoodie, captured while yawning or shouting. His mouth is wide open, showing his teeth and tongue. His eyes are squinting, eyebrows raised, and his skin shows visible acne and redness. The background is plain light gray, and the lighting is soft and even, clearly illuminating his face.'),
        ('00001_seed4507', 'A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature\'s embrace.'),
    ]
    prompt = []
    expr_name = 'vpt-deep-add-1'
    
    for iteration in [1, 50, 100, 150, 200, 250]:#[1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        for file_name, prompt in args:
            lora_path = f'expr/{expr_name}/pytorch_lora_weights_{iteration}.safetensors' if 'lora' in expr_name else ''
            if iteration == 0:
                lora_path = ''
            
            vpt_path = f'expr/{expr_name}/vpt-{iteration}.pt' if 'vpt' in expr_name else ''
            vpt_mode = 'deep-add-1'

            yield cmd.format(vpt_mode=vpt_mode, pretrained_path=vpt_path, expr_name=expr_name, file_name=file_name, prompt=prompt, lora_path=lora_path, iteration=iteration)


def assign_slots(func_name):
    slot_cmds = [[] for _ in devices]
    for idx, cmd in enumerate(functions[func_name]()):
        device_id = idx % n_slots
        device_prefix = ""
        if devices[device_id] != -1:
            device_prefix = f"CUDA_VISIBLE_DEVICES={devices[device_id]} "
        slot_cmds[device_id].append(f"{device_prefix}{cmd}")
    return slot_cmds


def worker(device_id, device_cmds):
    for idx, cmd in enumerate(device_cmds):
        print(f"=> Device [{device_id}] GPU [{devices[device_id]}]: Starting task {idx} / {len(device_cmds)}")
        print(cmd)
        subprocess.run(cmd, shell=True)


slot_cmds = assign_slots(args.func)

with concurrent.futures.ThreadPoolExecutor(len(devices)) as executor:
    # Submit each command for execution
    futures = [executor.submit(worker, device_id, device_cmds) for device_id, device_cmds in enumerate(slot_cmds)]

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)
