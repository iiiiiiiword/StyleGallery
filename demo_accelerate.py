import os
import gc
import torch
import argparse
from diffusers import DDIMScheduler, EulerDiscreteScheduler, LCMScheduler
from torchvision.utils import save_image
import utils
from pipeline import StyleGallery

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    mode_defaults = {
        "hyper": {
            "num_optimize_steps": 28,
            "iters": 3,
            "lr": 0.15,
            "c_ratio": 0.38,
            "lora_path": "pretrained_models/Hyper-SD15-4steps-lora.safetensors"
        },
        "lcm": {
            "num_optimize_steps": 28,
            "iters": 3,
            "lr": 0.25,
            "c_ratio": 0.33,
            "lora_path": "pretrained_models/lcm-lora-sd1.5"
        }
    }
    
    cfg = mode_defaults[args.mode]
    num_optimize_steps = args.num_optimize_steps if args.num_optimize_steps is not None else cfg["num_optimize_steps"]
    iters = args.iters if args.iters is not None else cfg["iters"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    c_ratio = args.c_ratio if args.c_ratio is not None else cfg["c_ratio"]
    lora_path = args.lora_path if args.lora_path is not None else cfg["lora_path"]

    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Loading model from {args.model_name}...")
    
    scheduler = DDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    controller = utils.Controller(self_layers=(args.start_layer, args.end_layer))

    pipe = StyleGallery.from_pretrained(
        args.model_name, 
        scheduler=scheduler, 
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)

    pipe.extractor = pipe.unet 
    torch.manual_seed(args.seed)

    content_original_mask = None
    if args.content_mask_path is not None:
        content_original_mask = utils.convert_mask_to_array(args.content_mask_path)

    print("Performing clustering and matching...")
    with torch.inference_mode():
        cluster_matches, content_dict, style_dict = pipe.cluster_match(
            args.content_image,
            args.style_images,
            steps=args.noise_steps,
            content_original_mask=content_original_mask,
            use_sam=args.use_sam,
            use_depth=args.use_depth 
        )

    if args.print:
        print("----------------------------------------------------------------------------------------------")
        match_summary = {}
        for content_cluster, match_info in cluster_matches.items():
            match_summary[content_cluster] = {
                'style_idx': match_info['style_dict_index'],
                'style_cluster': match_info['style_cluster'],
                'similarity': match_info['similarity']
            }
        print(f"Cluster Match Summary: {match_summary}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if args.mode == "hyper":
        print(f">>> Using Hyper-SD (Turbo Mode) with {lora_path}")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing"
        )
    elif args.mode == "lcm":
        print(f">>> Switching to LCM for Stage 2 (Optimization) using {lora_path}")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    print(f"Starting style transfer (steps={num_optimize_steps}, lr={lr}, iters={iters}, c_ratio={c_ratio})...")
    output_image = pipe.style_transfer(
        content_dict=content_dict,
        style_dict=style_dict,
        controller=controller,
        cluster_matches=cluster_matches,
        num_optimize_steps=num_optimize_steps, 
        iters=iters,
        lr=lr,
        c_ratio=c_ratio
    )
    
    pipe.unfuse_lora()
    
    save_path = os.path.join(args.output_folder, f"result_{args.mode}.png")
    save_image(output_image, save_path)
    print(f"Done! Image saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGallery Inference Script")
    parser.add_argument("--model_name", type=str, default="pretrained_models/runwayml_stable-diffusion-v1-5", help="path to SD 1.5")
    
    # 新增：模式选择
    parser.add_argument("--mode", type=str, choices=["hyper", "lcm"], default="lcm", help="Choose acceleration mode: hyper or lcm")
    parser.add_argument("--lora_path", type=str, default=None, help="Custom LoRA path (overrides mode defaults if provided)")
    
    parser.add_argument("--content_image", type=str, default="content/city_scene/0002.png", help="content image path")
    parser.add_argument("--style_images", type=str, nargs='+', default=["style/VanGogh/VanGogh010.png"], help="List of style image paths")
    parser.add_argument("--content_mask_path", type=str, default=None, help="additional mask path")
    parser.add_argument("--output_folder", type=str, default="outputs", help="output folder path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sam", action="store_true", default=False, help="Simply to verify scalability, perhaps not be very practical")
    parser.add_argument("--use_depth", action="store_true", default=True)
    parser.add_argument("--start_layer", type=int, default=10)
    parser.add_argument("--end_layer", type=int, default=16)
    parser.add_argument("--noise_steps", type=int, default=15)
    
    parser.add_argument("--num_optimize_steps", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--c_ratio", type=float, default=None)
    
    parser.add_argument("--print", action="store_true", default=False, help="show cluster mathcing results")

    args = parser.parse_args()
    main(args)