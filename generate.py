from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import HunyuanPaintPipeline
from diffusers import DiffusionPipeline
import trimesh
import torch
import os
import yaml
import json
from PIL import Image
import argparse
import glob
import time
from tqdm import tqdm

# --- Configuration Loading with Error Handling ---
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:  # Handle empty or invalid YAML file
                print(f"Warning: Config file at {config_path} is empty or invalid. Using defaults.")
                return {}
            return config
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using defaults.")
        return {}  # Return empty dict if file not found
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file: {e}")
        exit(1)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate highest-quality 3D models using Hunyuan3D 2.0.")
parser.add_argument("-i", "--input", required=True, help="Path to input image or directory of images.  Use -t with text prompts.")
parser.add_argument("-t", "--text", action="store_true", help="Use text prompts instead of images.")
parser.add_argument("-o", "--output_dir", default="outputs", help="Output directory (default: outputs).")
parser.add_argument("-c", "--config_dir", default="configs", help="Directory containing config files (default: configs).")  # Still useful for scheduler
parser.add_argument("-f", "--format", default="glb", choices=["glb", "obj", "stl"], help="Output format (glb, obj, or stl, default: glb).")

# --- MAX QUALITY DEFAULTS ---
parser.add_argument("--texture_resolution", type=int, default=4096, help="Desired texture resolution (e.g., 1024, 2048, 4096). Higher = better quality, but more VRAM needed.")
parser.add_argument("--dit_steps", type=int, default=300, help="Number of inference steps for DiT (shape generation). Higher = better quality.")
parser.add_argument("--paint_steps", type=int, default=250, help="Number of inference steps for Paint (texture generation). Higher = better quality.")
parser.add_argument("--delight_steps", type=int, default=150, help="Number of inference steps for Delight (relighting). Higher = better quality.")
parser.add_argument("--dit_guidance", type=float, default=9.5, help="Guidance scale for DiT. Higher = stronger adherence to prompt.")
parser.add_argument("--paint_guidance", type=float, default=8.5, help="Guidance scale for Paint.")
parser.add_argument("--delight_guidance", type=float, default=7.5, help="Guidance scale for Delight")
parser.add_argument("--octree_resolution", type=int, default=512, help="Octree resolution for mesh generation. Higher = more detail.")
parser.add_argument("--mc_level", type=float, default=0.0, help="Marching cubes level for mesh generation.")
args = parser.parse_args()


# --- Load Configurations ---
dit_config = load_config(os.path.join(args.config_dir, "high_quality_dit.yaml"))
paint_config = load_config(os.path.join(args.config_dir, "high_quality_paint.yaml"))

# Set defaults IF NOT present in the config files.  Command-line args STILL override.
if 'num_inference_steps' not in dit_config:
    dit_config['num_inference_steps'] = args.dit_steps
if 'guidance_scale' not in dit_config:
    dit_config['guidance_scale'] = args.dit_guidance
if 'num_inference_steps' not in paint_config:
    paint_config['num_inference_steps'] = args.paint_steps
if 'guidance_scale' not in paint_config:
    paint_config['guidance_scale'] = args.paint_guidance

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise RuntimeError("CUDA is required for this script.")

 # --- Model Loading (FP16) ---
 # Load the pipelines with MINIMAL arguments.
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', subfolder="hunyuan3d-dit-v2-0",  torch_dtype=torch.float16).to(device)  # Use FP16
texture_pipeline = HunyuanPaintPipeline.from_pretrained('tencent/Hunyuan3D-2', subfolder="hunyuan3d-paint-v2-0", torch_dtype=torch.float16).to(device)
delight_pipeline = DiffusionPipeline.from_pretrained('tencent/Hunyuan3D-2', subfolder="hunyuan3d-delight-v2-0", torch_dtype=torch.float16).to(device) # Use FP16

# --- Output Directory ---
os.makedirs(args.output_dir, exist_ok=True)


# --- Input Processing ---
def process_input(input_path, is_text):
    if is_text:
        # input_path is treated as the text prompt itself
        return [input_path]
    else:
        if os.path.isfile(input_path):
            return [input_path]
        elif os.path.isdir(input_path):
            # Find all image files in the directory
            image_files = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):  # Add other extensions if needed
                image_files.extend(glob.glob(os.path.join(input_path, ext)))
            return image_files
        else:
            print(f"Error: Input path '{input_path}' is not a file or directory.")
            exit(1)

inputs = process_input(args.input, args.text)

# --- Main Generation Loop ---
for i, input_item in enumerate(tqdm(inputs, desc="Generating 3D Models")):
    start_time = time.time()
    try:
        # --- Shape Generation ---
        with torch.cuda.amp.autocast():  # Use autocast for mixed precision
            mesh = shape_pipeline(image=input_item if args.text else input_item,
                                    octree_resolution=args.octree_resolution,
                                    mc_level=args.mc_level,
                                    num_inference_steps=args.dit_steps,
                                    guidance_scale=args.dit_guidance)[0]
        #mesh.export(os.path.join(args.output_dir, f"shape_{i}.obj")) # Optional: Save intermediate

        # --- Texture Generation ---
        with torch.cuda.amp.autocast():  # Use autocast for mixed precision
             textured_mesh = texture_pipeline(mesh,
                                             image=input_item if args.text else input_item,
                                             texture_resolution=args.texture_resolution,
                                             num_inference_steps=args.paint_steps,
                                             guidance_scale=args.paint_guidance)
        # --- Relighting with Delight ---
        with torch.cuda.amp.autocast():  # Use autocast for mixed precision
             textured_mesh = delight_pipeline(textured_mesh, num_inference_steps=args.delight_steps, guidance_scale=args.delight_guidance)


        # --- Save the Textured Mesh ---
        output_base = f"output_{'text' if args.text else 'image'}_{i}"
        output_filename = os.path.join(args.output_dir, f"{output_base}.{args.format}")

        # Handle Different Output Formats
        if args.format == "glb":
            textured_mesh.export(output_filename)
        elif args.format == "obj":
            with open(output_filename, 'w') as f:
                f.write(trimesh.exchange.obj.export_obj(textured_mesh))
        elif args.format == "stl":
             textured_mesh.export(output_filename, file_type='stl')
        else:
            print(f"Error: Unsupported output format: {args.format}")
            continue # Skip

        print(f"3D model generated and saved as {output_filename} (Time: {time.time() - start_time:.2f}s)")

    except Exception as e:
        print(f"An error occurred processing input {i+1}: {e}")
    finally:
        # Clear GPU memory after each generation.
        torch.cuda.empty_cache()
