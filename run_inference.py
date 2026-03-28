import os
import argparse
from src.pipeline import StableDiffusionQNNPipeline

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion v2.1 on Qualcomm NPU (QNN)")
    parser.add_argument("--prompt", type=str, default="A majestic lion in the savanna, cinematic lighting, 4k", help="Text prompt for generation")
    parser.add_argument("--neg_prompt", type=str, default="ugly, blurry, low quality, distorted", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the generated image")
    
    args = parser.parse_args()

    model_dir = r"c:\Users\rawat\Stable-Diffusion-QNN\models\stable_diffusion_v2_1-precompiled_qnn_onnx-w8a16-qualcomm_snapdragon_x_elite"
    
    print("Loading Stable Diffusion QNN Pipeline...")
    pipeline = StableDiffusionQNNPipeline(model_dir)

    print(f"Generating image for prompt: '{args.prompt}'")
    image = pipeline.generate(
        args.prompt, 
        neg_prompt=args.neg_prompt, 
        num_steps=args.steps, 
        guidance_scale=args.guidance, 
        seed=args.seed
    )

    image.save(args.output)
    print(f"Success! Image saved to {args.output}")

if __name__ == "__main__":
    main()
