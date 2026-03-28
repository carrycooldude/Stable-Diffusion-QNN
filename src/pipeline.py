import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from .scheduler import EulerDiscreteScheduler
from .utils import quantize, dequantize, postprocess_image

# Quantization Constants (from metadata.yaml)
Q_TEXT_EMB_S, Q_TEXT_EMB_ZP = 0.0003629151, 25915
Q_UNET_LATENT_S, Q_UNET_LATENT_ZP = 0.0003279145, 34760
Q_UNET_TIMESTEP_S, Q_UNET_TIMESTEP_ZP = 0.014770733, 0
Q_UNET_TEXT_EMB_S, Q_UNET_TEXT_EMB_ZP = 0.0003463204, 23638
Q_UNET_OUT_LATENT_S, Q_UNET_OUT_LATENT_ZP = 0.0001840945, 30388
Q_VAE_LATENT_S, Q_VAE_LATENT_ZP = 0.0003278456, 34752

class StableDiffusionQNNPipeline:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = Tokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.scheduler = EulerDiscreteScheduler()
        self._init_sessions()

    def _init_sessions(self):
        provider = "QNNExecutionProvider"
        provider_options = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        }
        session_options = ort.SessionOptions()

        self.text_enc = ort.InferenceSession(
            os.path.join(self.model_dir, "text_encoder.onnx"), 
            session_options, providers=[provider], provider_options=[provider_options]
        )
        self.unet = ort.InferenceSession(
            os.path.join(self.model_dir, "unet.onnx"), 
            session_options, providers=[provider], provider_options=[provider_options]
        )
        self.vae = ort.InferenceSession(
            os.path.join(self.model_dir, "vae.onnx"), 
            session_options, providers=[provider], provider_options=[provider_options]
        )

    def encode_text(self, text):
        tokens = self.tokenizer.encode(text)
        token_ids = tokens.ids
        if len(token_ids) > 77:
            token_ids = token_ids[:77]
        else:
            token_ids = token_ids + [49407] * (77 - len(token_ids))
        
        input_ids = np.array([token_ids], dtype=np.int32)
        out = self.text_enc.run(None, {"tokens": input_ids})[0]
        return dequantize(out, Q_TEXT_EMB_S, Q_TEXT_EMB_ZP)

    def generate(self, prompt, neg_prompt="", num_steps=20, guidance_scale=7.5, seed=42):
        np.random.seed(seed)
        
        # 1. Encode text
        prompt_emb = self.encode_text(prompt)
        neg_emb = self.encode_text(neg_prompt)
        text_embeddings = np.concatenate([neg_emb, prompt_emb])

        # 2. Prepare latents
        latents = np.random.randn(1, 64, 64, 4).astype(np.float32)
        timesteps = self.scheduler.set_timesteps(num_steps)
        latents = latents * self.scheduler.get_sigma(0)

        # 3. Denoising loop
        for i, t in enumerate(timesteps):
            sigma = self.scheduler.get_sigma(i)
            input_latents = latents / ((sigma**2 + 1)**0.5)
            
            # Unconditional pass
            q_latent = quantize(input_latents, Q_UNET_LATENT_S, Q_UNET_LATENT_ZP)
            q_t = quantize(np.array([[t]]), Q_UNET_TIMESTEP_S, Q_UNET_TIMESTEP_ZP)
            q_emb_uncond = quantize(text_embeddings[0:1], Q_UNET_TEXT_EMB_S, Q_UNET_TEXT_EMB_ZP)
            
            noise_uncond_q = self.unet.run(None, {"latent": q_latent, "timestep": q_t, "text_emb": q_emb_uncond})[0]
            noise_uncond = dequantize(noise_uncond_q, Q_UNET_OUT_LATENT_S, Q_UNET_OUT_LATENT_ZP)
            
            # Conditional pass
            q_emb_cond = quantize(text_embeddings[1:2], Q_UNET_TEXT_EMB_S, Q_UNET_TEXT_EMB_ZP)
            noise_cond_q = self.unet.run(None, {"latent": q_latent, "timestep": q_t, "text_emb": q_emb_cond})[0]
            noise_cond = dequantize(noise_cond_q, Q_UNET_OUT_LATENT_S, Q_UNET_OUT_LATENT_ZP)
            
            # Guidance
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, i, latents)
            print(f"Iteration {i+1}/{num_steps} complete")

        # 4. VAE Decoding
        latents = latents / 0.18215
        q_vae_latent = quantize(latents, Q_VAE_LATENT_S, Q_VAE_LATENT_ZP)
        image_data = self.vae.run(None, {"latent": q_vae_latent})[0]
        
        return postprocess_image(image_data)
