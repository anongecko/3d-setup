from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.utils.checkpoint
from PIL import Image
from diffusers import AutoencoderKL, DiffusionPipeline, ImagePipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline, retrieve_timesteps, rescale_noise_cfg
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging
from einops import rearrange
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import importlib  # Needed for instantiate_from_config
import yaml  # Needed for config loading
import os  # needed for paths
import glob
import json  # Needed for the main config
import logging

# Corrected import statement:
from .hunyuanpaint.unet.modules import UNet2p5DConditionModel

logger = logging.getLogger(__name__)


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == "RGB":
        return maybe_rgba
    elif maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class Hunyuan3DTexGenConfig:  # This is where the delight model would have been used
    def __init__(self, model_path, multiview_model_path=None):
        self.model_path = model_path
        self.multiview_model_path = multiview_model_path
        self.multiview_model = None  # Never used in the code, can remove if you don't need it
        self.device = "cuda"
        self.dtype = torch.float16  # Keep this as 16.


class HunyuanPaintPipeline(StableDiffusionPipeline):
    @classmethod
    def from_pretrained(cls, model_path, device="cuda", dtype=torch.float16, use_safetensors=None, variant=None, subfolder=None, **kwargs):
        original_model_path = model_path
        base_dir = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")

        extension = "ckpt" if not use_safetensors else "safetensors"
        variant = "" if variant is None else f".{variant}"

        # Construct the initial model path *without* subfolder.
        model_path = os.path.expanduser(os.path.join(base_dir, "models--" + model_path.replace("/", "--"), "snapshots", "*"))

        # Use glob to find the snapshot directory (handles the hash).
        matches = glob.glob(model_path)
        if not matches:
            # If path not available, attempt download
            print("Paint Model path not exists, trying to download from huggingface")
            try:
                import huggingface_hub

                path = huggingface_hub.snapshot_download(repo_id=original_model_path)
                # Use the downloaded path directly.
                model_path = path

            except ImportError:
                logger.warning("You need to install HuggingFace Hub to load models from the hub.")
                raise RuntimeError(f"Paint Model path {model_path} not found.")
            except Exception as e:
                raise e
        else:
            model_path = matches[0]  # Use the found snapshot directory

        # --- KEY CHANGE:  Adjust paths for Paint pipeline to always use top-level config.json ---
        # For Paint pipeline, config.json is always at the top level, even with subfolder.
        config_path = os.path.join(model_path, "config.json") # JSON at top level
        if subfolder is not None: # Keep model_path adjustment for subfolder for other files
            model_path = os.path.join(model_path, subfolder)
        ckpt_name = f"model{variant}.{extension}"
        ckpt_path = os.path.join(model_path, ckpt_name)
        # --- END KEY CHANGE ---

        print("Paint config:", config_path)
        print("Paint Checkpoint:", ckpt_path)
        # Now do the loading as in the shapegen
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)  #

        if use_safetensors:
            ckpt_path = ckpt_path.replace(".ckpt", ".safetensors")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Paint Model file {ckpt_path} not found")
        logger.info(f"Loading Paint model from {ckpt_path}")

        if use_safetensors:
            # parse safetensors
            import safetensors.torch

            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device="cpu")
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split(".")[0]
                new_key = key[len(model_name) + 1 :]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                    ckpt[model_name][new_key] = value
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")

        # load model
        unet = instantiate_from_config(config["unet"])
        unet.load_state_dict(ckpt["unet"])
        vae = instantiate_from_config(config["vae"])
        vae.load_state_dict(ckpt["vae"])
        text_encoder = instantiate_from_config(config["text_encoder"])
        if "text_encoder" in ckpt:
            text_encoder.load_state_dict(ckpt["text_encoder"])
        tokenizer = instantiate_from_config(config["tokenizer"])
        feature_extractor = instantiate_from_config(config["feature_extractor"])
        scheduler = instantiate_from_config(config["scheduler"])

        model_kwargs = dict(vae=vae, unet=unet, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor)
        model_kwargs.update(kwargs)

        return cls(**model_kwargs)

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2p5DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
        safety_checker=None,
        use_torch_compile=False,
    ):
        super().__init__()  # Use super() for a cleaner call

        safety_checker = None  # As per the original code
        self.register_modules(
            vae=torch.compile(vae) if use_torch_compile else vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=torch.compile(feature_extractor) if use_torch_compile else feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.no_grad()
    def encode_images(self, images):
        B = images.shape[0]
        images = rearrange(images, "b n c h w -> (b n) c h w")

        dtype = next(self.vae.parameters()).dtype
        images = (images - 0.5) * 2.0
        posterior = self.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        latents = rearrange(latents, "(b n) c h w -> b n c h w", b=B)
        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image = None,
        prompt=None,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=2.0,
        output_type: Optional[str] = "pil",
        num_inference_steps=28,
        return_dict=True,
        texture_resolution: int = 2048,  # ADDED TEXTURE RESOLUTION
        **cached_condition,
    ):
        if image is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(image, torch.Tensor)

        image = to_rgb_image(image)

        image_vae = torch.tensor(np.array(image) / 255.0, dtype=self.vae.dtype, device=self.vae.device)  # Combined for efficiency
        image_vae = image_vae.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)

        batch_size = image_vae.shape[0]
        assert batch_size == 1
        assert num_images_per_prompt == 1

        ref_latents = self.encode_images(image_vae)

        def convert_pil_list_to_tensor(images):
            bg_c = [1.0, 1.0, 1.0]
            images_tensor = []
            for batch_imgs in images:
                view_imgs = []
                for pil_img in batch_imgs:
                    img = np.asarray(pil_img, dtype=np.float32) / 255.0
                    if img.shape[2] > 3:
                        alpha = img[:, :, 3:]
                        img = img[:, :, :3] * alpha + bg_c * (1 - alpha)
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().to(dtype=torch.float16, device="cuda")  # half().to("cuda")
                    view_imgs.append(img)
                view_imgs = torch.cat(view_imgs, dim=0)
                images_tensor.append(view_imgs.unsqueeze(0))

            images_tensor = torch.cat(images_tensor, dim=0)
            return images_tensor

        for key in ("normal_imgs", "position_imgs"):
            if key in cached_condition:
                if isinstance(cached_condition[key], List):
                    cached_condition[key] = convert_pil_list_to_tensor(cached_condition[key])
                cached_condition[key] = self.encode_images(cached_condition[key])

        for key in ("camera_info_gen", "camera_info_ref"):
            if key in cached_condition:
                camera_info = cached_condition[key]
                if isinstance(camera_info, List):
                    camera_info = torch.tensor(camera_info)
                camera_info = camera_info.to(image_vae.device).to(torch.int64)
                cached_condition[key] = camera_info

        cached_condition["ref_latents"] = ref_latents

        if guidance_scale > 1:
            negative_ref_latents = torch.zeros_like(cached_condition["ref_latents"])
            cached_condition["ref_latents"] = torch.cat([negative_ref_latents, cached_condition["ref_latents"]])
            cached_condition["ref_scale"] = torch.as_tensor([0.0, 1.0]).to(cached_condition["ref_latents"].device)  # Corrected device placement
            for key in ("normal_imgs", "position_imgs", "position_maps", "camera_info_gen", "camera_info_ref"):
                if key in cached_condition:
                    cached_condition[key] = torch.cat((cached_condition[key], cached_condition[key]))

        prompt_embeds = self.unet.learned_text_clip_gen.repeat(num_images_per_prompt, 1, 1)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        if not hasattr(self, "device") or self.device is None:
            self.device = self._execution_device
        if not hasattr(self, "dtype") or self.dtype is None:
            self.dtype = next(self.parameters()).dtype
        latents: torch.Tensor = self.denoise(
            None,
            *args,
            cross_attention_kwargs=None,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            width=self.unet.config.sample_size * self.vae_scale_factor,  # Default size for latents
            height=self.unet.config.sample_size * self.vae_scale_factor,  # Default size for latents
            **cached_condition,
        ).images

        if not output_type == "latent":
            # --- CRITICAL CHANGE HERE ---
            # Decode to the specified texture_resolution
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=None)[0]
            image = torch.nn.functional.interpolate(
                image,
                size=(texture_resolution, texture_resolution),  # Resize to target resolution
                mode="bilinear",  # Use bilinear interpolation
                align_corners=False,
            )
            # --- END CRITICAL CHANGE ---
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def denoise(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation. (Documentation omitted for brevity)
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        # Get lora scale from cross attention kwargs, if it exists.
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
        assert num_images_per_prompt == 1
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * kwargs["num_in_batch"],  # num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) else None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latents = rearrange(latents, "(b n) c h w -> b n c h w", n=kwargs["num_in_batch"])
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = rearrange(latent_model_input, "b n c h w -> (b n) c h w")
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = rearrange(latent_model_input, "(b n) c h w ->b n c h w", n=kwargs["num_in_batch"])

                # predict the noise residual

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    **kwargs,
                )[0]
                latents = rearrange(latents, "b n c h w -> (b n) c h w")
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents[:, :num_channels_latents, :, :], **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image = torch.nn.functional.interpolate(
                image,
                size=(texture_resolution, texture_resolution),  # Resize to target resolution
                mode="bilinear",  # Use bilinear interpolation
                align_corners=False,
            )
            image, has_nsfw_concept = self.run_safety_checker(image, self.device, prompt_embeds.dtype)  # self.device
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def get_timesteps(self, num_inference_steps, strength, device):
        return retrieve_timesteps(self.scheduler, num_inference_steps, device=device)

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_image(self, image):
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]
        image_pts = []
        mask_pts = []
        for img in image:
            image_pt, mask_pt = self.image_processor(img, return_mask=True)
            image_pts.append(image_pt)
            mask_pts.append(mask_pt)

        image_pts = torch.cat(image_pts, dim=0).to(self.device, dtype=self.dtype)
        if mask_pts[0] is not None:
            mask_pts = torch.cat(mask_pts, dim=0).to(self.device, dtype=self.dtype)
        else:
            mask_pts = None
        return image_pts, mask_pts

    # Re-added check inputs, which was necessary for StableDiffusionPipeline
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(v in self._callback_tensor_inputs for v in callback_on_step_end_tensor_inputs):
            raise ValueError(f"`callback_on_step_end_tensor_inputs` has to be a subset of {self._callback_tensor_inputs}, but found {callback_on_step_end_tensor_inputs}")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt only forward one of the two.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_timesteps(self, num_inference_steps, strength, device):
        return retrieve_timesteps(self.scheduler, num_inference_steps, device=device)

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))