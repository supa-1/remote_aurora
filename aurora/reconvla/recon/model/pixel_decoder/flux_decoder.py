import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class FluxDecoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.pixel_decoder = AutoencoderKL.from_pretrained(config.mm_pixel_decoder) 
        # my change for the recommend vae error
        # try:
        #     self.pixel_decoder = AutoencoderKL.from_pretrained(config.mm_pixel_decoder)
        # except ValueError as e:
        #     # Some local VAE checkpoints are partially compatible and require non-low-mem loading.
        #     if "keys are missing" not in str(e):
        #         raise
        #     self.pixel_decoder = AutoencoderKL.from_pretrained(
        #         config.mm_pixel_decoder,
        #         low_cpu_mem_usage=False,
        #         device_map=None,
        #     )
            
        self.pixel_decoder.requires_grad_(False)
        self.pixel_decoder.float()
        self.pixel_decoder.eval() 

    @property
    def scaling_factor(self):
        return self.pixel_decoder.config.scaling_factor

    @property
    def shift_factor(self):
        return self.pixel_decoder.config.shift_factor

    @property
    def latent_dim(self):
        return self.pixel_decoder.config.latent_channels * 4 # 4 is the upsampling factor of the decoder（VAE）

    def encode(self, x):
        # Keep VAE in fp32 for numerical stability and avoid fp16 autocast mismatch.
        x = x.to(dtype=torch.float32)
        vae_device = next(self.pixel_decoder.parameters()).device
        if vae_device != x.device:
            self.pixel_decoder.to(x.device)
        if x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self.pixel_decoder.encode(x)
        return self.pixel_decoder.encode(x)

    def decode(self, z):
        z = z.to(dtype=torch.float32)
        vae_device = next(self.pixel_decoder.parameters()).device
        if vae_device != z.device:
            self.pixel_decoder.to(z.device)
        if z.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self.pixel_decoder.decode(z)
        return self.pixel_decoder.decode(z)
