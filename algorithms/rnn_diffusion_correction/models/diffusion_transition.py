from typing import Optional, Tuple, Union
import numpy as np
from random import random, randint
from collections import namedtuple
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .vdt import VDT
from .u_net3d import Unet3D
from .utils import extract, default, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start", "model_out"])


class DiffusionCorrectionransitionModel(nn.Module):
    backbones: Dict[str, nn.Module] = {
        "unet3d": Unet3D,
        "vdt": VDT,
    }
    def __init__(self, x_shape, z_shape, external_cond_dim, cfg, backbone_cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone_cfg = backbone_cfg
        self.x_shape = x_shape
        self.z_shape = z_shape
        self.external_cond_dim = external_cond_dim
        self.mask_unet = cfg.mask_unet
        self.num_gru_layers = cfg.num_gru_layers
        self.num_mlp_layers = cfg.num_mlp_layers
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.beta_schedule = cfg.beta_schedule
        self.objective = cfg.objective
        self.use_snr = cfg.use_snr
        self.use_cum_snr = cfg.use_cum_snr
        self.snr_clip = cfg.snr_clip
        self.cum_snr_decay = cfg.cum_snr_decay
        self.ddim_sampling_eta = cfg.ddim_sampling_eta
        self.clip_noise = cfg.clip_noise
        self.p2_loss_weight_gamma = cfg.p2_loss_weight_gamma
        self.p2_loss_weight_k = cfg.p2_loss_weight_k
        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        self.self_condition = cfg.self_condition
        self.network_size = cfg.network_size
        self.return_all_timesteps = cfg.return_all_timesteps
        self.noise_level_sampling = cfg.noise_level_sampling
        self.max_noise_level_gap = cfg.max_noise_level_gap # Only used when noise_level_sampling is linear_increasing
        assert self.noise_level_sampling in ["random", "linear_increasing", "constant"]

        if self.objective not in ["pred_noise", "pred_x0", "pred_v"]:
            raise ValueError("objective must be either pred_noise or pred_x0 or pred_v ")

        self._build_model()
        self._build_buffer()

    def _build_model(self):
        x_channel = self.x_shape[0]
        z_channel = self.z_shape[0]
        self.model = None
        if len(self.x_shape) == 3:
            backbone = self.backbones[self.backbone_cfg.name]
            self.model = backbone(
                cfg=self.backbone_cfg,
                input_size=self.x_shape[-1],
                in_channels=x_channel,
                z_channels=z_channel,
            )

        elif len(self.x_shape) == 1:
            # self.model = TransitionMlp(
            #     z_dim=z_channel,
            #     x_dim=x_channel,
            #     external_cond_dim=self.external_cond_dim,
            #     network_size=self.network_size,
            #     num_gru_layers=self.num_gru_layers,
            #     num_mlp_layers=self.num_mlp_layers,
            #     self_condition=self.self_condition,
            # )

            # self.x_from_z = nn.Linear(z_channel, x_channel)
            raise NotImplementedError("3D x_shape is not supported yet")
        else:
            raise ValueError(f"x_shape must have 1 or 3 dims but got shape {self.x_shape}")

    def _build_buffer(self):
        if self.beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif self.beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif self.beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {self.beta_schedule}")

        betas = beta_schedule_fn(self.timesteps, **self.schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # calculate p2 reweighting

        register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -self.p2_loss_weight_gamma,
        )

        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        clipped_snr = snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)

        register_buffer("clipped_snr", clipped_snr)
        register_buffer("snr", snr)

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        deterministic_t: Optional[Union[float, int]] = None,
        cum_snr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param z: z at current time step that generates x (z_t -> x_t)
        :param x: ground truth x
        :param external_cond: external_cond to be conditioned on
        :param deterministic_t: set a noise level t directly instead of sampling.
        :param cum_snr: cumulative snr for previous time steps

        :return: z_next_pred, x_next_pred, loss(unreduced), cum_snr
        """
        B, T = z.shape[:2]
        if deterministic_t is None:
            # t = torch.randint(0, self.num_timesteps, (B,T), device=z.device).long()
            t = self.sample_noise_level(B, T, z.device)
        elif isinstance(deterministic_t, float):
            deterministic_t = round(deterministic_t * (self.num_timesteps - 1))
            t = torch.full((B,), deterministic_t, device=z.device).long()
        elif isinstance(deterministic_t, int):
            deterministic_t = deterministic_t if deterministic_t >= 0 else self.timesteps + deterministic_t
            t = torch.full((B,), deterministic_t, device=z.device).long()

        # get noised version of x
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        noised_x = self.q_sample(x_start=x, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(noised_x, z, t, external_cond=external_cond).pred_x_start
                x_self_cond.detach()

        model_pred = self.model_predictions(noised_x, z, t, external_cond=external_cond, x_self_cond=x_self_cond)
        x_pred = model_pred.pred_x_start

        pred = model_pred.model_out

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, t, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred, target.detach(), reduction="none")

        # normalized_clipped_snr = self.clipped_snr[t] / self.snr_clip
        # normalized_snr = self.snr[t] / self.snr_clip

        # if cum_snr is None or not self.use_cum_snr:
        #     cum_snr_next = normalized_clipped_snr
        #     clipped_fused_snr = normalized_clipped_snr
        #     fused_snr = normalized_snr
        # else:
        #     cum_snr_next = cum_snr * self.cum_snr_decay + normalized_clipped_snr * (1 - self.cum_snr_decay)
        #     clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_clipped_snr)
        #     fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (1 - normalized_snr)

        # if self.use_snr:
        #     if self.objective == "pred_noise":
        #         loss_weight = clipped_fused_snr / fused_snr
        #     elif self.objective == "pred_x0":
        #         loss_weight = clipped_fused_snr * self.snr_clip
        #     elif self.objective == "pred_v":
        #         loss_weight = clipped_fused_snr * self.snr_clip / (fused_snr * self.snr_clip + 1)

        #     loss_weight = loss_weight.view(B, *((1,) * (len(loss.shape) - 1)))
        # elif self.use_cum_snr and cum_snr is not None:
        #     loss_weight = cum_snr * self.snr_clip
        #     loss_weight = loss_weight.view(B, *((1,) * (len(loss.shape) - 1)))
        # else:
        #     loss_weight = torch.ones_like(loss)
        #     loss_weight *= self.snr_clip * 0.5  # multiply by a constant so weight scale is similar to snr

        # loss = loss * loss_weight

        # return x_pred, loss, cum_snr_next
        return x_pred, loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, z_cond, t, external_cond=None, x_self_cond=None) -> ModelPrediction:
        #z_next = self.model(x, t, z_cond, external_cond, x_self_cond)
        model_output = self.model(x, z_cond, t.float()) #TODO: add external_cond, x_self_cond

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start, model_output)

    def p_mean_variance(self, x, t, z_cond, external_cond=None, x_self_cond=None):
        model_pred = self.model_predictions(x, t, z_cond, external_cond=external_cond, x_self_cond=x_self_cond)
        x_start = model_pred.pred_x_start
        # pred_z = model_pred.pred_z

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # @torch.no_grad()
    def p_sample(self, x, t, z_cond, external_cond=None, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x, batched_times, z_cond, external_cond=external_cond, x_self_cond=x_self_cond
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_x, x_start

    # @torch.no_grad()
    def p_sample_loop(self, shape, z_cond, external_cond=None, return_all_timesteps=False):
        batch, device = shape[0], self.betas.device

        x = torch.randn(shape, device=device)
        x = torch.clamp(x, -self.clip_noise, self.clip_noise)
        xs = [x]

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            x, x_start = self.p_sample(x, t, z_cond, external_cond, x_self_cond=self_cond)
            xs.append(x)

        ret = x if not return_all_timesteps else xs

        return ret

    # @torch.no_grad()
    def ddim_sample(self, shape, z_cond, external_cond=None, return_all_timesteps=False):
        batch, timesteps, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            shape[1],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        x = torch.clamp(x, -self.clip_noise, self.clip_noise)
        xs = [x]

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch, timesteps), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            #TODO: edit here, no return z
            model_pred = self.model_predictions(
                x, z_cond, time_cond, external_cond=external_cond, x_self_cond=self_cond
            )
            pred_noise, x_start = model_pred

            if time_next < 0:
                x = x_start
                xs.append(x)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(x)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            xs.append(x)

        ret = x if not return_all_timesteps else xs

        return ret

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def ddim_sample_step(
        self, 
        x, # (B, T, C, H, W)
        z_cond, # (B, T, Z, H, W)
        index: np.ndarray, # (T) 
        external_cond=None, 
        return_x_start=False, 
        return_guidance_const=False
    ):
        # clamp noise where x is pure noise
        mask = (torch.tensor(index) == 0).view(1, -1, *(1,)*(x.dim()-2)).to(x.device)
        x = torch.where(mask, torch.clamp(x, -self.clip_noise, self.clip_noise), x)
        # if index == 0:
        #     x = torch.clamp(x, -self.clip_noise, self.clip_noise)

        batch, n_frames, device, total_timesteps, sampling_timesteps, eta = (
            x.shape[0],
            x.shape[1],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = np.array(list(zip(times[:-1], times[1:])))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = x.to(device)

        time, time_next = [v for v in time_pairs[index].T]
        # time, time_next = time_pairs[index]
        # print('time', time, 'time_next', time_next)
        if type(time) in [int, float, np.int64]:
            time_cond = torch.full((batch, n_frames), time, device=device, dtype=torch.long)
        else:
            time_cond = torch.tensor(time, device=device, dtype=torch.long)
            time_cond = repeat(time_cond, 't -> b t', b=batch)

        model_pred = self.model_predictions(x, z_cond, time_cond, external_cond=external_cond)
        pred_noise = model_pred.pred_noise
        x_start = model_pred.pred_x_start

        guidance_scale = 0
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        # Eta = 0 => Deterministic sampling
        # Eta = 1 => DDPM
        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma**2).sqrt()
        guidance_scale = (1 - alpha) - c * (1 - alpha).sqrt()  # (t)

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        # For broadcasting multiply
        alpha_next = alpha_next.view(1, -1, *(1,)*(x_start.dim()-2))
        c = c.view(1, -1, *(1,)*(x_start.dim()-2))
        sigma = sigma.view(1, -1, *(1,)*(x_start.dim()-2))

        x_tm1 = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        mask = (torch.tensor(time_next) < 0).view(1, -1, *(1,)*(x.dim()-2)).to(x.device)
        x = torch.where(mask, x_start, x_tm1)

        result = [x]
        if return_x_start:
            result.append(x_start)
        if return_guidance_const:
            result.append(guidance_scale)
        return result

    def sample_noise_level(self, b, t, device):
        if self.noise_level_sampling == "random":
            return torch.randint(0, self.num_timesteps, (b,t), device=device).long()
        elif self.noise_level_sampling == "linear_increasing":
            first_noise_level = randint(0, self.num_timesteps - 1)
            noise_level_gap = randint(1, self.max_noise_level_gap+1)
            noise_levels = torch.clamp(torch.arange(first_noise_level, first_noise_level + t*noise_level_gap, noise_level_gap, device=device), 0, self.num_timesteps-1)
            return repeat(noise_levels, 't -> b t', b=b).long()
        elif self.noise_level_sampling == "constant":
            noise_level = randint(0, self.num_timesteps - 1)
            return torch.full((b,t), noise_level, device=device).long()
        else:
            raise ValueError(f"unknown noise level sampling {self.noise_level_sampling}")