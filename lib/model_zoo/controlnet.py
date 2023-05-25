import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import copy
from functools import partial
from contextlib import contextmanager
from lib.model_zoo.common.get_model import get_model, register
from lib.log_service import print_log

from .openaimodel import \
    TimestepEmbedSequential, conv_nd, zero_module, \
    ResBlock, AttentionBlock, SpatialTransformer, \
    Downsample, timestep_embedding

####################
# preprocess depth #
####################

# depth_model = None

# def unload_midas_model():
#     global depth_model
#     if depth_model is not None:
#         depth_model = depth_model.cpu()

# def apply_midas(input_image, a=np.pi*2.0, bg_th=0.1, device='cpu'):
#     import cv2
#     from einops import rearrange
#     from .controlnet_annotators.midas import MiDaSInference
#     global depth_model
#     if depth_model is None:
#         depth_model = MiDaSInference(model_type="dpt_hybrid")
#         depth_model = depth_model.to(device)
    
#     assert input_image.ndim == 3
#     image_depth = input_image
#     with torch.no_grad():
#         image_depth = torch.from_numpy(image_depth).float()
#         image_depth = image_depth.to(device)
#         image_depth = image_depth / 127.5 - 1.0
#         image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
#         depth = depth_model(image_depth)[0]

#         depth_pt = depth.clone()
#         depth_pt -= torch.min(depth_pt)
#         depth_pt /= torch.max(depth_pt)
#         depth_pt = depth_pt.cpu().numpy()
#         depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

#         depth_np = depth.cpu().numpy()
#         x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
#         y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
#         z = np.ones_like(x) * a
#         x[depth_pt < bg_th] = 0
#         y[depth_pt < bg_th] = 0
#         normal = np.stack([x, y, z], axis=2)
#         normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
#         normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

#         return depth_image, normal_image


@register('controlnet')
class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if disable_self_attentions is not None:
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (num_attention_blocks is None) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

    def get_device(self):
        return self.time_embed[0].weight.device

    def get_dtype(self):
        return self.time_embed[0].weight.dtype

    def preprocess(self, x, type='canny', **kwargs):
        import torchvision.transforms as tvtrans
        if isinstance(x, str):
            import PIL.Image
            device, dtype = self.get_device(), self.get_dtype()
            x_list = [PIL.Image.open(x)]
        elif isinstance(x, torch.Tensor):
            x_list = [tvtrans.ToPILImage()(xi) for xi in x]
            device, dtype = x.device, x.dtype
        else:
            assert False

        if type == 'none' or type is None:
            return None

        elif type in ['input', 'shuffle_v11e']:
            y_torch = torch.stack([tvtrans.ToTensor()(xi) for xi in x_list])
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type in ['canny', 'canny_v11p']:
            low_threshold = kwargs.pop('low_threshold', 100)
            high_threshold = kwargs.pop('high_threshold', 200)
            from .controlnet_annotator.canny import apply_canny
            y_list = [apply_canny(np.array(xi), low_threshold, high_threshold) for xi in x_list]
            y_torch = torch.stack([tvtrans.ToTensor()(yi) for yi in y_list])
            y_torch = y_torch.repeat(1, 3, 1, 1) # Make is RGB
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type == 'depth':
            from .controlnet_annotator.midas import apply_midas
            y_list, _ = zip(*[apply_midas(input_image=np.array(xi), a=np.pi*2.0, device=device) for xi in x_list])
            y_torch = torch.stack([tvtrans.ToTensor()(yi) for yi in y_list])
            y_torch = y_torch.repeat(1, 3, 1, 1) # Make is RGB
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type in ['hed', 'softedge_v11p']:
            from .controlnet_annotator.hed import apply_hed
            y_list = [apply_hed(np.array(xi), device=device) for xi in x_list]
            y_torch = torch.stack([tvtrans.ToTensor()(yi) for yi in y_list])
            y_torch = y_torch.repeat(1, 3, 1, 1) # Make is RGB
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type in ['mlsd', 'mlsd_v11p']:
            thr_v = kwargs.pop('thr_v', 0.1)
            thr_d = kwargs.pop('thr_d', 0.1)
            from .controlnet_annotator.mlsd import apply_mlsd
            y_list = [apply_mlsd(np.array(xi), thr_v=thr_v, thr_d=thr_d, device=device) for xi in x_list]
            y_torch = torch.stack([tvtrans.ToTensor()(yi) for yi in y_list])
            y_torch = y_torch.repeat(1, 3, 1, 1) # Make is RGB
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type == 'normal':
            bg_th = kwargs.pop('bg_th', 0.4)
            from .controlnet_annotator.midas import apply_midas
            _, y_list = zip(*[apply_midas(input_image=np.array(xi), a=np.pi*2.0, bg_th=bg_th, device=device) for xi in x_list])
            y_torch = torch.stack([tvtrans.ToTensor()(yi.copy()) for yi in y_list])
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type in ['openpose', 'openpose_v11p']:
            from .controlnet_annotator.openpose import OpenposeModel
            from functools import partial
            wrapper = OpenposeModel()
            apply_openpose = partial(
                wrapper.run_model, include_body=True, include_hand=False, include_face=False, 
                json_pose_callback=None, device=device)
            y_list = [apply_openpose(np.array(xi)) for xi in x_list]
            y_torch = torch.stack([tvtrans.ToTensor()(yi.copy()) for yi in y_list])
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type in ['openpose_withface', 'openpose_withface_v11p']:
            from .controlnet_annotator.openpose import OpenposeModel
            from functools import partial
            wrapper = OpenposeModel()
            apply_openpose = partial(
                wrapper.run_model, include_body=True, include_hand=False, include_face=True, 
                json_pose_callback=None, device=device)
            y_list = [apply_openpose(np.array(xi)) for xi in x_list]
            y_torch = torch.stack([tvtrans.ToTensor()(yi.copy()) for yi in y_list])
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type in ['openpose_withfacehand', 'openpose_withfacehand_v11p']:
            from .controlnet_annotator.openpose import OpenposeModel
            from functools import partial
            wrapper = OpenposeModel()
            apply_openpose = partial(
                wrapper.run_model, include_body=True, include_hand=True, include_face=True, 
                json_pose_callback=None, device=device)
            y_list = [apply_openpose(np.array(xi)) for xi in x_list]
            y_torch = torch.stack([tvtrans.ToTensor()(yi.copy()) for yi in y_list])
            y_torch = y_torch.to(device).to(torch.float32)
            return y_torch

        elif type == 'scribble':
            method = kwargs.pop('method', 'pidinet')

            import cv2
            def nms(x, t, s):
                x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)
                f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
                f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
                f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
                f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
                y = np.zeros_like(x)
                for f in [f1, f2, f3, f4]:
                    np.putmask(y, cv2.dilate(x, kernel=f) == x, x)
                z = np.zeros_like(y, dtype=np.uint8)
                z[y > t] = 255
                return z

            def make_scribble(result):
                result = nms(result, 127, 3.0)
                result = cv2.GaussianBlur(result, (0, 0), 3.0)
                result[result > 4] = 255
                result[result < 255] = 0
                return result

            if method == 'hed':
                from .controlnet_annotator.hed import apply_hed
                y_list = [apply_hed(np.array(xi), device=device) for xi in x_list]
                y_list = [make_scribble(yi) for yi in y_list]
                y_torch = torch.stack([tvtrans.ToTensor()(yi) for yi in y_list])
                y_torch = y_torch.repeat(1, 3, 1, 1) # Make is RGB
                y_torch = y_torch.to(device).to(torch.float32)
                return y_torch
            
            elif method == 'pidinet':
                from .controlnet_annotator.pidinet import apply_pidinet
                y_list = [apply_pidinet(np.array(xi), device=device) for xi in x_list]
                y_list = [make_scribble(yi) for yi in y_list]
                y_torch = torch.stack([tvtrans.ToTensor()(yi) for yi in y_list])
                y_torch = y_torch.repeat(1, 3, 1, 1) # Make is RGB
                y_torch = y_torch.to(device).to(torch.float32)
                return y_torch

            elif method == 'xdog':
                threshold = kwargs.pop('threshold', 32)
                def apply_scribble_xdog(img):
                    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
                    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
                    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
                    result = np.zeros_like(img, dtype=np.uint8)
                    result[2 * (255 - dog) > threshold] = 255
                    return result

                y_list = [apply_scribble_xdog(np.array(xi), device=device) for xi in x_list]
                y_torch = torch.stack([tvtrans.ToTensor()(yi) for yi in y_list])
                y_torch = y_torch.repeat(1, 3, 1, 1) # Make is RGB
                y_torch = y_torch.to(device).to(torch.float32)
                return y_torch

            else:
                raise ValueError

        elif type == 'seg':
            method = kwargs.pop('method', 'ufade20k')
            if method == 'ufade20k':
                from .controlnet_annotator.uniformer import apply_uniformer
                y_list = [apply_uniformer(np.array(xi), palette='ade20k', device=device) for xi in x_list]
                y_torch = torch.stack([tvtrans.ToTensor()(yi) for yi in y_list])
                y_torch = y_torch.to(device).to(torch.float32)
                return y_torch

            else:
                raise ValueError
