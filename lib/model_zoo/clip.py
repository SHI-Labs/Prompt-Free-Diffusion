import torch
import torch.nn as nn
import numpy as np
from functools import partial
from lib.model_zoo.common.get_model import register

symbol = 'clip'

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

from transformers import CLIPTokenizer, CLIPTextModel

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

@register('clip_text_context_encoder_sdv1')
class CLIPTextContextEncoderSDv1(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77, freeze=True):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        with torch.no_grad():
            batch_encoding = self.tokenizer(
                text, truncation=True, max_length=self.max_length, return_length=True,
                return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.device)
        max_token_n = self.transformer.text_model.embeddings.position_ids.shape[1]
        positional_ids = torch.arange(max_token_n)[None].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, 
            position_ids=positional_ids, )
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

#############################
# copyed from justin's code #
#############################

@register('clip_image_context_encoder_justin')
class CLIPImageContextEncoderJustin(AbstractEncoder):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model='ViT-L/14',
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        from . import clip_justin
        self.model, _ = clip_justin.load(name=model, device=device, jit=jit)
        self.device = device
        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

        # I didn't call this originally, but seems like it was frozen anyway
        self.freeze()

    def freeze(self):
        self.transformer = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        import kornia
        # Expects inputs in the range -1, 1
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x)).float()

    def encode(self, im):
        return self(im).unsqueeze(1)

###############
# for vd next #
###############

from transformers import CLIPModel

@register('clip_text_context_encoder')
class CLIPTextContextEncoder(AbstractEncoder):
    def __init__(self, 
                 version="openai/clip-vit-large-patch14", 
                 max_length=77, 
                 fp16=False, ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version)
        self.max_length = max_length
        self.fp16 = fp16
        self.freeze()

    def get_device(self):
        # A trick to get device
        return self.model.text_projection.weight.device

    def freeze(self):
        self.model = self.model.eval()
        self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False
        
    def encode(self, text):
        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        outputs = self.model.text_model(input_ids=tokens)
        z = self.model.text_projection(outputs.last_hidden_state)
        z_pooled = self.model.text_projection(outputs.pooler_output)
        z = z / torch.norm(z_pooled.unsqueeze(1), dim=-1, keepdim=True)
        return z

from transformers import CLIPProcessor

@register('clip_image_context_encoder')
class CLIPImageContextEncoder(AbstractEncoder):
    def __init__(self, 
                 version="openai/clip-vit-large-patch14", 
                 fp16=False, ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version)
        self.fp16 = fp16
        self.freeze()

    def get_device(self):
        # A trick to get device
        return self.model.text_projection.weight.device

    def freeze(self):
        self.model = self.model.eval()
        self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def _encode(self, images):
        if isinstance(images, torch.Tensor):
            import torchvision.transforms as tvtrans
            images = [tvtrans.ToPILImage()(i) for i in images]
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs['pixel_values'].half() if self.fp16 else inputs['pixel_values']
        pixels = pixels.to(self.get_device())
        outputs = self.model.vision_model(pixel_values=pixels)
        z = outputs.last_hidden_state
        z = self.model.vision_model.post_layernorm(z)
        z = self.model.visual_projection(z)
        z_pooled = z[:, 0:1]
        z = z / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z

    @torch.no_grad()
    def _encode_wmask(self, images, masks):
        assert isinstance(masks, torch.Tensor)
        assert (len(masks.shape)==4) and (masks.shape[1]==1)
        masks = torch.clamp(masks, 0, 1)
        masked_images = images*masks
        masks = masks.float()
        masks = F.interpolate(masks, [224, 224], mode='bilinear')
        if masks.sum() == masks.numel():
            return self._encode(images)

        device = images.device
        dtype = images.dtype
        gscale = masks.mean(axis=[1, 2, 3], keepdim=True).flatten(2)

        vtoken_kernel_size = self.model.vision_model.embeddings.patch_embedding.kernel_size
        vtoken_stride = self.model.vision_model.embeddings.patch_embedding.stride
        mask_kernal = torch.ones([1, 1, *vtoken_kernel_size], device=device, requires_grad=False).float()
        vtoken_mask = torch.nn.functional.conv2d(masks, mask_kernal, stride=vtoken_stride).flatten(2).transpose(1, 2)
        vtoken_mask = vtoken_mask/np.prod(vtoken_kernel_size)
        vtoken_mask = torch.concat([gscale, vtoken_mask], axis=1)

        import types
        def customized_embedding_forward(self, pixel_values):
            batch_size = pixel_values.shape[0]
            patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            embeddings = embeddings + self.position_embedding(self.position_ids)
            embeddings = embeddings*vtoken_mask.to(embeddings.dtype)
            return embeddings

        old_forward = self.model.vision_model.embeddings.forward
        self.model.vision_model.embeddings.forward = types.MethodType(
            customized_embedding_forward, self.model.vision_model.embeddings)

        z = self._encode(images)
        self.model.vision_model.embeddings.forward = old_forward
        z = z * vtoken_mask.to(dtype)
        return z

    # def _encode_wmask(self, images, masks):
    #     assert isinstance(masks, torch.Tensor)
    #     assert (len(masks.shape)==4) and (masks.shape[1]==1)
    #     masks = torch.clamp(masks, 0, 1)
    #     masks = masks.float()
    #     masks = F.interpolate(masks, [224, 224], mode='bilinear')
    #     if masks.sum() == masks.numel():
    #         return self._encode(images)

    #     device = images.device
    #     dtype = images.dtype

    #     vtoken_kernel_size = self.model.vision_model.embeddings.patch_embedding.kernel_size
    #     vtoken_stride = self.model.vision_model.embeddings.patch_embedding.stride
    #     mask_kernal = torch.ones([1, 1, *vtoken_kernel_size], device=device, requires_grad=False).float()
    #     vtoken_mask = torch.nn.functional.conv2d(masks, mask_kernal, stride=vtoken_stride).flatten(2).transpose(1, 2)
    #     vtoken_mask = vtoken_mask/np.prod(vtoken_kernel_size)

    #     z = self._encode(images)
    #     z[:, 1:, :] = z[:, 1:, :] * vtoken_mask.to(dtype)
    #     z[:, 0, :] = 0
    #     return z

    def encode(self, images, masks=None):
        if masks is None:
            return self._encode(images)
        else:
            return self._encode_wmask(images, masks)

@register('clip_image_context_encoder_position_agnostic')
class CLIPImageContextEncoderPA(CLIPImageContextEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import types
        def customized_embedding_forward(self, pixel_values):
            batch_size = pixel_values.shape[0]
            patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            pembeddings = self.position_embedding(self.position_ids)
            pembeddings = torch.cat([
                pembeddings[:, 0:1], 
                pembeddings[:, 1: ].mean(dim=1, keepdim=True).repeat(1, 256, 1)], dim=1)
            embeddings = embeddings + pembeddings
            return embeddings

        self.model.vision_model.embeddings.forward = types.MethodType(
            customized_embedding_forward, self.model.vision_model.embeddings)

##############
# from sd2.0 #
##############

import open_clip
import torch.nn.functional as F

@register('openclip_text_context_encoder_sdv2')
class FrozenOpenCLIPTextEmbedderSDv2(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

@register('openclip_text_context_encoder')
class FrozenOpenCLIPTextEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    def __init__(self, 
                 arch="ViT-H-14", 
                 version="laion2b_s32b_b79k", 
                 max_length=77,
                 freeze=True,):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model
        self.max_length = max_length
        self.device = 'cpu'
        if freeze:
            self.freeze()

    def to(self, device):
        self.device = device
        super().to(device)

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        self.device = self.model.ln_final.weight.device # urgly trick
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        x_pool = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        # x_pool_debug = F.normalize(x_pool, dim=-1)
        x = x @ self.model.text_projection
        x = x / x_pool.norm(dim=1, keepdim=True).unsqueeze(1)
        return x

    def encode(self, text):
        return self(text)

@register('openclip_image_context_encoder')
class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    def __init__(self, 
                 arch="ViT-H-14", 
                 version="laion2b_s32b_b79k",
                 freeze=True,):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=version)
        self.model = model.visual
        self.device = 'cpu'
        import torchvision.transforms as tvtrans
        # we only need resize & normalization
        preprocess.transforms[0].size = [224, 224] # make it more precise
        self.preprocess = tvtrans.Compose([
            preprocess.transforms[0],
            preprocess.transforms[4],])
        if freeze:
            self.freeze()

    def to(self, device):
        self.device = device
        super().to(device)

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        z = self.preprocess(image)
        z = self.encode_with_transformer(z)
        return z

    def encode_with_transformer(self, image):
        x = self.model.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.model.class_embedding.to(x.dtype) 
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x], dim=1)
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.model.ln_post(x)
        if self.model.proj is not None:
            x = x @ self.model.proj

        x_pool = x[:, 0, :]
        # x_pool_debug = self.model(image)
        # x_pooln_debug = F.normalize(x_pool_debug, dim=-1)
        x = x / x_pool.norm(dim=1, keepdim=True).unsqueeze(1)
        return x

    def _encode(self, image):
        return self(image)

    def _encode_wmask(self, images, masks):
        z = self._encode(images)
        device = z.device
        vtoken_kernel_size = self.model.conv1.kernel_size
        vtoken_stride = self.model.conv1.stride
        mask_kernal = torch.ones([1, 1, *vtoken_kernel_size], device=device, dtype=z.dtype, requires_grad=False)
        mask_kernal /= np.prod(vtoken_kernel_size)

        assert isinstance(masks, torch.Tensor)
        assert (len(masks.shape)==4) and (masks.shape[1]==1)
        masks = torch.clamp(masks, 0, 1)
        masks = F.interpolate(masks, [224, 224], mode='bilinear')

        vtoken_mask = torch.nn.functional.conv2d(1-masks, mask_kernal, stride=vtoken_stride).flatten(2).transpose(1, 2)
        z[:, 1:, :] = z[:, 1:, :] * vtoken_mask
        z[:, 0, :] = 0
        return z

    def encode(self, images, masks=None):
        if masks is None:
            return self._encode(images)
        else:
            return self._encode_wmask(images, masks)

############################
# def customized tokenizer #
############################

from open_clip import SimpleTokenizer

@register('openclip_text_context_encoder_sdv2_customized_tokenizer_v1')
class FrozenOpenCLIPEmbedderSDv2CustomizedTokenizerV1(FrozenOpenCLIPTextEmbedderSDv2):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    def __init__(self, customized_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(customized_tokens, str):
            customized_tokens = [customized_tokens]
        self.tokenizer = open_clip.SimpleTokenizer(special_tokens=customized_tokens)
        self.num_regular_tokens = self.model.token_embedding.weight.shape[0] 
        self.embedding_dim = self.model.ln_final.weight.shape[0]
        self.customized_token_embedding = nn.Embedding(
            len(customized_tokens), embedding_dim=self.embedding_dim)
        nn.init.normal_(self.customized_token_embedding.weight, std=0.02)

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.tokenizer.encoder["<start_of_text>"]
        eot_token = self.tokenizer.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        maxn = self.num_regular_tokens
        regular_tokens = [[ti if ti < maxn else 0 for ti in tokens] for tokens in all_tokens]
        token_mask = [[0 if ti < maxn else 1 for ti in tokens] for tokens in all_tokens]
        customized_tokens = [[ti-maxn if ti >= maxn else 0 for ti in tokens] for tokens in all_tokens]
        return regular_tokens, customized_tokens, token_mask

    def pad_to_length(self, tokens, context_length=77, eot_token=None):
        result = torch.zeros(len(tokens), context_length, dtype=torch.long)
        eot_token = self.tokenizer.encoder["<end_of_text>"] if eot_token is None else eot_token
        for i, tokens in enumerate(tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

    def forward(self, text):
        self.device = self.model.ln_final.weight.device # urgly trick
        regular_tokens, customized_tokens, token_mask = self.tokenize(text)
        regular_tokens = self.pad_to_length(regular_tokens).to(self.device)
        customized_tokens = self.pad_to_length(customized_tokens, eot_token=0).to(self.device)
        token_mask = self.pad_to_length(token_mask, eot_token=0).to(self.device)
        z0 = self.encode_with_transformer(regular_tokens)
        z1 = self.customized_token_embedding(customized_tokens)
        token_mask = token_mask[:, :, None].type(z0.dtype)
        z = z0 * (1-token_mask) + z1 * token_mask
        return z

@register('openclip_text_context_encoder_sdv2_customized_tokenizer_v2')
class FrozenOpenCLIPEmbedderSDv2CustomizedTokenizerV2(FrozenOpenCLIPTextEmbedderSDv2):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    def __init__(self, customized_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(customized_tokens, str):
            customized_tokens = [customized_tokens]
        self.tokenizer = open_clip.SimpleTokenizer(special_tokens=customized_tokens)
        self.num_regular_tokens = self.model.token_embedding.weight.shape[0] 
        self.embedding_dim = self.model.token_embedding.weight.shape[1]
        self.customized_token_embedding = nn.Embedding(
            len(customized_tokens), embedding_dim=self.embedding_dim)
        nn.init.normal_(self.customized_token_embedding.weight, std=0.02)

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.tokenizer.encoder["<start_of_text>"]
        eot_token = self.tokenizer.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        maxn = self.num_regular_tokens
        regular_tokens = [[ti if ti < maxn else 0 for ti in tokens] for tokens in all_tokens]
        token_mask = [[0 if ti < maxn else 1 for ti in tokens] for tokens in all_tokens]
        customized_tokens = [[ti-maxn if ti >= maxn else 0 for ti in tokens] for tokens in all_tokens]
        return regular_tokens, customized_tokens, token_mask

    def pad_to_length(self, tokens, context_length=77, eot_token=None):
        result = torch.zeros(len(tokens), context_length, dtype=torch.long)
        eot_token = self.tokenizer.encoder["<end_of_text>"] if eot_token is None else eot_token
        for i, tokens in enumerate(tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

    def forward(self, text):
        self.device = self.model.token_embedding.weight.device # urgly trick
        regular_tokens, customized_tokens, token_mask = self.tokenize(text)
        regular_tokens = self.pad_to_length(regular_tokens).to(self.device)
        customized_tokens = self.pad_to_length(customized_tokens, eot_token=0).to(self.device)
        token_mask = self.pad_to_length(token_mask, eot_token=0).to(self.device)
        z = self.encode_with_transformer(regular_tokens, customized_tokens, token_mask)
        return z

    def encode_with_transformer(self, token, customized_token, token_mask):
        x0 = self.model.token_embedding(token)
        x1 = self.customized_token_embedding(customized_token)
        token_mask = token_mask[:, :, None].type(x0.dtype)
        x = x0 * (1-token_mask) + x1 * token_mask        
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

class ln_freezed_temp(nn.LayerNorm):
    def forward(self, x):
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        return super().forward(x)

@register('openclip_text_context_encoder_sdv2_customized_tokenizer_v3')
class FrozenOpenCLIPEmbedderSDv2CustomizedTokenizerV3(FrozenOpenCLIPEmbedderSDv2CustomizedTokenizerV2):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    def __init__(self, customized_tokens, texpand=4, lora_rank=None, lora_bias_trainable=True, *args, **kwargs):
        super().__init__(customized_tokens, *args, **kwargs)
        if isinstance(customized_tokens, str):
            customized_tokens = [customized_tokens]
        self.texpand = texpand
        self.customized_token_embedding = nn.Embedding(
            len(customized_tokens)*texpand, embedding_dim=self.embedding_dim)
        nn.init.normal_(self.customized_token_embedding.weight, std=0.02)

        if lora_rank is not None:
            from .lora import freeze_param, freeze_module, to_lora
            def convert_resattnblock(module):
                module.ln_1.__class__ = ln_freezed_temp
                # freeze_module(module.ln_1)
                module.attn = to_lora(module.attn, lora_rank, lora_bias_trainable)
                module.ln_2.__class__ = ln_freezed_temp
                # freeze_module(module.ln_2)
                module.mlp.c_fc = to_lora(module.mlp.c_fc, lora_rank, lora_bias_trainable)
                module.mlp.c_proj = to_lora(module.mlp.c_proj, lora_rank, lora_bias_trainable)
            freeze_param(self.model, 'positional_embedding')
            freeze_param(self.model, 'text_projection')
            freeze_param(self.model, 'logit_scale')
            for idx, resattnblock in enumerate(self.model.transformer.resblocks):
                convert_resattnblock(resattnblock)
            freeze_module(self.model.token_embedding)
            self.model.ln_final.__class__ = ln_freezed_temp
            # freeze_module(self.model.ln_final)

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.tokenizer.encoder["<start_of_text>"]
        eot_token = self.tokenizer.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        maxn = self.num_regular_tokens
        regular_tokens = [[[ti] if ti < maxn else [0]*self.texpand for ti in tokens] for tokens in all_tokens]
        token_mask     = [[[ 0] if ti < maxn else [1]*self.texpand for ti in tokens] for tokens in all_tokens]
        custom_tokens  = [[[ 0] if ti < maxn else [
            (ti-maxn)*self.texpand+ii for ii in range(self.texpand)]
                for ti in tokens] for tokens in all_tokens]

        from itertools import chain
        regular_tokens = [[i for i in chain(*tokens)] for tokens in regular_tokens]
        token_mask     = [[i for i in chain(*tokens)] for tokens in token_mask]
        custom_tokens  = [[i for i in chain(*tokens)] for tokens in custom_tokens]
        return regular_tokens, custom_tokens, token_mask

###################
# clip expandable #
###################

@register('clip_text_sdv1_customized_embedding')
class CLIPTextSD1CE(nn.Module):
    def __init__(
            self, 
            replace_info="text|elon musk",
            version="openai/clip-vit-large-patch14", 
            max_length=77):
        super().__init__()

        self.name = 'clip_text_sdv1_customized_embedding'
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.reset_replace_info(replace_info)
        self.max_length = max_length
        self.special_token = "<new_token>"

    def reset_replace_info(self, replace_info):
        rtype, rpara = replace_info.split("|")
        self.replace_type = rtype
        if rtype == "token_embedding":
            ce_num = int(rpara)
            ce_dim = self.transformer.text_model.embeddings.token_embedding.weight.size(1)
            self.cembedding = nn.Embedding(ce_num, ce_dim)
            self.cembedding = self.cembedding.to(self.get_device())
        elif rtype == "context_embedding":
            ce_num = int(rpara)
            ce_dim = self.transformer.text_model.encoder.layers[-1].layer_norm2.weight.size(0)
            self.cembedding = nn.Embedding(ce_num, ce_dim)
            self.cembedding = self.cembedding.to(self.get_device())
        else:
            assert rtype=="text"
            self.replace_type = "text"
            self.replace_string = rpara
            self.cembedding = None

    def get_device(self):
        return self.transformer.text_model.embeddings.token_embedding.weight.device

    def position_to_mask(self, tokens, positions):
        mask = torch.zeros_like(tokens)
        for idxb, idxs, idxe in zip(*positions):
            mask[idxb, idxs:idxe] = 1
        return mask

    def forward(self, text):
        tokens, positions = self.tokenize(text)
        mask = self.position_to_mask(tokens, positions)
        max_token_n = tokens.size(1)
        positional_ids = torch.arange(max_token_n)[None].to(self.get_device())
        
        if self.replace_what == 'token_embedding':
            cembeds = self.cembedding(tokens * mask)

            def embedding_customized_forward(
                    self, input_ids=None, position_ids=None, inputs_embeds=None,):
                seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
                if position_ids is None:
                    position_ids = self.position_ids[:, :seq_length]
                if inputs_embeds is None:
                    inputs_embeds = self.token_embedding(input_ids)
                    inputs_embeds = inputs_embeds * (1-mask.float())[:, :, None]
                    inputs_embeds = inputs_embeds + cembeds
                position_embeddings = self.position_embedding(position_ids)
                embeddings = inputs_embeds + position_embeddings
                return embeddings

            import types
            self.transformer.text_model.embeddings.forward = types.MethodType(
                embedding_customized_forward, self.transformer.text_model.embeddings)
            
        else:
            # TODO: Implement
            assert False

        outputs = self.transformer(
            input_ids=tokens, 
            position_ids=positional_ids, )
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

    @torch.no_grad()
    def tokenize(self, text):
        if isinstance(text, str):
            text = [text]

        bos_special_text = "<|startoftext|>"
        text = [ti.replace(self.special_token, bos_special_text) for ti in text]

        batch_encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]

        bosid = tokens[0,  0]
        eosid = tokens[0, -1]
        bs, maxn = tokens.shape

        if self.replace_what in ['token_embedding', 'context_embedding']:
            newtokens = []
            ce_num = self.cembedding.weight.size(0)
            idxi = []; idxstart = []; idxend = [];
            for idxii, tokeni in enumerate(tokens):
                newtokeni = []
                idxjj = 0
                for ii, tokenii in enumerate(tokeni):
                    if (tokenii == bosid) and (ii != 0):
                        newtokeni.extend([i for i in range(ce_num)])
                        idxi.append(idxii); idxstart.append(idxjj);
                        idxjj += ce_num
                        idxjj_record = idxjj if idxjj<=maxn-1 else maxn-1
                        idxend.append(idxjj_record);
                    else:
                        newtokeni.extend([tokenii])
                        idxjj += 1
                newtokeni = newtokeni[:maxn]
                newtokeni[-1] = eosid
                newtokens.append(newtokeni)
            return torch.LongTensor(newtokens).to(self.get_device()), (idxi, idxstart, idxend)
        else:
            # TODO: Implement
            assert False
