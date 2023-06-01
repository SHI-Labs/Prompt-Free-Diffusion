################################################################################
# Copyright (C) 2023 Xingqian Xu - All Rights Reserved                         #
#                                                                              #
# Please visit Prompt-Free-Diffusion's arXiv paper for more details, link at   #
# arxiv.org/abs/2305.16223                                                     #
#                                                                              #
################################################################################

import gradio as gr
import os.path as osp
from PIL import Image
import numpy as np
import time

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

from collections import OrderedDict
from lib.model_zoo.ddim import DDIMSampler

n_sample_image = 1

controlnet_path = OrderedDict([
    ['canny'             , ('canny'   , 'pretrained/controlnet/control_sd15_canny_slimmed.safetensors')],
    ['canny_v11p'        , ('canny'   , 'pretrained/controlnet/control_v11p_sd15_canny_slimmed.safetensors')],
    ['depth'             , ('depth'   , 'pretrained/controlnet/control_sd15_depth_slimmed.safetensors')],
    ['hed'               , ('hed'     , 'pretrained/controlnet/control_sd15_hed_slimmed.safetensors')],
    ['softedge_v11p'     , ('hed'     , 'pretrained/controlnet/control_v11p_sd15_softedge_slimmed.safetensors')],
    ['mlsd'              , ('mlsd'    , 'pretrained/controlnet/control_sd15_mlsd_slimmed.safetensors')],
    ['mlsd_v11p'         , ('mlsd'    , 'pretrained/controlnet/control_v11p_sd15_mlsd_slimmed.safetensors')],
    ['normal'            , ('normal'  , 'pretrained/controlnet/control_sd15_normal_slimmed.safetensors')],
    ['openpose'          , ('openpose', 'pretrained/controlnet/control_sd15_openpose_slimmed.safetensors')],
    ['openpose_v11p'     , ('openpose', 'pretrained/controlnet/control_v11p_sd15_openpose_slimmed.safetensors')],
    ['scribble'          , ('scribble', 'pretrained/controlnet/control_sd15_scribble_slimmed.safetensors')],
    ['seg'               , ('none'    , 'pretrained/controlnet/control_sd15_seg_slimmed.safetensors')],
    ['lineart_v11p'      , ('none'    , 'pretrained/controlnet/control_v11p_sd15_lineart_slimmed.safetensors')],
    ['lineart_anime_v11p', ('none'    , 'pretrained/controlnet/control_v11p_sd15s2_lineart_anime_slimmed.safetensors')],
])

preprocess_method = [
    'canny'                ,
    'depth'                ,
    'hed'                  ,
    'mlsd'                 ,
    'normal'               ,
    'openpose'             ,
    'openpose_withface'    ,
    'openpose_withfacehand',
    'scribble'             ,
    'none'                 ,
]

diffuser_path = OrderedDict([
    ['SD-v1.5'             , 'pretrained/pfd/diffuser/SD-v1-5.safetensors'],
    ['OpenJouney-v4'       , 'pretrained/pfd/diffuser/OpenJouney-v4.safetensors'],
    ['Deliberate-v2.0'     , 'pretrained/pfd/diffuser/Deliberate-v2-0.safetensors'],
    ['RealisticVision-v2.0', 'pretrained/pfd/diffuser/RealisticVision-v2-0.safetensors'],
    ['Anything-v4'         , 'pretrained/pfd/diffuser/Anything-v4.safetensors'],
    ['Oam-v3'              , 'pretrained/pfd/diffuser/AbyssOrangeMix-v3.safetensors'],
    ['Oam-v2'              , 'pretrained/pfd/diffuser/AbyssOrangeMix-v2.safetensors'],
])

ctxencoder_path = OrderedDict([
    ['SeeCoder'      , 'pretrained/pfd/seecoder/seecoder-v1-0.safetensors'],
    ['SeeCoder-PA'   , 'pretrained/pfd/seecoder/seecoder-pa-v1-0.safetensors'],
    ['SeeCoder-Anime', 'pretrained/pfd/seecoder/seecoder-anime-v1-0.safetensors'],
])

##########
# helper #
##########

def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

def load_sd_from_file(target):
    if osp.splitext(target)[-1] == '.ckpt':
        sd = torch.load(target, map_location='cpu')['state_dict']
    elif osp.splitext(target)[-1] == '.pth':
        sd = torch.load(target, map_location='cpu')
    elif osp.splitext(target)[-1] == '.safetensors':
        from safetensors.torch import load_file as stload
        sd = OrderedDict(stload(target, device='cpu'))
    else:
        assert False, "File type must be .ckpt or .pth or .safetensors"
    return sd

########
# main #
########

class prompt_free_diffusion(object):
    def __init__(self, 
                 fp16=False, 
                 tag_ctx=None,
                 tag_diffuser=None,
                 tag_ctl=None,):

        self.tag_ctx = tag_ctx
        self.tag_diffuser = tag_diffuser
        self.tag_ctl = tag_ctl
        self.strict_sd = True

        cfgm = model_cfg_bank()('pfd_seecoder_with_controlnet')
        self.net = get_model()(cfgm)
        
        self.action_load_ctx(tag_ctx)
        self.action_load_diffuser(tag_diffuser)
        self.action_load_ctl(tag_ctl)
 
        if fp16:
            highlight_print('Running in FP16')
            self.net.ctx['image'].fp16 = True
            self.net = self.net.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net.to('cuda')

        self.net.eval()
        self.sampler = DDIMSampler(self.net)

        self.n_sample_image = n_sample_image
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.image_latent_dim = 4

    def load_ctx(self, pretrained):
        sd = load_sd_from_file(pretrained)
        sd_extra = [(ki, vi) for ki, vi in self.net.state_dict().items() \
            if ki.find('ctx.')!=0]
        sd.update(OrderedDict(sd_extra))

        self.net.load_state_dict(sd, strict=True)
        print('Load context encoder from [{}] strict [{}].'.format(pretrained, True))

    def load_diffuser(self, pretrained):
        sd = load_sd_from_file(pretrained)
        if len([ki for ki in sd.keys() if ki.find('diffuser.image.context_blocks.')==0]) == 0:
            sd = [(
                ki.replace('diffuser.text.context_blocks.', 'diffuser.image.context_blocks.'), vi) 
                    for ki, vi in sd.items()]
            sd = OrderedDict(sd)
        sd_extra = [(ki, vi) for ki, vi in self.net.state_dict().items() \
            if ki.find('diffuser.')!=0]
        sd.update(OrderedDict(sd_extra))
        self.net.load_state_dict(sd, strict=True)
        print('Load diffuser from [{}] strict [{}].'.format(pretrained, True))

    def load_ctl(self, pretrained):
        sd = load_sd_from_file(pretrained)
        self.net.ctl.load_state_dict(sd, strict=True)
        print('Load controlnet from [{}] strict [{}].'.format(pretrained, True))

    def action_load_ctx(self, tag):
        pretrained = ctxencoder_path[tag]
        if tag == 'SeeCoder-PA':
            from lib.model_zoo.seecoder import PPE_MLP
            pe_layer = \
                PPE_MLP(freq_num=20, freq_max=None, out_channel=768, mlp_layer=3)
            if self.dtype == torch.float16:
                pe_layer = pe_layer.half()
            if self.use_cuda:
                pe_layer.to('cuda')
            pe_layer.eval()
            self.net.ctx['image'].qtransformer.pe_layer = pe_layer
        else:
            self.net.ctx['image'].qtransformer.pe_layer = None
        if pretrained is not None:
            self.load_ctx(pretrained)
        self.tag_ctx = tag
        return tag

    def action_load_diffuser(self, tag):
        pretrained = diffuser_path[tag]
        if pretrained is not None:
            self.load_diffuser(pretrained)
        self.tag_diffuser = tag
        return tag

    def action_load_ctl(self, tag):
        pretrained = controlnet_path[tag][1]
        if pretrained is not None:
            self.load_ctl(pretrained)
        self.tag_ctl = tag
        return tag

    def action_autoset_hw(self, imctl):
        if imctl is None:
            return 512, 512
        w, h = imctl.size
        w = w//64 * 64
        h = h//64 * 64
        w = w if w >=512 else 512
        w = w if w <=1536 else 1536
        h = h if h >=512 else 512
        h = h if h <=1536 else 1536
        return h, w

    def action_autoset_method(self, tag):
        return controlnet_path[tag][0]

    def action_inference(
            self, im, imctl, ctl_method, do_preprocess, 
            h, w, ugscale, seed, 
            tag_ctx, tag_diffuser, tag_ctl,):

        if tag_ctx != self.tag_ctx:
            self.action_load_ctx(tag_ctx)
        if tag_diffuser != self.tag_diffuser:
            self.action_load_diffuser(tag_diffuser)
        if tag_ctl != self.tag_ctl:
            self.action_load_ctl(tag_ctl)

        n_samples = self.n_sample_image

        sampler = self.sampler
        device = self.net.device

        w = w//64 * 64
        h = h//64 * 64
        if imctl is not None:
            imctl = imctl.resize([w, h], Image.Resampling.BICUBIC)

        craw = tvtrans.ToTensor()(im)[None].to(device).to(self.dtype)
        c = self.net.ctx_encode(craw, which='image').repeat(n_samples, 1, 1)
        u = torch.zeros_like(c)

        if tag_ctx in ["SeeCoder-Anime"]:
            u = torch.load('assets/anime_ug.pth')[None].to(device).to(self.dtype)
            pad = c.size(1) - u.size(1)
            u = torch.cat([u, torch.zeros_like(u[:, 0:1].repeat(1, pad, 1))], axis=1)

        if tag_ctl != 'none':
            ccraw = tvtrans.ToTensor()(imctl)[None].to(device).to(self.dtype)
            if do_preprocess:
                cc = self.net.ctl.preprocess(ccraw, type=ctl_method, size=[h, w])
                cc = cc.to(self.dtype)
            else:
                cc = ccraw
        else:
            cc = None

        shape = [n_samples, self.image_latent_dim, h//8, w//8]

        if seed < 0:
            np.random.seed(int(time.time()))
            torch.manual_seed(-seed + 100)
        else:
            np.random.seed(seed + 100)
            torch.manual_seed(seed)

        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'image',},
            c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':ugscale,
                    'control':cc,},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)

        ccout = [tvtrans.ToPILImage()(i) for i in cc] if cc is not None else []
        imout = self.net.vae_decode(x, which='image')
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return imout + ccout

pfd_inference = prompt_free_diffusion(
    fp16=True, tag_ctx = 'SeeCoder', tag_diffuser = 'Deliberate-v2.0', tag_ctl = 'canny',)

#################
# sub interface #
#################

cache_examples = True

def get_example():
    case = [
        [
            'assets/examples/ghibli-input.jpg', 
            'assets/examples/ghibli-canny.png', 
            'canny', False, 
            768, 1024, 1.8, 23, 
            'SeeCoder', 'Deliberate-v2.0', 'canny', ],
        [
            'assets/examples/astronautridinghouse-input.jpg', 
            'assets/examples/astronautridinghouse-canny.png', 
            'canny', False, 
            512, 768, 2.0, 21, 
            'SeeCoder', 'Deliberate-v2.0', 'canny', ],
        [
            'assets/examples/grassland-input.jpg', 
            'assets/examples/grassland-scribble.png', 
            'scribble', False, 
            768, 512, 2.0, 41, 
            'SeeCoder', 'Deliberate-v2.0', 'scribble', ],
        [
            'assets/examples/jeep-input.jpg', 
            'assets/examples/jeep-depth.png', 
            'depth', False, 
            512, 768, 2.0, 30, 
            'SeeCoder', 'Deliberate-v2.0', 'depth', ],
        [
            'assets/examples/bedroom-input.jpg', 
            'assets/examples/bedroom-mlsd.png', 
            'mlsd', False, 
            512, 512, 2.0, 31, 
            'SeeCoder', 'Deliberate-v2.0', 'mlsd', ],
        [
            'assets/examples/nightstreet-input.jpg', 
            'assets/examples/nightstreet-canny.png', 
            'canny', False, 
            768, 512, 2.3, 20, 
            'SeeCoder', 'Deliberate-v2.0', 'canny', ],
        [
            'assets/examples/woodcar-input.jpg', 
            'assets/examples/woodcar-depth.png', 
            'depth', False, 
            768, 512, 2.0, 20, 
            'SeeCoder', 'Deliberate-v2.0', 'depth', ],
        [
            'assets/examples-anime/miku.jpg', 
            'assets/examples-anime/miku-canny.png', 
            'canny', False, 
            768, 576, 1.5, 22, 
            'SeeCoder-Anime', 'Anything-v4', 'canny', ],
        [
            'assets/examples-anime/random1.jpg', 
            'assets/examples-anime/pose.png', 
            'openpose', False, 
            768, 1536, 2.5, 28, 
            'SeeCoder-Anime', 'Oam-v2', 'openpose_v11p', ], 
        [
            'assets/examples-anime/camping.jpg', 
            'assets/examples-anime/pose.png', 
            'openpose', False, 
            768, 1536, 2.0, 35, 
            'SeeCoder-Anime', 'Anything-v4', 'openpose_v11p', ],
        [
            'assets/examples-anime/hanfu_girl.jpg', 
            'assets/examples-anime/pose.png', 
            'openpose', False, 
            768, 1536, 2.0, 20, 
            'SeeCoder-Anime', 'Anything-v4', 'openpose_v11p', ],
    ]
    return case

def interface():
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label='Image Input', type='pil', elem_id='customized_imbox')
            with gr.Row():
                out_width  = gr.Slider(label="Width" , minimum=512, maximum=1536, value=512, step=64, visible=True)
                out_height = gr.Slider(label="Height", minimum=512, maximum=1536, value=512, step=64, visible=True)
            with gr.Row():
                scl_lvl = gr.Slider(label="CFGScale", minimum=0, maximum=10, value=2, step=0.01, visible=True)
                seed = gr.Number(20, label="Seed", precision=0)
            with gr.Row():
                tag_ctx = gr.Dropdown(label='Context Encoder', choices=[pi for pi in ctxencoder_path.keys()], value='SeeCoder')
                tag_diffuser = gr.Dropdown(label='Diffuser', choices=[pi for pi in diffuser_path.keys()], value='Deliberate-v2.0')
            button = gr.Button("Run")
        with gr.Column():
            ctl_input = gr.Image(label='Control Input', type='pil', elem_id='customized_imbox')
            do_preprocess = gr.Checkbox(label='Preprocess', value=False)
            with gr.Row():
                ctl_method = gr.Dropdown(label='Preprocess Type', choices=preprocess_method, value='canny')
                tag_ctl    = gr.Dropdown(label='ControlNet',      choices=[pi for pi in controlnet_path.keys()], value='canny')
        with gr.Column():
            img_output = gr.Gallery(label="Image Result", elem_id='customized_imbox').style(grid=n_sample_image+1)

    tag_ctl.change(
        pfd_inference.action_autoset_method,
        inputs = [tag_ctl],
        outputs = [ctl_method],)

    ctl_input.change(
        pfd_inference.action_autoset_hw,
        inputs = [ctl_input],
        outputs = [out_height, out_width],)

    # tag_ctx.change(
    #     pfd_inference.action_load_ctx,
    #     inputs = [tag_ctx],
    #     outputs = [tag_ctx],)

    # tag_diffuser.change(
    #     pfd_inference.action_load_diffuser,
    #     inputs = [tag_diffuser],
    #     outputs = [tag_diffuser],)

    # tag_ctl.change(
    #     pfd_inference.action_load_ctl,
    #     inputs = [tag_ctl],
    #     outputs = [tag_ctl],)

    button.click(
        pfd_inference.action_inference,
        inputs=[img_input, ctl_input, ctl_method, do_preprocess, 
                out_height, out_width, scl_lvl, seed, 
                tag_ctx, tag_diffuser, tag_ctl, ],
        outputs=[img_output])
    
    gr.Examples(
        label='Examples', 
        examples=get_example(), 
        fn=pfd_inference.action_inference,
        inputs=[img_input, ctl_input, ctl_method, do_preprocess,
                out_height, out_width, scl_lvl, seed, 
                tag_ctx, tag_diffuser, tag_ctl, ],
        outputs=[img_output],
        cache_examples=cache_examples,)

#############
# Interface #
#############

css = """
    #customized_imbox {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"] {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"]>div {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"]>iframe {
        min-height: 450px;
    }
    #customized_imbox>div.unpadded_box {
        min-height: 450px;
    }
    #myinst {
        font-size: 0.8rem; 
        margin: 0rem;
        color: #6B7280;
    }
    #maskinst {
        text-align: justify;
        min-width: 1200px;
    }
    #maskinst>img {
        min-width:399px;
        max-width:450px;
        vertical-align: top;
        display: inline-block;
    }
    #maskinst:after {
        content: "";
        width: 100%;
        display: inline-block;
    }
"""

if True:
    with gr.Blocks(css=css) as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                Prompt-Free Diffusion
            </h1>
            <p style="font-size: 1rem; margin: 0rem">
                Xingqian Xu<sup>1,6</sup>, Jiayi Guo<sup>1,2</sup>, Zhangyang Wang<sup>3,6</sup>, Gao Huang<sup>2</sup>, Irfan Essa<sup>4,5</sup>, and Humphrey Shi<sup>1,6</sup>
            </p>
            <p style="font-size: 0.8rem; margin: 0rem; line-height: 1em">
                <sup>1</sup>SHI Labs @ UIUC & Oregon, <sup>2</sup>Tsinghua University, <sup>3</sup>UT Austin, <sup>4</sup>Georgia Tech, <sup>5</sup>Google Research, <sup>6</sup>Picsart AI Research (PAIR)
            </p>
            <p style="font-size: 0.9rem; margin: 0rem; line-height: 1.2em; margin-top:1em">
                The performance of Text2Image is largely dependent on text prompts. 
                In Prompt-Free Diffusion, no prompt is needed, just a reference images! 
                At the core of Prompt-Free Diffusion is an image-only semantic context encoder (SeeCoder). 
                SeeCoder is reusable to most CLIP-based T2I models: just drop in and replace CLIP, then you will create your own prompt-free diffusion.
                <a href="https://github.com/SHI-Labs/Prompt-Free-Diffusion">[Github]</a> <a href="https://arxiv.org/abs/2305.16223">[arXiv]</a>
            </p>
            </div>
            """)

        interface()

        # gr.HTML(
        #     """
        #     <div style="text-align: justify; max-width: 1200px; margin: 20px auto;">
        #     <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        #     <b>Version</b>: {}
        #     </h3>
        #     </div>
        #     """.format(' '+str(pfd_inference.pretrained)))

    demo.launch(server_name="0.0.0.0", server_port=11234)
    # demo.launch()
