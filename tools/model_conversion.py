# A tool to convert sdwebui/huggingface model to pfd or vice versa

import os.path as osp
import torch
from collections import OrderedDict
from safetensors.torch import load_file, save_file

class sdwebui_diffuser_to_pfd_mover():
    def __init__(self):
        pass

    def move_tembed_blocks(self):
        mapping = []
        mapping.append(['time_embed.0.weight', 'time_embed.0.weight'])
        mapping.append(['time_embed.0.bias'  , 'time_embed.0.bias'  ])
        mapping.append(['time_embed.2.weight', 'time_embed.2.weight'])
        mapping.append(['time_embed.2.bias'  , 'time_embed.2.bias'  ])
        return mapping

    def move_first_several_layers(self):
        mapping = []
        mapping.append(['input_blocks.0.0.weight', 'data_blocks.0.0.weight'])
        mapping.append(['input_blocks.0.0.bias'  , 'data_blocks.0.0.bias'  ])
        return mapping

    def move_down_resblocks(self, ref=[1, 2, 3], to=[1, 2, 3]):
        mapping = []
        for ri, ti in zip(ref[0:2], to[0:2]):
            t = []
            t.append(['input_blocks.X.0.in_layers.0.weight'  ,'data_blocks.X.0.in_layers.0.weight' ])
            t.append(['input_blocks.X.0.in_layers.0.bias'    ,'data_blocks.X.0.in_layers.0.bias'   ])
            t.append(['input_blocks.X.0.in_layers.2.weight'  ,'data_blocks.X.0.in_layers.2.weight' ])
            t.append(['input_blocks.X.0.in_layers.2.bias'    ,'data_blocks.X.0.in_layers.2.bias'   ])
            t.append(['input_blocks.X.0.emb_layers.1.weight' ,'data_blocks.X.0.emb_layers.1.weight'])
            t.append(['input_blocks.X.0.emb_layers.1.bias'   ,'data_blocks.X.0.emb_layers.1.bias'  ])
            t.append(['input_blocks.X.0.out_layers.0.weight' ,'data_blocks.X.0.out_layers.0.weight'])
            t.append(['input_blocks.X.0.out_layers.0.bias'   ,'data_blocks.X.0.out_layers.0.bias'  ])
            t.append(['input_blocks.X.0.out_layers.3.weight' ,'data_blocks.X.0.out_layers.3.weight'])
            t.append(['input_blocks.X.0.out_layers.3.bias'   ,'data_blocks.X.0.out_layers.3.bias'  ])
            if ri in [4, 7]:
                t.append(['input_blocks.X.0.skip_connection.weight' ,'data_blocks.X.0.skip_connection.weight'])
                t.append(['input_blocks.X.0.skip_connection.bias'   ,'data_blocks.X.0.skip_connection.bias'  ])
            for m0, m1 in t:
                mapping.append([m0.replace('X', str(ri)), m1.replace('X', str(ti))])
        if len(ref)>2:
            mapping.append(['input_blocks.{}.0.op.weight'.format(ref[-1]), 'data_blocks.{}.0.op.weight'.format(to[-1])])
            mapping.append(['input_blocks.{}.0.op.bias'  .format(ref[-1]), 'data_blocks.{}.0.op.bias'  .format(to[-1])])
        return mapping

    def move_mid_resblocks(self, ref=[0, 2], to=[12, 13]):
        mapping = []
        for ri, ti in zip(ref, to):
            t = []
            t.append(['middle_block.X.in_layers.0.weight' , 'data_blocks.X.0.in_layers.0.weight' ])
            t.append(['middle_block.X.in_layers.0.bias'   , 'data_blocks.X.0.in_layers.0.bias'   ])
            t.append(['middle_block.X.in_layers.2.weight' , 'data_blocks.X.0.in_layers.2.weight' ])
            t.append(['middle_block.X.in_layers.2.bias'   , 'data_blocks.X.0.in_layers.2.bias'   ])
            t.append(['middle_block.X.emb_layers.1.weight', 'data_blocks.X.0.emb_layers.1.weight'])
            t.append(['middle_block.X.emb_layers.1.bias'  , 'data_blocks.X.0.emb_layers.1.bias'  ])
            t.append(['middle_block.X.out_layers.0.weight', 'data_blocks.X.0.out_layers.0.weight'])
            t.append(['middle_block.X.out_layers.0.bias'  , 'data_blocks.X.0.out_layers.0.bias'  ])
            t.append(['middle_block.X.out_layers.3.weight', 'data_blocks.X.0.out_layers.3.weight'])
            t.append(['middle_block.X.out_layers.3.bias'  , 'data_blocks.X.0.out_layers.3.bias'  ])
            for m0, m1 in t:
                mapping.append([m0.replace('X', str(ri)), m1.replace('X', str(ti))])
        return mapping

    def move_up_resblocks(self, ref=[0, 1, 2], to=[14, 15, 16]):
        mapping = []
        for ri, ti in zip(ref[0:3], to[0:3]):
            t = []
            t.append(['output_blocks.X.0.in_layers.0.weight' , 'data_blocks.X.0.in_layers.0.weight' ])
            t.append(['output_blocks.X.0.in_layers.0.bias'   , 'data_blocks.X.0.in_layers.0.bias'   ])
            t.append(['output_blocks.X.0.in_layers.2.weight' , 'data_blocks.X.0.in_layers.2.weight' ])
            t.append(['output_blocks.X.0.in_layers.2.bias'   , 'data_blocks.X.0.in_layers.2.bias'   ])
            t.append(['output_blocks.X.0.emb_layers.1.weight', 'data_blocks.X.0.emb_layers.1.weight'])
            t.append(['output_blocks.X.0.emb_layers.1.bias'  , 'data_blocks.X.0.emb_layers.1.bias'  ])
            t.append(['output_blocks.X.0.out_layers.0.weight', 'data_blocks.X.0.out_layers.0.weight'])
            t.append(['output_blocks.X.0.out_layers.0.bias'  , 'data_blocks.X.0.out_layers.0.bias'  ])
            t.append(['output_blocks.X.0.out_layers.3.weight', 'data_blocks.X.0.out_layers.3.weight'])
            t.append(['output_blocks.X.0.out_layers.3.bias'  , 'data_blocks.X.0.out_layers.3.bias'  ])
            t.append(['output_blocks.X.0.skip_connection.weight', 'data_blocks.X.0.skip_connection.weight'])
            t.append(['output_blocks.X.0.skip_connection.bias'  , 'data_blocks.X.0.skip_connection.bias'  ])
            for m0, m1 in t:
                mapping.append([m0.replace('X', str(ri)), m1.replace('X', str(ti))])
        if (len(ref)>3) and (ref[-1]==2):
            mapping.append(['output_blocks.{}.1.conv.weight'.format(ref[-1]), 'data_blocks.{}.0.conv.weight'.format(to[-1])])
            mapping.append(['output_blocks.{}.1.conv.bias'  .format(ref[-1]), 'data_blocks.{}.0.conv.bias'  .format(to[-1])])
        elif (len(ref)>3) and (ref[-1]!=2):
            mapping.append(['output_blocks.{}.2.conv.weight'.format(ref[-1]), 'data_blocks.{}.0.conv.weight'.format(to[-1])])
            mapping.append(['output_blocks.{}.2.conv.bias'  .format(ref[-1]), 'data_blocks.{}.0.conv.bias'  .format(to[-1])])
        return mapping

    def move_last_several_layers(self):
        mapping = []
        mapping.append(['out.0.weight', 'data_blocks.29.0.0.weight', ])
        mapping.append(['out.0.bias'  , 'data_blocks.29.0.0.bias'  , ])
        mapping.append(['out.2.weight', 'data_blocks.29.0.2.weight', ])
        mapping.append(['out.2.bias'  , 'data_blocks.29.0.2.bias'  , ])
        return mapping

    def move_down_attn(self, ref=[1, 2], to=[0, 1]):
        mapping = []
        for ri, ti in zip(ref, to):
            t = []
            t.append(['input_blocks.X.1.norm.weight'   , 'context_blocks.X.0.norm.weight'   ])
            t.append(['input_blocks.X.1.norm.bias'     , 'context_blocks.X.0.norm.bias'     ])
            t.append(['input_blocks.X.1.proj_in.weight', 'context_blocks.X.0.proj_in.weight'])
            t.append(['input_blocks.X.1.proj_in.bias'  , 'context_blocks.X.0.proj_in.bias'  ])  
            t.append(['input_blocks.X.1.transformer_blocks.0.attn1.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_q.weight'    ])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn1.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_k.weight'    ])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn1.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_v.weight'    ])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn1.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.weight'])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn1.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.bias'  ])
            t.append(['input_blocks.X.1.transformer_blocks.0.ff.net.0.proj.weight' , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.weight' ])
            t.append(['input_blocks.X.1.transformer_blocks.0.ff.net.0.proj.bias'   , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.bias'   ])
            t.append(['input_blocks.X.1.transformer_blocks.0.ff.net.2.weight'      , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.weight'      ])
            t.append(['input_blocks.X.1.transformer_blocks.0.ff.net.2.bias'        , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.bias'        ])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn2.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_q.weight'    ])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn2.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_k.weight'    ])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn2.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_v.weight'    ])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn2.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.weight'])
            t.append(['input_blocks.X.1.transformer_blocks.0.attn2.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.bias'  ])
            t.append(['input_blocks.X.1.transformer_blocks.0.norm1.weight', 'context_blocks.X.0.transformer_blocks.0.norm1.weight'])
            t.append(['input_blocks.X.1.transformer_blocks.0.norm1.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm1.bias'  ])
            t.append(['input_blocks.X.1.transformer_blocks.0.norm2.weight', 'context_blocks.X.0.transformer_blocks.0.norm2.weight'])
            t.append(['input_blocks.X.1.transformer_blocks.0.norm2.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm2.bias'  ])
            t.append(['input_blocks.X.1.transformer_blocks.0.norm3.weight', 'context_blocks.X.0.transformer_blocks.0.norm3.weight'])
            t.append(['input_blocks.X.1.transformer_blocks.0.norm3.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm3.bias'  ])
            t.append(['input_blocks.X.1.proj_out.weight', 'context_blocks.X.0.proj_out.weight'])
            t.append(['input_blocks.X.1.proj_out.bias'  , 'context_blocks.X.0.proj_out.bias'  ])
            for m0, m1 in t:
                mapping.append([m0.replace('X', str(ri)), m1.replace('X', str(ti))])
        return mapping

    def move_mid_attn(self, to=[6]):
        mapping = []
        for ti in to:
            t = []
            t.append(['middle_block.1.norm.weight'   , 'context_blocks.X.0.norm.weight'   ])
            t.append(['middle_block.1.norm.bias'     , 'context_blocks.X.0.norm.bias'     ])
            t.append(['middle_block.1.proj_in.weight', 'context_blocks.X.0.proj_in.weight'])
            t.append(['middle_block.1.proj_in.bias'  , 'context_blocks.X.0.proj_in.bias'  ])  
            t.append(['middle_block.1.transformer_blocks.0.attn1.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_q.weight'    ])
            t.append(['middle_block.1.transformer_blocks.0.attn1.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_k.weight'    ])
            t.append(['middle_block.1.transformer_blocks.0.attn1.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_v.weight'    ])
            t.append(['middle_block.1.transformer_blocks.0.attn1.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.weight'])
            t.append(['middle_block.1.transformer_blocks.0.attn1.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.bias'  ])
            t.append(['middle_block.1.transformer_blocks.0.ff.net.0.proj.weight' , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.weight' ])
            t.append(['middle_block.1.transformer_blocks.0.ff.net.0.proj.bias'   , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.bias'   ])
            t.append(['middle_block.1.transformer_blocks.0.ff.net.2.weight'      , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.weight'      ])
            t.append(['middle_block.1.transformer_blocks.0.ff.net.2.bias'        , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.bias'        ])
            t.append(['middle_block.1.transformer_blocks.0.attn2.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_q.weight'    ])
            t.append(['middle_block.1.transformer_blocks.0.attn2.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_k.weight'    ])
            t.append(['middle_block.1.transformer_blocks.0.attn2.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_v.weight'    ])
            t.append(['middle_block.1.transformer_blocks.0.attn2.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.weight'])
            t.append(['middle_block.1.transformer_blocks.0.attn2.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.bias'  ])
            t.append(['middle_block.1.transformer_blocks.0.norm1.weight', 'context_blocks.X.0.transformer_blocks.0.norm1.weight'])
            t.append(['middle_block.1.transformer_blocks.0.norm1.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm1.bias'  ])
            t.append(['middle_block.1.transformer_blocks.0.norm2.weight', 'context_blocks.X.0.transformer_blocks.0.norm2.weight'])
            t.append(['middle_block.1.transformer_blocks.0.norm2.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm2.bias'  ])
            t.append(['middle_block.1.transformer_blocks.0.norm3.weight', 'context_blocks.X.0.transformer_blocks.0.norm3.weight'])
            t.append(['middle_block.1.transformer_blocks.0.norm3.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm3.bias'  ])
            t.append(['middle_block.1.proj_out.weight', 'context_blocks.X.0.proj_out.weight'])
            t.append(['middle_block.1.proj_out.bias'  , 'context_blocks.X.0.proj_out.bias'  ])
            for m0, m1 in t:
                mapping.append([m0, m1.replace('X', str(ti))])
        return mapping

    def move_up_attn(self, ref, to):
        mapping = []
        for ri, ti in zip(ref, to):
            t = []
            t.append(['output_blocks.X.1.norm.weight'   , 'context_blocks.X.0.norm.weight'   ])
            t.append(['output_blocks.X.1.norm.bias'     , 'context_blocks.X.0.norm.bias'     ])
            t.append(['output_blocks.X.1.proj_in.weight', 'context_blocks.X.0.proj_in.weight'])
            t.append(['output_blocks.X.1.proj_in.bias'  , 'context_blocks.X.0.proj_in.bias'  ])  
            t.append(['output_blocks.X.1.transformer_blocks.0.attn1.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_q.weight'    ])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn1.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_k.weight'    ])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn1.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_v.weight'    ])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn1.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.weight'])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn1.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.bias'  ])
            t.append(['output_blocks.X.1.transformer_blocks.0.ff.net.0.proj.weight' , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.weight' ])
            t.append(['output_blocks.X.1.transformer_blocks.0.ff.net.0.proj.bias'   , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.bias'   ])
            t.append(['output_blocks.X.1.transformer_blocks.0.ff.net.2.weight'      , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.weight'      ])
            t.append(['output_blocks.X.1.transformer_blocks.0.ff.net.2.bias'        , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.bias'        ])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn2.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_q.weight'    ])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn2.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_k.weight'    ])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn2.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_v.weight'    ])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn2.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.weight'])
            t.append(['output_blocks.X.1.transformer_blocks.0.attn2.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.bias'  ])
            t.append(['output_blocks.X.1.transformer_blocks.0.norm1.weight', 'context_blocks.X.0.transformer_blocks.0.norm1.weight'])
            t.append(['output_blocks.X.1.transformer_blocks.0.norm1.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm1.bias'  ])
            t.append(['output_blocks.X.1.transformer_blocks.0.norm2.weight', 'context_blocks.X.0.transformer_blocks.0.norm2.weight'])
            t.append(['output_blocks.X.1.transformer_blocks.0.norm2.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm2.bias'  ])
            t.append(['output_blocks.X.1.transformer_blocks.0.norm3.weight', 'context_blocks.X.0.transformer_blocks.0.norm3.weight'])
            t.append(['output_blocks.X.1.transformer_blocks.0.norm3.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm3.bias'  ])
            t.append(['output_blocks.X.1.proj_out.weight', 'context_blocks.X.0.proj_out.weight'])
            t.append(['output_blocks.X.1.proj_out.bias'  , 'context_blocks.X.0.proj_out.bias'  ])
            for m0, m1 in t:
                mapping.append([m0.replace('X', str(ri)), m1.replace('X', str(ti))])
        return mapping

    def get_mapping(self):
        r = []
        d = []
        d += self.move_tembed_blocks()
        d += self.move_first_several_layers()
        d += self.move_down_resblocks([1, 2, 3], [1, 2, 3])
        d += self.move_down_resblocks([4, 5, 6], [4, 5, 6])
        d += self.move_down_resblocks([7, 8, 9], [7, 8, 9])
        d += self.move_down_resblocks([10, 11], [10, 11])
        d += self.move_mid_resblocks([0, 2], [12, 13])
        d += self.move_up_resblocks([0, 1, 2, 2], [14, 15, 16, 17])
        d += self.move_up_resblocks([3, 4, 5, 5], [18, 19, 20, 21])
        d += self.move_up_resblocks([6, 7, 8, 8], [22, 23, 24, 25])
        d += self.move_up_resblocks([9, 10, 11] , [26, 27, 28])
        d += self.move_last_several_layers()

        for dfrom, dto in d:
            r.append(['model.diffusion_model.'+dfrom, 'diffuser.image.'+dto])        
        c = []
        c += self.move_down_attn([1, 2, 4, 5, 7, 8], [0, 1, 2, 3, 4, 5])
        c += self.move_mid_attn([6])
        c += self.move_up_attn([3, 4, 5, 6, 7, 8, 9, 10, 11], [7, 8, 9, 10, 11, 12, 13, 14, 15])
        for cfrom, cto in c:
            r.append(['model.diffusion_model.'+cfrom, 'diffuser.text.'+cto])
        return r
 
    def __call__(self, sd, reverse=False, ema=False):
        newsd = []
        mapping = self.get_mapping()
        for mfrom, mto in mapping:
            if ema:
                mfrom = mfrom.replace('model.diffusion_model', 'model_ema|diffusion_model')
                mfrom = mfrom.replace('.', '')
                mfrom = mfrom.replace('|', '.')
            if not reverse:
                newsd.append([mto, sd[mfrom]])
            else:
                newsd.append([mfrom, sd[mto]])
        return OrderedDict(newsd)

class sdwebui_ctx_to_pfd_mover():
    def __init__(self):
        pass

    def __call__(self, sd, reverse=False):
        newsd = []
        for ki, vi in sd.items():
            if ki.find('cond_stage_model.')==0:
                if not reverse:
                    newsd.append([ki.replace('cond_stage_model.', ''), vi])
                else:
                    newsd.append(['cond_stage_model.' + ki, vi])
        return OrderedDict(newsd)

class sdwebui_vae_to_pfd_mover():
    def __init__(self):
        pass

    def __call__(self, sd, reverse=False):
        newsd = []
        for ki, vi in sd.items():
            if ki.find('first_stage_model.')==0:
                if not reverse:
                    newsd.append([ki.replace('first_stage_model.', ''), vi])
                else:
                    newsd.append(['first_stage_model.' + ki, vi])
        return OrderedDict(newsd)

class sdhuggingface_diffuser_to_pfd_mover():
    def __init__(self):
        pass

    def move_tembed_blocks(self):
        mapping = []
        mapping.append(['time_embedding.linear_1.weight', 'time_embed.0.weight'])
        mapping.append(['time_embedding.linear_1.bias'  , 'time_embed.0.bias'  ])
        mapping.append(['time_embedding.linear_2.weight', 'time_embed.2.weight'])
        mapping.append(['time_embedding.linear_2.bias'  , 'time_embed.2.bias'  ])
        return mapping

    def move_first_several_layers(self):
        mapping = []
        mapping.append(['conv_in.weight', 'data_blocks.0.0.weight'])
        mapping.append(['conv_in.bias'  , 'data_blocks.0.0.bias'  ])
        return mapping

    def move_down_resblocks(self, ref=[0], to=[1]):
        mapping = []
        for ri, ti in zip(ref, to):
            for subri, subti in zip([0, 1], [ti, ti+1]):
                t = []
                t.append(['down_blocks.X.resnets.Y.norm1.weight'        ,'data_blocks.X.0.in_layers.0.weight' ])
                t.append(['down_blocks.X.resnets.Y.norm1.bias'          ,'data_blocks.X.0.in_layers.0.bias'   ])
                t.append(['down_blocks.X.resnets.Y.conv1.weight'        ,'data_blocks.X.0.in_layers.2.weight' ])
                t.append(['down_blocks.X.resnets.Y.conv1.bias'          ,'data_blocks.X.0.in_layers.2.bias'   ])
                t.append(['down_blocks.X.resnets.Y.time_emb_proj.weight','data_blocks.X.0.emb_layers.1.weight'])
                t.append(['down_blocks.X.resnets.Y.time_emb_proj.bias'  ,'data_blocks.X.0.emb_layers.1.bias'  ])
                t.append(['down_blocks.X.resnets.Y.norm2.weight'        ,'data_blocks.X.0.out_layers.0.weight'])
                t.append(['down_blocks.X.resnets.Y.norm2.bias'          ,'data_blocks.X.0.out_layers.0.bias'  ])
                t.append(['down_blocks.X.resnets.Y.conv2.weight'        ,'data_blocks.X.0.out_layers.3.weight'])
                t.append(['down_blocks.X.resnets.Y.conv2.bias'          ,'data_blocks.X.0.out_layers.3.bias'  ])
                if (ri > 0) and ((ri < 3)) and (subri == 0):
                    t.append(['down_blocks.X.resnets.Y.conv_shortcut.weight', 'data_blocks.X.0.skip_connection.weight'])
                    t.append(['down_blocks.X.resnets.Y.conv_shortcut.bias'  , 'data_blocks.X.0.skip_connection.bias'  ])
                for m0, m1 in t:
                    mapping.append([
                        m0.replace('X', str(ri)).replace('Y', str(subri)), 
                        m1.replace('X', str(subti))])
            if (ri < 3):
                t = []
                t.append(['down_blocks.X.downsamplers.0.conv.weight','data_blocks.X.0.op.weight'])
                t.append(['down_blocks.X.downsamplers.0.conv.bias'  ,'data_blocks.X.0.op.bias'  ])
                for m0, m1 in t:
                    mapping.append([m0.replace('X', str(ri)), m1.replace('X', str(ti+2))])
        return mapping

    def move_mid_resblocks(self, to=[12]):
        mapping = []
        for ti in to:
            for subri, subti in zip([0, 1], [ti, ti+1]):
                t = []
                t.append(['mid_block.resnets.Y.norm1.weight'        ,'data_blocks.X.0.in_layers.0.weight' ])
                t.append(['mid_block.resnets.Y.norm1.bias'          ,'data_blocks.X.0.in_layers.0.bias'   ])
                t.append(['mid_block.resnets.Y.conv1.weight'        ,'data_blocks.X.0.in_layers.2.weight' ])
                t.append(['mid_block.resnets.Y.conv1.bias'          ,'data_blocks.X.0.in_layers.2.bias'   ])
                t.append(['mid_block.resnets.Y.time_emb_proj.weight','data_blocks.X.0.emb_layers.1.weight'])
                t.append(['mid_block.resnets.Y.time_emb_proj.bias'  ,'data_blocks.X.0.emb_layers.1.bias'  ])
                t.append(['mid_block.resnets.Y.norm2.weight'        ,'data_blocks.X.0.out_layers.0.weight'])
                t.append(['mid_block.resnets.Y.norm2.bias'          ,'data_blocks.X.0.out_layers.0.bias'  ])
                t.append(['mid_block.resnets.Y.conv2.weight'        ,'data_blocks.X.0.out_layers.3.weight'])
                t.append(['mid_block.resnets.Y.conv2.bias'          ,'data_blocks.X.0.out_layers.3.bias'  ])
                for m0, m1 in t:
                    mapping.append([m0.replace('Y', str(subri)), m1.replace('X', str(subti))])
        return mapping

    def move_up_resblocks(self, ref=[0], to=[14]):
        mapping = []
        for ri, ti in zip(ref, to):
            for subri, subti in zip([0, 1, 2], [ti, ti+1, ti+2]):
                t = []
                t.append(['up_blocks.X.resnets.Y.norm1.weight'        ,'data_blocks.X.0.in_layers.0.weight' ])
                t.append(['up_blocks.X.resnets.Y.norm1.bias'          ,'data_blocks.X.0.in_layers.0.bias'   ])
                t.append(['up_blocks.X.resnets.Y.conv1.weight'        ,'data_blocks.X.0.in_layers.2.weight' ])
                t.append(['up_blocks.X.resnets.Y.conv1.bias'          ,'data_blocks.X.0.in_layers.2.bias'   ])
                t.append(['up_blocks.X.resnets.Y.time_emb_proj.weight','data_blocks.X.0.emb_layers.1.weight'])
                t.append(['up_blocks.X.resnets.Y.time_emb_proj.bias'  ,'data_blocks.X.0.emb_layers.1.bias'  ])
                t.append(['up_blocks.X.resnets.Y.norm2.weight'        ,'data_blocks.X.0.out_layers.0.weight'])
                t.append(['up_blocks.X.resnets.Y.norm2.bias'          ,'data_blocks.X.0.out_layers.0.bias'  ])
                t.append(['up_blocks.X.resnets.Y.conv2.weight'        ,'data_blocks.X.0.out_layers.3.weight'])
                t.append(['up_blocks.X.resnets.Y.conv2.bias'          ,'data_blocks.X.0.out_layers.3.bias'  ])
                t.append(['up_blocks.X.resnets.Y.conv_shortcut.weight','data_blocks.X.0.skip_connection.weight'])
                t.append(['up_blocks.X.resnets.Y.conv_shortcut.bias'  ,'data_blocks.X.0.skip_connection.bias'  ])
                for m0, m1 in t:
                    mapping.append([
                        m0.replace('X', str(ri)).replace('Y', str(subri)), 
                        m1.replace('X', str(subti))])
            if (ri < 3):
                t = []
                t.append(['up_blocks.X.upsamplers.0.conv.weight','data_blocks.X.0.conv.weight'])
                t.append(['up_blocks.X.upsamplers.0.conv.bias'  ,'data_blocks.X.0.conv.bias'  ])
                for m0, m1 in t:
                    mapping.append([m0.replace('X', str(ri)), m1.replace('X', str(ti+3))])
        return mapping

    def move_last_several_layers(self):
        mapping = []
        mapping.append(['conv_norm_out.weight', 'data_blocks.29.0.0.weight', ])
        mapping.append(['conv_norm_out.bias'  , 'data_blocks.29.0.0.bias'  , ])
        mapping.append(['conv_out.weight'     , 'data_blocks.29.0.2.weight', ])
        mapping.append(['conv_out.bias'       , 'data_blocks.29.0.2.bias'  , ])
        return mapping

    def move_down_attn(self, ref=[1, 2], to=[0, 1]):
        mapping = []
        for ri, ti in zip(ref, to):
            for subri, subti in zip([0, 1], [ti, ti+1]):
                t = []
                t.append(['down_blocks.X.attentions.Y.norm.weight'   , 'context_blocks.X.0.norm.weight'   ])
                t.append(['down_blocks.X.attentions.Y.norm.bias'     , 'context_blocks.X.0.norm.bias'     ])
                t.append(['down_blocks.X.attentions.Y.proj_in.weight', 'context_blocks.X.0.proj_in.weight'])
                t.append(['down_blocks.X.attentions.Y.proj_in.bias'  , 'context_blocks.X.0.proj_in.bias'  ])  
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_q.weight'    ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_k.weight'    ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_v.weight'    ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.weight'])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.bias'  ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.ff.net.0.proj.weight' , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.weight' ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.ff.net.0.proj.bias'   , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.bias'   ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.ff.net.2.weight'      , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.weight'      ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.ff.net.2.bias'        , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.bias'        ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_q.weight'    ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_k.weight'    ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_v.weight'    ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.weight'])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.bias'  ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.norm1.weight', 'context_blocks.X.0.transformer_blocks.0.norm1.weight'])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.norm1.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm1.bias'  ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.norm2.weight', 'context_blocks.X.0.transformer_blocks.0.norm2.weight'])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.norm2.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm2.bias'  ])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.norm3.weight', 'context_blocks.X.0.transformer_blocks.0.norm3.weight'])
                t.append(['down_blocks.X.attentions.Y.transformer_blocks.0.norm3.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm3.bias'  ])
                t.append(['down_blocks.X.attentions.Y.proj_out.weight', 'context_blocks.X.0.proj_out.weight'])
                t.append(['down_blocks.X.attentions.Y.proj_out.bias'  , 'context_blocks.X.0.proj_out.bias'  ])
                for m0, m1 in t:
                    mapping.append([
                        m0.replace('X', str(ri)).replace('Y', str(subri)), 
                        m1.replace('X', str(subti))])
        return mapping

    def move_mid_attn(self, to=[6]):
        mapping = []
        for ti in to:
            t = []
            t.append(['mid_block.attentions.0.norm.weight'   , 'context_blocks.X.0.norm.weight'   ])
            t.append(['mid_block.attentions.0.norm.bias'     , 'context_blocks.X.0.norm.bias'     ])
            t.append(['mid_block.attentions.0.proj_in.weight', 'context_blocks.X.0.proj_in.weight'])
            t.append(['mid_block.attentions.0.proj_in.bias'  , 'context_blocks.X.0.proj_in.bias'  ])  
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_q.weight'    ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn1.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_k.weight'    ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn1.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_v.weight'    ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.weight'])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.bias'  ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight' , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.weight' ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.bias'   , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.bias'   ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.ff.net.2.weight'      , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.weight'      ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.ff.net.2.bias'        , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.bias'        ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_q.weight'    ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_k.weight'    ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_v.weight'    ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.weight'])
            t.append(['mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.bias'  ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.norm1.weight', 'context_blocks.X.0.transformer_blocks.0.norm1.weight'])
            t.append(['mid_block.attentions.0.transformer_blocks.0.norm1.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm1.bias'  ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.norm2.weight', 'context_blocks.X.0.transformer_blocks.0.norm2.weight'])
            t.append(['mid_block.attentions.0.transformer_blocks.0.norm2.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm2.bias'  ])
            t.append(['mid_block.attentions.0.transformer_blocks.0.norm3.weight', 'context_blocks.X.0.transformer_blocks.0.norm3.weight'])
            t.append(['mid_block.attentions.0.transformer_blocks.0.norm3.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm3.bias'  ])
            t.append(['mid_block.attentions.0.proj_out.weight', 'context_blocks.X.0.proj_out.weight'])
            t.append(['mid_block.attentions.0.proj_out.bias'  , 'context_blocks.X.0.proj_out.bias'  ])
            for m0, m1 in t:
                mapping.append([m0, m1.replace('X', str(ti))])
        return mapping

    def move_up_attn(self, ref=[1], to=[7]):
        mapping = []
        for ri, ti in zip(ref, to):
            for subri, subti in zip([0, 1, 2], [ti, ti+1, ti+2]):
                t = []
                t.append(['up_blocks.X.attentions.Y.norm.weight'   , 'context_blocks.X.0.norm.weight'   ])
                t.append(['up_blocks.X.attentions.Y.norm.bias'     , 'context_blocks.X.0.norm.bias'     ])
                t.append(['up_blocks.X.attentions.Y.proj_in.weight', 'context_blocks.X.0.proj_in.weight'])
                t.append(['up_blocks.X.attentions.Y.proj_in.bias'  , 'context_blocks.X.0.proj_in.bias'  ])  
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_q.weight'    ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_k.weight'    ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn1.to_v.weight'    ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.weight'])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn1.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn1.to_out.0.bias'  ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.ff.net.0.proj.weight' , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.weight' ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.ff.net.0.proj.bias'   , 'context_blocks.X.0.transformer_blocks.0.ff.net.0.proj.bias'   ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.ff.net.2.weight'      , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.weight'      ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.ff.net.2.bias'        , 'context_blocks.X.0.transformer_blocks.0.ff.net.2.bias'        ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_q.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_q.weight'    ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_k.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_k.weight'    ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_v.weight'    , 'context_blocks.X.0.transformer_blocks.0.attn2.to_v.weight'    ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_out.0.weight', 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.weight'])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.attn2.to_out.0.bias'  , 'context_blocks.X.0.transformer_blocks.0.attn2.to_out.0.bias'  ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.norm1.weight', 'context_blocks.X.0.transformer_blocks.0.norm1.weight'])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.norm1.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm1.bias'  ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.norm2.weight', 'context_blocks.X.0.transformer_blocks.0.norm2.weight'])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.norm2.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm2.bias'  ])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.norm3.weight', 'context_blocks.X.0.transformer_blocks.0.norm3.weight'])
                t.append(['up_blocks.X.attentions.Y.transformer_blocks.0.norm3.bias'  , 'context_blocks.X.0.transformer_blocks.0.norm3.bias'  ])
                t.append(['up_blocks.X.attentions.Y.proj_out.weight', 'context_blocks.X.0.proj_out.weight'])
                t.append(['up_blocks.X.attentions.Y.proj_out.bias'  , 'context_blocks.X.0.proj_out.bias'  ])
                for m0, m1 in t:
                    mapping.append([
                        m0.replace('X', str(ri)).replace('Y', str(subri)), 
                        m1.replace('X', str(subti))])
        return mapping

    def get_mapping(self):
        r = []
        d = []
        d += self.move_tembed_blocks()
        d += self.move_first_several_layers()
        d += self.move_down_resblocks([0, 1, 2, 3], [1, 4, 7, 10])
        d += self.move_mid_resblocks([12])
        d += self.move_up_resblocks([0, 1, 2, 3], [14, 18, 22, 26])
        d += self.move_last_several_layers()

        for dfrom, dto in d:
            r.append([dfrom, 'diffuser.image.'+dto])        
        c = []
        c += self.move_down_attn([0, 1, 2], [0, 2, 4])
        c += self.move_mid_attn([6])
        c += self.move_up_attn([1, 2, 3], [7, 10, 13])
        for cfrom, cto in c:
            r.append([cfrom, 'diffuser.text.'+cto])
        return r
 
    def __call__(self, sd):
        newsd = []
        mapping = self.get_mapping()
        for mfrom, mto in mapping:
            newsd.append([mto, sd[mfrom]])
        return OrderedDict(newsd)

class sdhuggingface_vae_to_pfd_mover():
    def __init__(self):
        pass

    def move_encoder_convin(self):
        mapping = []
        mapping.append(['conv_in.weight', 'conv_in.weight'])
        mapping.append(['conv_in.bias'  , 'conv_in.bias'  ])
        return mapping

    def move_encoder_down(self, ref=[0, 1, 2, 3], to=[0, 1, 2, 3]):
        mapping = []
        for ri, ti in zip(ref, to):
            for subri, subti in zip([0, 1], [0, 1]):
                t = []
                t.append(['X.resnets.Y.norm1.weight', 'X.block.Y.norm1.weight' ])
                t.append(['X.resnets.Y.norm1.bias'  , 'X.block.Y.norm1.bias'   ])
                t.append(['X.resnets.Y.conv1.weight', 'X.block.Y.conv1.weight' ])
                t.append(['X.resnets.Y.conv1.bias'  , 'X.block.Y.conv1.bias'   ])
                t.append(['X.resnets.Y.norm2.weight', 'X.block.Y.norm2.weight'])
                t.append(['X.resnets.Y.norm2.bias'  , 'X.block.Y.norm2.bias'  ])
                t.append(['X.resnets.Y.conv2.weight', 'X.block.Y.conv2.weight'])
                t.append(['X.resnets.Y.conv2.bias'  , 'X.block.Y.conv2.bias'  ])
                if (ri > 0) and (ri < 3) and (subri == 0):
                    t.append(['X.resnets.Y.conv_shortcut.weight', 'X.block.Y.nin_shortcut.weight'])
                    t.append(['X.resnets.Y.conv_shortcut.bias'  , 'X.block.Y.nin_shortcut.bias'  ])
                if (ri < 3) and (subri == 1):
                    t.append(['X.downsamplers.0.conv.weight', 'X.downsample.conv.weight'])
                    t.append(['X.downsamplers.0.conv.bias'  , 'X.downsample.conv.bias'  ])

                for m0, m1 in t:
                    mapping.append([
                        m0.replace('X', str(ri)).replace('Y', str(subri)), 
                        m1.replace('X', str(ti)).replace('Y', str(subti)), ])
        return mapping

    def move_encoder_mid(self):
        mapping = []
        for subri, subti in zip([0, 1], [1, 2]):
            t = []
            t.append(['resnets.Y.norm1.weight', 'block_Y.norm1.weight' ])
            t.append(['resnets.Y.norm1.bias'  , 'block_Y.norm1.bias'   ])
            t.append(['resnets.Y.conv1.weight', 'block_Y.conv1.weight' ])
            t.append(['resnets.Y.conv1.bias'  , 'block_Y.conv1.bias'   ])
            t.append(['resnets.Y.norm2.weight', 'block_Y.norm2.weight'])
            t.append(['resnets.Y.norm2.bias'  , 'block_Y.norm2.bias'  ])
            t.append(['resnets.Y.conv2.weight', 'block_Y.conv2.weight'])
            t.append(['resnets.Y.conv2.bias'  , 'block_Y.conv2.bias'  ])
            f = lambda x : x[:, :, None, None]
            if subri == 0:
                t.append(['attentions.0.group_norm.weight', 'attn_1.norm.weight'    ])
                t.append(['attentions.0.group_norm.bias'  , 'attn_1.norm.bias'      ])
                t.append(['attentions.0.query.weight'     , 'attn_1.q.weight'       , f])
                t.append(['attentions.0.query.bias'       , 'attn_1.q.bias'         ])
                t.append(['attentions.0.key.weight'       , 'attn_1.k.weight'       , f])
                t.append(['attentions.0.key.bias'         , 'attn_1.k.bias'         ])
                t.append(['attentions.0.value.weight'     , 'attn_1.v.weight'       , f])
                t.append(['attentions.0.value.bias'       , 'attn_1.v.bias'         ])
                t.append(['attentions.0.proj_attn.weight' , 'attn_1.proj_out.weight', f])
                t.append(['attentions.0.proj_attn.bias'   , 'attn_1.proj_out.bias'  ])
            for m in t:
                if len(m) == 2:
                    mapping.append([
                        m[0].replace('Y', str(subri)), 
                        m[1].replace('Y', str(subti)), ])
                else:
                    mapping.append([
                        m[0].replace('Y', str(subri)), 
                        m[1].replace('Y', str(subti)), 
                        m[2], ])
        return mapping

    def move_encoder_out(self):
        mapping = []
        mapping.append(['conv_norm_out.weight', 'norm_out.weight'])
        mapping.append(['conv_norm_out.bias'  , 'norm_out.bias'  ])
        mapping.append(['conv_out.weight', 'conv_out.weight'])
        mapping.append(['conv_out.bias'  , 'conv_out.bias'  ])
        return mapping

    def move_decoder_convin(self):
        return self.move_encoder_convin()

    def move_decoder_mid(self):
        return self.move_encoder_mid()

    def move_decoder_up(self, ref=[3, 2, 1, 0], to=[0, 1, 2, 3]):
        mapping = []
        for ri, ti in zip(ref, to):
            for subri, subti in zip([0, 1, 2], [0, 1, 2]):
                t = []
                t.append(['X.resnets.Y.norm1.weight', 'X.block.Y.norm1.weight' ])
                t.append(['X.resnets.Y.norm1.bias'  , 'X.block.Y.norm1.bias'   ])
                t.append(['X.resnets.Y.conv1.weight', 'X.block.Y.conv1.weight' ])
                t.append(['X.resnets.Y.conv1.bias'  , 'X.block.Y.conv1.bias'   ])
                t.append(['X.resnets.Y.norm2.weight', 'X.block.Y.norm2.weight'])
                t.append(['X.resnets.Y.norm2.bias'  , 'X.block.Y.norm2.bias'  ])
                t.append(['X.resnets.Y.conv2.weight', 'X.block.Y.conv2.weight'])
                t.append(['X.resnets.Y.conv2.bias'  , 'X.block.Y.conv2.bias'  ])
                if (ri in [2, 3]) and (subri==0):
                    t.append(['X.resnets.Y.conv_shortcut.weight', 'X.block.Y.nin_shortcut.weight'])
                    t.append(['X.resnets.Y.conv_shortcut.bias'  , 'X.block.Y.nin_shortcut.bias'  ])
                if ri in [0, 1, 2]:
                    t.append(['X.upsamplers.0.conv.weight', 'X.upsample.conv.weight'])
                    t.append(['X.upsamplers.0.conv.bias'  , 'X.upsample.conv.bias'  ])

                for m0, m1 in t:
                    mapping.append([
                        m0.replace('X', str(ri)).replace('Y', str(subri)), 
                        m1.replace('X', str(ti)).replace('Y', str(subti)), ])
        return mapping

    def move_decoder_out(self):
        return self.move_encoder_out()

    def move_rest(self):
        mapping = []
        mapping.append(['quant_conv.weight'     , 'quant_conv.weight'     ])
        mapping.append(['quant_conv.bias'       , 'quant_conv.bias'       ])
        mapping.append(['post_quant_conv.weight', 'post_quant_conv.weight'])
        mapping.append(['post_quant_conv.bias'  , 'post_quant_conv.bias'  ])
        return mapping

    def get_mapping(self):
        r = []
        d = self.move_encoder_convin()
        for dfrom, dto in d:
            r.append(['encoder.'+dfrom, 'encoder.'+dto])
        d = self.move_encoder_down([0, 1, 2, 3], [0, 1, 2, 3])
        for dfrom, dto in d:
            r.append(['encoder.down_blocks.'+dfrom, 'encoder.down.'+dto])
        d = self.move_encoder_mid()
        for dd in d:
            dfrom, dto = dd[0:2]
            if len(dd) == 2:
                r.append(['encoder.mid_block.'+dfrom, 'encoder.mid.'+dto])
            elif len(dd) == 3:
                r.append(['encoder.mid_block.'+dfrom, 'encoder.mid.'+dto, dd[2]])
        d = self.move_encoder_out()
        for dfrom, dto in d:
            r.append(['encoder.'+dfrom, 'encoder.'+dto])

        d = self.move_decoder_convin()
        for dfrom, dto in d:
            r.append(['decoder.'+dfrom, 'decoder.'+dto])
        d = self.move_decoder_mid()
        for dd in d:
            dfrom, dto = dd[0:2]
            if len(dd) == 2:
                r.append(['decoder.mid_block.'+dfrom, 'decoder.mid.'+dto])
            elif len(dd) == 3:
                r.append(['decoder.mid_block.'+dfrom, 'decoder.mid.'+dto, dd[2]])
        d = self.move_decoder_up([3, 2, 1, 0], [0, 1, 2, 3])
        for dfrom, dto in d:
            r.append(['decoder.up_blocks.'+dfrom, 'decoder.up.'+dto])
        d = self.move_decoder_out()
        for dfrom, dto in d:
            r.append(['decoder.'+dfrom, 'decoder.'+dto])
        d = self.move_rest()
        for dfrom, dto in d:
            r.append([dfrom, dto])
        return r
 
    def __call__(self, sd):
        newsd = []
        mapping = self.get_mapping()
        for minfo in mapping:
            if len(minfo) == 2:
                mfrom, mto = minfo
                newsd.append([mto, sd[mfrom]])
            else:
                mfrom, mto, f = minfo
                if f is not None:
                    newsd.append([mto, f(sd[mfrom])])
        return OrderedDict(newsd)

class sdhuggingface_ctx_to_pfd_mover():
    def __init__(self):
        pass

    def __call__(self, sd):
        newsd = []
        for ki, vi in sd.items():
            newsd.append(['transformer.'+ki, vi])
        return OrderedDict(newsd)

if __name__ == '__main__':

    path_in            = 'pretrained/xx_sdwebui.pth'
    path_out_diffuser  = 'pretrained/xx_diffuser.safetensors'

    mover_diffuser = sdwebui_diffuser_to_pfd_mover()

    if osp.splitext(path_in)[1] == '.pth':
        sd = torch.load(path_in)
    elif osp.splitext(path_in)[1] == '.ckpt':
        sd = torch.load(path_in)['state_dict']
    elif osp.splitext(path_in)[1] == '.safetensors':
        sd = load_file(path_in, "cpu")
        sd = OrderedDict(sd)
    else:
        raise ValueError

    sdnew = mover_diffuser(sd)
    save_file(OrderedDict(sdnew), path_out_diffuser)
