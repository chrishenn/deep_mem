############################################################################################################
#### Deep Mem Storage Model
#### Author: Chris Henn
############################################################################################################

import os
from numba import njit

import numpy as np
import torch as t
from torch import nn as nn

from cuda_lib.frnn_opt_brute import frnn_cpu
from model import object_utils
from model.object_samplers import OOSampler
import object_utils as o_utils

t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/cuda_lib/frnn_opt_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/cuda_lib/write_row/build/libwrite_row.so")




############################################################################################################
#### Deep Mem Storage Model
#       using torch.COO sparse tensor
############################################################################################################

# TODO: missing a torch-COO recall function
# TODO: missing the constructor for this class version
# TODO: missing the query function for these recall-compositors
class Deep_Mem(nn.Module):
    def __init__(self, opt):
        super().__init__()


        ## set module mode for this execution run
        if self.opt.mode == 'store':
            self._forward = self.store_torch_coo
        elif self.opt.mode == 'recall':
            self._forward = self.recall_torch_coo

    def forward(self, batch): return self._forward(batch)


    def filter_sample(self, batch):
        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1

        if not hasattr(self, 'sampler'): self.sampler = OOSampler(self.opt).to(self.opt.gpu_ids[0])

        sizes = t.zeros([1], dtype=t.int)
        data = self.sampler(batch, sizes)
        return data

    def filter_thresh(self, batch):
        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1
        return batch


    def store_torch_coo(self, batch):

        data, rel_vec0 = self.build_rel_one(batch)
        rolled_rel0 = self.roll_rel(rel_vec0, 5)

        data, rel_vec1 = self.build_rel_one(batch)
        rolled_rel1 = self.roll_rel(rel_vec1, 5)

        rel_vec = t.cat([rolled_rel0, rolled_rel1])

        # #### torch sparse coo
        mem_to_add = t.sparse_coo_tensor(indices=rel_vec.clone().T, values=t.ones_like(rel_vec[:, 0]).long(), size=tuple(self.mem_size))
        self.mem.add_(mem_to_add)

        self.calls += 1
        if self.calls % 10 == 0:
            self.mem = self.mem.coalesce()


    def build_rel_one(self, batch):

        data = self.filter_thresh(self, batch)
        data = self.sample(data)
        tex, pts, imgid, batch_size = data

        edges = frnn_cpu.frnn_cpu(pts.numpy(), imgid.numpy(), self.lin_rad.item())
        edges = t.from_numpy(edges)

        edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])
        edges = edges.to(self.opt.gpu_ids[0])

        ## vecs in rel_vec will bin to [0, mem_width)
        locs = pts[:, :2].clone()
        locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
        vecs = locs_rt - locs_lf
        vecs.div_(self.lin_rad)
        vecs.add_(1)
        vecs.div_(2)
        vecs.mul_(self.mem_width)
        vecs.round_()
        vecs = vecs.to( t.int32 )

        tex = tex.round().to( t.int32 )
        tex_rt = tex[edges[:,1]]

        args = [tex, tex_rt, vecs, edges[:, 0].contiguous() ]
        args = [te.numpy() for te in args]
        rel_vec = t.from_numpy( build_rel_cpu( self.rel_vec_width.item(), *args ) )

        return rel_vec, pts

    def build_rel_one_gpu(self, batch):

        data = self.filter_sample(batch)
        tex, pts, imgid, batch_size = data

        edges = t.ops.my_ops.frnn_ts_kernel(pts.cuda(), imgid.cuda(), self.lin_rad.cuda(), t.tensor(1).cuda(), batch_size.cuda())[0]
        edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])

        ## vecs in rel_vec will bin to [0, mem_width)
        locs = pts[:, :2].clone()
        locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
        vecs = locs_rt - locs_lf
        vecs.div_(self.lin_rad)
        vecs.add_(1)
        vecs.div_(2)
        vecs.mul_(self.mem_width-1)
        vecs.round_()
        vecs = vecs.to( t.int32 )

        tex = tex.round().to( t.int32 )
        tex_rt = tex[edges[:,1]]

        t.ops.row_op.write_row_bind(self.rel_vec, self.write_cols, tex.cuda(), tex_rt.cuda(), vecs.cuda(), edges[:,0].cuda().contiguous())

        return self.rel_vec.clone().cpu(), pts


    def roll_rel(self, rel_vec, pts, n_roll):

        rolled_rel = t.empty_like(rel_vec).repeat(n_roll+1, 1)

        rolled_rel[ : rel_vec.size(0), :] = rel_vec.clone()

        for i in range(1, n_roll+1):

            roll = rel_vec.clone()
            home_tex = roll[:,0]
            roll_vecs = roll[:, 1:].roll(3, dims=1)
            roll[:, 0] = home_tex
            roll[:, 1:] = roll_vecs

            rolled_rel[i * rel_vec.size(0) : (i+1) * rel_vec.size(0), :] = roll

        pts = pts.repeat(n_roll+1, 1)

        return rolled_rel, pts


    def recall_torch_coo(self, batch):
        self.mem = self.mem.coalesce()

        batch[0,0][t.arange(batch.size(2),device=batch.device).gt(self.img_size//2-2)] = 0
        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1
        seed = batch

        comp = self.recall_ann(seed, n_comp=20)

        object_utils.tensor_imshow(seed[0], dpi=150)
        object_utils.tensor_imshow(comp[0], dpi=150)
        comp_tmp = comp.cpu().numpy()[0, 0]

        comp.mean(), comp.max()

        comp_tmp = comp.clamp(0, comp.mean().item()*3)
        object_utils.tensor_imshow(comp_tmp[0], dpi=150)

        comp_tmp = comp.clone()
        comp_tmp[0,0][t.arange(comp.size(2)) > 26] = 0
        comp_tmp.clamp_(0,650)
        object_utils.tensor_imshow(comp_tmp[0], dpi=150)

        print("done")

    def recall_inc(self, batch, comp=None):

        (tex, pts, imgid), edges, vecs, rel_vec = self.build_rel_one(batch)

        pred_config = rel_vec.clone()
        init_conf = self.query_many(pred_config)

        ## if I can make a change to a single value in a config that increases the confidence, write just that change to the neighborhood
        ## the resulting prediction will not include the values from the seed - because changing them would not increase the conf of the neighborhood

        ## build configs - each with a single change
        configs = t.empty([pred_config.size(0) * self.n_nebs_hashed, pred_config.size(1)], dtype=pred_config.dtype, device=pred_config.device)

        for i, col in enumerate(range(3, self.rel_vec_width.item(), 3)):

            config_flipped = pred_config.clone()
            config_flipped[:, col] = t.where(config_flipped[:, col].eq(0), t.ones_like(config_flipped[:, col]), t.zeros_like(config_flipped[:, col]))

            configs[pred_config.size(0) * i : pred_config.size(0) * (i+1), :] = config_flipped

        confs = self.query_many(configs)

        good_flip = confs.gt( init_conf.repeat(self.n_nebs_hashed) )

        gflip_rows = t.arange(good_flip.size(0),device=good_flip.device)
        gflip_cols = t.arange(confs.size(0),device=confs.device).floor_divide(pred_config.size(0)).mul(3).add(1)[:,None].repeat(1,2)
        gflip_cols[:,1].add_(1)
        gflip_rows, gflip_cols = gflip_rows[good_flip], gflip_cols[good_flip]

        ## select vecs where a flip has increased the confidence, values that caused the increase
        vecs_0 = configs[(gflip_rows[:,None], gflip_cols)]

        ## TODO: these '0' vals are a vote for lower value at loc
        ## TODO: weight these votes by config confidence?
        vals = configs[(gflip_rows, gflip_cols[:,1].add(1))]

        ## transform relative vecs to locations
        locs = pts[:,:2].repeat(self.n_nebs_hashed, 1)[good_flip]

        vecs = vecs_0.float()
        vecs.div_(self.mem_width-1).mul_(2).sub_(1).mul_(self.lin_rad)

        pred_locs = locs.add(vecs)

        ## drop padded vecs
        not_pad = vecs_0[:,0].ne(0) | vecs_0[:,1].ne(0)
        pred_locs = pred_locs[not_pad]
        pred_vals = vals[not_pad]

        if pred_locs.size(0) > 0:
            assert pred_locs.min() >= -1e-5
            assert pred_locs.max() < self.img_size

        pred_locs.clamp_(0, self.img_size-1)

        imgid = t.zeros_like(pred_locs[:, 0]).long()
        comp = o_utils.regrid(pred_vals.float(), pred_locs, imgid, self.img_size, avg=False, batch=comp)
        return comp

    def recall_hash(self, batch, comp=None):
        (tex, pts, imgid), edges, vecs, rel_vec = self.build_rel_one(batch)

        pred_config = rel_vec.clone()
        init_conf = self.query_ann(pred_config)

        ## build configs - each with a single change
        configs = t.empty([pred_config.size(0) * self.n_nebs_hashed, pred_config.size(1)], dtype=pred_config.dtype, device=pred_config.device)

        for i, col in enumerate(range(3, self.rel_vec_width.item(), 3)):
            config_flipped = pred_config.clone()
            config_flipped[:, col] = t.where(config_flipped[:, col].eq(0), t.ones_like(config_flipped[:, col]), t.zeros_like(config_flipped[:, col]))

            configs[pred_config.size(0) * i: pred_config.size(0) * (i + 1), :] = config_flipped

        confs = self.query_ann(configs)

        good_flip = confs.gt(init_conf.repeat(self.n_nebs_hashed))

        gflip_rows = t.arange(good_flip.size(0), device=good_flip.device)
        gflip_cols = t.arange(confs.size(0), device=confs.device).floor_divide(pred_config.size(0)).mul(3).add(1)[:, None].repeat(1, 2)
        gflip_cols[:, 1].add_(1)
        gflip_rows, gflip_cols = gflip_rows[good_flip], gflip_cols[good_flip]

        ## select vecs where a flip has increased the confidence, values that caused the increase
        vecs_0 = configs[(gflip_rows[:, None], gflip_cols)]

        vals = configs[(gflip_rows, gflip_cols[:, 1].add(1))]
        vals.mul_(2).sub_(1).mul_( init_conf.repeat(self.n_nebs_hashed)[gflip_rows] )
        vals[vals.lt(0)] = 0

        ## transform relative vecs to locations
        locs = pts[:, :2].repeat(self.n_nebs_hashed, 1)[good_flip]

        vecs = vecs_0.float()
        vecs.div_(self.mem_width - 1).mul_(2).sub_(1).mul_(self.lin_rad)

        pred_locs = locs.add(vecs)

        ## drop padded vecs
        not_pad = vecs_0[:, 0].ne(0) | vecs_0[:, 1].ne(0)
        pred_locs = pred_locs[not_pad]
        pred_vals = vals[not_pad]

        if pred_locs.size(0) > 0:
            assert pred_locs.min() >= -1e-5
            assert pred_locs.max() < self.img_size

        pred_locs.clamp_(0, self.img_size - 1)

        imgid = t.zeros_like(pred_locs[:, 0]).long()
        comp = o_utils.regrid(pred_vals.float(), pred_locs, imgid, self.img_size, avg=False, batch=comp)
        return comp








############################################################################################################
#### Helpers
############################################################################################################

@njit
def build_rel_cpu(rel_vec_width, tex, tex_rt, vecs, rowids):
    write_cols = np.ones(tex.shape[0],dtype=np.int32)
    rel_vec = np.zeros( (tex.shape[0], rel_vec_width),dtype=np.int32)

    rel_vec[:,0] = tex[:,0].copy()

    n_cols = rel_vec.shape[1]
    n_vecs = vecs.shape[0]
    for vec_i in range(n_vecs):

        o_id = rowids[vec_i]
        col = write_cols[o_id]

        if col+2 >= n_cols: continue

        write_cols[o_id] += 3

        vec = vecs[vec_i]
        vec_tex = tex_rt[vec_i,0]

        rel_vec[o_id, col + 0] = vec[0]
        rel_vec[o_id, col + 1] = vec[1]
        rel_vec[o_id, col + 2] = vec_tex

    return rel_vec









































