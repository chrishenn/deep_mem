import os

import torch as t
import torch.nn as nn

import sparse
import numpy as np

import object_utils as o_utils
import model.object_samplers as samplers
import datasets.artificial_dataset as art_dset

t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/cuda_lib/frnn_opt_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/cuda_lib/write_row/build/libwrite_row.so")




####
## Slightly more complex Deep Mem model with absolute relationships, and sparse.COO matrix for memory storage
####



############################################################################################################
#### Deep Mem Storage Model
#       using scipy.sparse.COO sparse vector
#       pixel relationships described in absolute image-coordinates.
############################################################################################################

class Deep_Mem_AbsRelate_SparseCOO(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) / 2
        self.train_flag = True
        self.opt = opt

        self.register_buffer('lin_rad', t.tensor(8))   ## rad 8: nebs 193, rad 16: nebs 780

        mem_width = mw = 16
        self.register_buffer('mem_width', t.tensor(mem_width) )
        mem_size = [2, mw,mw,2, mw,mw,2, mw,mw,2, mw,mw,2, mw,mw,2]
        self.register_buffer('mem_size', t.tensor(mem_size) )
        n_nebs_hashed = (len(self.mem_size)-1) // 3
        self.register_buffer('n_nebs_hashed', t.tensor(n_nebs_hashed) )

        self.mem = sparse.COO([], [], shape=mem_size)

        self.rel_vec_width = rel_vec_width = (self.n_nebs_hashed * 3) + 1
        rel_vec = t.zeros([opt.batch_size * 812, rel_vec_width], dtype=t.long)
        self.register_buffer('rel_vec', rel_vec)

        write_cols = t.ones([opt.batch_size * 812], dtype=t.int)
        self.register_buffer('write_cols', write_cols)

        self.calls = 0

        ## set module mode for this execution run
        if self.opt.mode == 'store':
            self._forward = self.store_sparse_coo
        elif self.opt.mode == 'recall':
            self._forward = self.recall_sparse_coo


    def forward(self, batch):
        return self._forward(batch)

    def shutdown(self):
        pass


    ## Build relationship vectors between pixels, using oodl objects
    def build_relationship_vecs(self, batch):
        if isinstance(batch, tuple): batch, sizes = batch
        else: sizes = t.zeros([1],dtype=t.int)

        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1

        if not hasattr(self, 'sampler'): self.sampler = samplers.OOSampler(self.opt).cuda()

        data = self.sampler(batch, sizes)
        tex, pts, imgid, batch_size = data

        if not hasattr(self, 'edges'):
            ## we're only putting a few nebs into mem - constrained by n_nebs_hashed
            edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, self.lin_rad, t.tensor(1), batch_size)[0]
            edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])
            self.edges = edges

            locs = pts[:, :2].clone()
            locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
            vecs = locs_rt - locs_lf
            vecs.div_(self.lin_rad)
            vecs.add_(1)
            vecs.div_(2)

            ## vecs in rel_vec will bin to [0, mem_width)
            vecs.mul_(self.mem_width-1)
            vecs.round_()
            vecs = vecs.to( self.rel_vec.dtype )

            self.vecs = vecs
        else:
            edges = self.edges
            vecs = self.vecs

        tex = tex.round().to( self.rel_vec.dtype )
        tex_lf, tex_rt = tex[edges[:,0]], tex[edges[:,1]]

        t.ops.row_op.write_row_bind(self.rel_vec, self.write_cols, tex, tex_lf, tex_rt, vecs, edges[:,0].contiguous())

        return (tex,pts,imgid), edges, vecs, self.rel_vec

    ## Store a batch of images in sparse-COO memory
    def store_sparse_coo(self, batch):
        if isinstance(batch, tuple): batch, sizes = batch
        else: sizes = t.zeros([1], dtype=t.int)

        (tex, pts, imgid), edges, vecs, rel_vec = self.build_relationship_vecs(batch)

        ## store in sparse 'memory' self.mem
        mem_to_add = sparse.COO(coords=rel_vec.clone().T.cpu().numpy(), data=1, shape=tuple(te.item() for te in self.mem_size))
        self.mem += mem_to_add

    ## Recall: build the most likely image from memory, given some seed image.
    def recall_sparse_coo(self, batch):

        seed = art_dset.make_topbox(self.opt).to(batch.device)
        # seed = t.zeros_like(batch)

        ## composite many most-likely pixels into a prediction image
        comp = None
        for i in range(100): comp = self.recall_single_pred_increase_likelihood(seed.clone(), comp)

        ## compare original seed and composite images
        o_utils.tensor_imshow(seed[0], dpi=150)
        o_utils.tensor_imshow(comp[0], dpi=150)
        o_utils.tensor_imshow(comp.add(seed)[0], dpi=150)

    ## If I can make a change to a single value in a config (set of pixel relationships) that increases the likelihood,
    ## write just that change to the neighborhood. The resulting prediction will not include the values from the seed -
    ## because changing them would not increase the likelihood of the neighborhood
    def recall_single_pred_increase_likelihood(self, batch, comp=None):

        (tex, pts, imgid), edges, vecs, rel_vec = self.build_relationship_vecs(batch)

        pred_config = rel_vec.clone()
        init_conf = self.query_many(pred_config)

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

    ## query many value-vectors from memory
    def query_many(self, queries):

        ## sparse lib
        dev = queries.device
        queries = queries.cpu().numpy()
        values = np.empty_like(queries[:,0]).astype( queries.dtype )

        for i,vec in enumerate(queries):
            values[i] = self.mem[tuple(vec)]
        values = t.from_numpy( values ).to(dev)

        return values

















