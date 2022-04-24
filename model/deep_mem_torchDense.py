import os

import torch as t
import torch.nn as nn

import object_utils as o_utils
import model.object_samplers as samplers
import datasets.artificial_dataset as art_dset

t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/cuda_lib/frnn_opt_brute/build/libfrnn_ts.so")
t.ops.load_library(os.path.split(os.path.split(__file__)[0])[0] + "/cuda_lib/write_row/build/libwrite_row.so")




####
## Simplest Deep Mem models with absolute relationships, and dense torch tensors for memory storage
####



############################################################################################################
#### Deep Mem Storage Model
#       using torch tensors as memory
#       absolute relationships, between active pixels only
############################################################################################################

class Deep_Mem_ActiveOnly(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) /2
        self.train_flag = True
        self.opt = opt

        mem = t.zeros([65,65,2, 65,65, 2])
        self.register_buffer('mem', mem)

        self._forward = self.forward_exp

        if opt.debug:
            self._forward = self.forward_debug


    def set_train(self, state):
        self.train_flag = state

    def forward(self, data): return self._forward(data)

    def forward_exp(self, batch):

        if not hasattr(self, 'sampler'): self.sampler = samplers.OOSampler(self.opt).to(self.mem.device)

        # self.store(batch)

        seed = art_dset.make_nine(self.opt).to(batch.device)
        self.recall(seed)

    def store(self, batch):
        if isinstance(batch, tuple): batch, sizes = batch
        else: sizes = t.zeros([1],dtype=t.int)

        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1

        data = self.sampler(batch, sizes)
        tex, pts, imgid, batch_size = data

        ## fully-connected in each image
        o_ids = t.arange(pts.size(0),device=pts.device)
        edges = t.cartesian_prod(o_ids, o_ids)

        ##
        edges = edges[ tex[edges[:,0]].gt(0.5).squeeze() & tex[edges[:,1]].gt(0.5).squeeze() ]

        locs = pts[:,:2]
        locs_lf, locs_rt = locs[edges[:,0]], locs[edges[:,1]]
        rel_vec = t.cat([locs_lf, locs_rt], 1)
        rel_vec.add_(32)
        rel_vec = rel_vec.round().long()
        rel_vec.clamp_(0,64)

        self.mem.index_put_( (rel_vec[:,0], rel_vec[:,1], rel_vec[:,2], rel_vec[:,3]), t.ones_like(rel_vec[:,0]).float(), accumulate=True)

    def dump_mem(self, batch):
        if isinstance(batch, tuple): batch, sizes = batch
        else: sizes = t.zeros([1],dtype=t.int)

        data = self.sampler(batch, sizes)
        _, pts, imgid, batch_size = data

        ## fully-connected in each image
        o_ids = t.arange(pts.size(0),device=pts.device)
        edges = t.cartesian_prod(o_ids, o_ids)

        locs = pts[:, :2]
        locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
        yx = locs_rt - locs_lf
        rel_vec = t.cat([locs_lf, locs_rt, yx], 1)
        rel_vec.add_(32)
        rel_vec = rel_vec.round().long()
        rel_vec.clamp_(0, 64)

        recalled_act = self.mem[ ((rel_vec[:,0], rel_vec[:,1], rel_vec[:,2], rel_vec[:,3])) ]

        locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]

        recalled_img = o_utils.regrid(recalled_act[:,None], locs_lf, imgid[edges[:,0]], self.img_size)
        recalled_img = o_utils.regrid(recalled_act[:,None], locs_rt, imgid[edges[:,0]], self.img_size, batch=recalled_img)
        o_utils.tensor_imshow(recalled_img.squeeze())

    def recall(self, batch):
        if isinstance(batch, tuple):
            batch, sizes = batch
        else:
            sizes = t.zeros([1], dtype=t.int)

        data = self.sampler(batch, sizes)
        tex, pts, imgid, batch_size = data

        locs = pts[:, :2]
        locs = locs[tex.gt(0.5).squeeze()]
        locs.add_(32)
        locs = locs.round().long()

        local_mem = self.mem[(locs[:, 0], locs[:, 1])]

        k = 30
        local_mem = local_mem.reshape(local_mem.size(0), -1)
        ravld_vals, ravld_locs = local_mem.topk(k, dim=1)

        locs_rt_y = ravld_locs.floor_divide(65)
        locs_rt_x = ravld_locs.fmod(65)

        locs_pred = t.stack([locs_rt_y.flatten(), locs_rt_x.flatten()], 1)
        vals_pred = ravld_vals.flatten()

        imgid = t.zeros_like(locs_pred[:, 0]).long()

        pred_im = o_utils.regrid(vals_pred, locs_pred.sub(32), imgid, self.img_size, avg=True)

        o_utils.tensor_imshow(batch[0])
        path = '/graphs/abs_vecs/seed_nine.png'
        # plt.savefig(path, bbox_inches='tight', pad_inches=0)

        o_utils.tensor_imshow(pred_im[0])
        path = '/home/chris/Documents/deep_mem/graphs/heatmap_nine.png'
        # plt.savefig(path, bbox_inches='tight', pad_inches=0)

        print("done")







############################################################################################################
#### Deep Mem Storage Model
#       using torch tensors as memory
#       absolute relationships, between active objects
############################################################################################################

class Deep_Mem_AbsLocs(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) /2
        self.train_flag = True
        self.opt = opt

        mem = t.zeros([65,65,2, 65,65,2])
        self.register_buffer('mem', mem)

        ## set module mode for this execution run
        if self.opt.mode == 'store':
            self._forward = self.store
        elif self.opt.mode == 'recall':
            self._forward = self.recall


    def forward(self, batch):
        return self._forward(batch)


    ## store batch in self.mem. Absolute image locations are used.
    def store(self, batch):
        if isinstance(batch, tuple): batch, sizes = batch
        else: sizes = t.zeros([1],dtype=t.int)

        if not hasattr(self, 'sampler'): self.sampler = samplers.OOSampler(self.opt).to(self.mem.device)

        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1

        data = self.sampler(batch, sizes)
        tex, pts, imgid, batch_size = data

        edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, t.tensor(16).cuda(), t.tensor(1), batch_size)[0]
        edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])

        l_tex = tex.round().long().clamp(0,1).squeeze()
        tex_lf, tex_rt = l_tex[edges[:,0]], l_tex[edges[:,1]]

        locs = pts[:,:2]
        locs_lf, locs_rt = locs[edges[:,0]], locs[edges[:,1]]
        rel_vec = t.cat([locs_lf, locs_rt], 1)
        rel_vec.add_(32)
        rel_vec = rel_vec.round().long()
        rel_vec.clamp_(0,64)

        self.mem.index_put_( (rel_vec[:,0], rel_vec[:,1], tex_lf, rel_vec[:,2], rel_vec[:,3], tex_rt), t.ones_like(rel_vec[:,0]).float(), accumulate=True)


    ## A seed generates neighboring pixels that are likely to be active, given each neighborhood's configuration.
    ## Predicted active locations are composited into a predicted image.
    def recall(self, batch):
        if isinstance(batch, tuple): batch, sizes = batch
        else: sizes = t.zeros([1], dtype=t.int)


        seed = art_dset.make_nine(self.opt).to(batch.device)

        data = self.sampler(seed, sizes)
        tex, pts, imgid, batch_size = data

        edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, t.tensor(16).cuda(), t.tensor(1), batch_size)[0]
        edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])

        l_tex = tex.round().long().clamp(0,1).squeeze()
        tex_lf = l_tex[edges[:,0]]

        locs = pts[:,:2]
        locs_lf = locs[edges[:,0]]
        rel_vec = locs_lf
        rel_vec.add_(32)
        rel_vec = rel_vec.round().long()
        rel_vec.clamp_(0,64)

        # local_mem = self.mem[ (rel_vec[:,0], rel_vec[:,1], tex_lf) ]
        local_mem = self.mem.cpu()[ (rel_vec[:,0].cpu(), rel_vec[:,1].cpu(), tex_lf.cpu()) ]

        k = 30
        local_mem = local_mem.permute(0,3,1,2)
        local_mem = local_mem.reshape(local_mem.size(0), 2, -1)
        ravld_vals, ravld_locs = local_mem.topk(k, dim=2)

        locs_rt_y = ravld_locs.floor_divide(65)
        locs_rt_x = ravld_locs.fmod(65)

        locs_pred = t.stack([locs_rt_y.flatten(), locs_rt_x.flatten()], 1)

        ## votes into bin [:,0,:] are votes for the destination pixel being 0
        ravld_vals[:,0,:].mul_(-1)
        vals_confidence = ravld_vals.flatten()

        imgid = t.zeros_like(locs_pred[:, 0]).long()

        pred_im = o_utils.regrid(vals_confidence, locs_pred.sub(32), imgid, self.img_size, avg=True)
        return pred_im





############################################################################################################
#### Deep Mem Storage Model
#       using torch tensors as memory
#       absolute relationships, between pixels, hashes projected into lower-dimensional space
############################################################################################################

class Deep_Mem_RelativeLocs_ProjectedLowerDim(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) /2
        self.train_flag = True
        self.opt = opt

        self.register_buffer('lin_rad', t.tensor(8))
        ## rad 8: nebs 193
        ## rad 16: nebs 780

        self.mem_chan = mem_chan = 7
        self.mem_width = mem_width = 20
        # self.mem_size = [mem_width] * mem_chan
        self.mem_size = [2, 10,10,2, 10,10,2, 10,10,2 ]
        self.n_nebs_hashed = n_nebs_hashed = 80

        mem = t.zeros(self.mem_size, dtype=t.long)
        self.register_buffer('mem', mem)

        self.rel_vec_width = rel_vec_width = (n_nebs_hashed * 3) + 1
        rel_vec = t.zeros([opt.batch_size*812, rel_vec_width], dtype=t.float)
        self.register_buffer('rel_vec', rel_vec)

        write_cols = t.ones([opt.batch_size*812], dtype=t.int)
        self.register_buffer('write_cols', write_cols)

        self.hash = nn.Linear(rel_vec_width, mem_chan, bias=False).requires_grad_(False)
        nn.init.uniform_(self.hash.weight.data)
        # self.hash.weight.data.mul_(0.31)
        self.hash.weight.data.mul_(0.15)

        ## set module mode for this execution run
        if self.opt.mode == 'store':
            self._forward = self.store
        elif self.opt.mode == 'recall':
            self._forward = self.recall

    def forward(self, batch): return self._forward(batch)


    #### Build relationship vectors between pixel values, in the form of oodl objects. Embed relationships in absolute
    ## image-referenced space.
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
            edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, self.lin_rad, t.tensor(1), batch_size)[0]
            edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])

            # print("max nebs: ",edges[:,0].bincount().max().item(), "nebs hashed: ", self.n_nebs_hashed)
            # assert edges[:,0].bincount().max().item() < self.n_nebs_hashed

            self.edges = edges

            locs = pts[:, :2].clone()
            locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
            vecs = locs_rt - locs_lf
            vecs.div_(self.lin_rad)
            vecs.add_(1)
            vecs.div_(2)

            self.vecs = vecs

        else:
            edges = self.edges
            vecs = self.vecs

        tex_lf, tex_rt = tex[edges[:,0]], tex[edges[:,1]]

        ## rel_vec by row: [tex here | vecy, vecx, tex_neb | vecy, vecx, tex_neb | ... ]
        t.ops.row_op.write_row_bind(self.rel_vec, self.write_cols, tex, tex_lf, tex_rt, vecs, edges[:,0].contiguous())

        return tex,pts,imgid


    #### Store
    ## place new relationships from this batch into the self.rel_vec buffer.
    ## project those relationships into lower-dim space with an fc transform.
    ## count the occurrance of the relationship by incrementing the corresponding location in self.mem
    def store(self, batch):
        self.build_relationship_vecs(batch)

        config = self.rel_vec.clone()
        hash = self.make_hash(config)

        self.mem.index_put_( hash.split(1, dim=1), t.ones_like(hash[:,0]), accumulate=True )


    def make_hash(self, config):
        hash = self.hash(config)

        # print("hash min: ", hash.min().item(), "hash max: ", hash.max().item(), "mem width: ", self.mem_width)
        # assert hash.min().ge(0) and hash.max().le(self.mem_width - 0.01)

        hash = hash.long().clamp(0, self.mem_width-1)
        return hash

    #### Recall likely images from memory, given a seed
    ## Repeatedly query the topk most likely non-zero relative pixels suggested by the seeded image. Composite those
    ## predicted relative active pixels onto a prediction image.
    def recall(self, batch):
        # seed_batch = art_dset.make_nine(self.opt).cuda()
        # seed_batch = make_vbar(self.opt).cuda()
        seed_batch = art_dset.make_tetris(self.opt).cuda()

        comp = None
        for i in range(100):
            comp = self.recall_topk(seed_batch.clone(), comp)

        o_utils.tensor_imshow(seed_batch[0], dpi=150)
        o_utils.tensor_imshow(comp[0], dpi=150)

        comp[0,0][ t.arange(comp.size(2)) > 22] = 0
        o_utils.tensor_imshow(comp[0], dpi=150)

    def recall_topk(self, batch, comp=None):
        k = 40

        data = self.build_relationship_vecs(batch)
        tex,pts,imgid = data
        config = self.rel_vec.clone()

        pred_config = config.clone()
        hash = self.make_hash(config)
        init_conf = self.mem[hash.split(1, dim=1)].squeeze()

        keep = init_conf.gt(0)
        init_conf = init_conf[keep]
        pred_config = pred_config[keep]
        locs = pts[:,:2].clone()[keep]

        confs = t.empty([pred_config.size(0),2,self.n_nebs_hashed], device=pred_config.device)
        for i,col in enumerate( range(3, self.rel_vec_width, 3) ):

            for val in [0, 1]:
                config_tmp = pred_config.clone()
                config_tmp[:, col] = val

                hash = self.make_hash(config_tmp)
                config_conf = self.mem[ hash.split(1, dim=1) ]
                confs[:,val,i] = config_conf.squeeze()

        ravl_confs, ravl_ids = confs.reshape(confs.size(0), -1).topk(k=k, dim=1)

        ## keep only where confidence in this config is > 0
        keep = ravl_confs[:,0].gt(0)
        ravl_confs, ravl_ids = ravl_confs[keep], ravl_ids[keep]
        locs = locs[keep]
        init_conf = init_conf[keep]

        ## from [obj, val, col] -> [obj, val*col]. ravl_ids index into dim=1. need to extract val as pred_val, col as its corresponding vec in rel_vec
        pred_vals = ravl_ids.floor_divide( confs.size(2) )
        pred_vals = pred_vals.mul(2).sub(1).mul(-1).mul(init_conf[:,None])
        pred_vals = t.cat( pred_vals.split(1, dim=1) )

        ## translate from i to col in rel_vec
        pred_cols = ravl_ids.fmod( confs.size(2) ).mul(3).add(1)
        pred_cols = t.cat( pred_cols.split(1, dim=1) )
        pred_cols = pred_cols.repeat(1,2)
        pred_cols[:,1].add_(1)

        pred_vecs = pred_config.repeat(k, 1).gather(1,pred_cols)
        pred_vecs.mul_(2).sub_(1).mul_(self.lin_rad)

        locs = locs.repeat(k, 1)
        pred_locs = locs.add(pred_vecs)

        ## drop values from padded entries in rel_vec
        keep = pred_vecs.add(self.lin_rad).abs()
        keep = keep[:,0].gt(0.5) | keep[:,1].gt(0.5)
        pred_locs, pred_vals = pred_locs[keep], pred_vals[keep]

        ###########################
        # keep = pred_vals.ge(0).squeeze()
        # pred_locs, pred_vals = pred_locs[keep], pred_vals[keep]

        assert pred_locs.min() >= -1e-5
        assert pred_locs.max() < self.img_size

        imgid = t.zeros_like(pred_locs[:, 0]).long()
        comp = o_utils.regrid(pred_vals.float(), pred_locs, imgid, self.img_size, avg=False, batch=comp)
        return comp





