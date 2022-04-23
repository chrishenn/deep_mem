


########################################################################################################################
########################################################################################################################
########################################################################################################################
###### OOLayer


def find_geom(pts_geom, lin_radius, scale_radius):
    pts, imgid, edges, batch_size = pts_geom

    edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, lin_radius, scale_radius, batch_size)[0]
    edges = t.cat([edges, t.stack([edges[:,1],edges[:,0]], 1)])
    edges = t.cat([edges, t.arange(pts.size(0),device=edges.device)[:,None].repeat(1,2)])

    ## relative angles
    src_angles, dst_angles = pts[:,3][edges[:, 0]], pts[:,3][edges[:, 1]]
    diff_ang = dst_angles - src_angles
    diff_ang_leaf = t.stack([diff_ang.sin(), diff_ang.cos()], 1)

    ## relative distances
    locs_src, locs_dst = pts[:, :2][edges[:, 0]], pts[:, :2][edges[:, 1]]
    diff_yx = (locs_dst - locs_src)

    lin_ratio = pts[:, 4][edges[:, 1]].mul(pts[:, 4][edges[:, 0]])
    lin_ratio = t.where(lin_ratio.lt(1), t.ones_like(lin_ratio), lin_ratio)
    diff_mag = F.pairwise_distance(locs_dst, locs_src).div(lin_radius).div(lin_ratio)
    diff_mag_leaf = t.stack([diff_mag.sin(), diff_mag.cos()], 1).sub(0.5).mul(2)

    ## rotated relative vec-angles
    diff_rot = t.atan2(diff_yx[:,0], diff_yx[:,1] +1e-5)
    diff_rot = t.where(edges[:,0].eq(edges[:,1]), src_angles, diff_rot)  ## remove image-ref'd angles due to self-edges
    diff_rot = diff_rot - src_angles
    diff_rot_leaf = t.stack([diff_rot.sin(), diff_rot.cos()], 1)

    pts_geom = pts, imgid, edges, batch_size
    leaf_geom = diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, None
    return pts_geom, leaf_geom

class OOLayer_0(nn.Module):
    def __init__(self, opt, radius_factor, chan_in, chan_out, ptwise=True, bn=True, relu=True):
        super().__init__()

        if radius_factor is not None:
            lin_radius = radius_factor + 0.05
            self.register_buffer('lin_radius',   t.tensor(lin_radius, dtype=t.float))
            self.register_buffer('scale_radius', t.tensor(1.0, dtype=t.float))
        if bn: self.bn = nn.BatchNorm1d(chan_out, track_running_stats=False)
        if relu: self.relu = nn.ReLU()

        std = 0.1
        self.register_parameter('center_kern', nn.Parameter(t.empty([1, chan_out])) )
        nn.init.normal_(self.center_kern.data, mean=0, std=std)
        self.register_parameter('kernel_bias', nn.Parameter(t.zeros([1, chan_out])))

        dtex_in = chan_in
        if ptwise:
            self.pt_wise = nn.Linear(chan_in,chan_out)
            nn.init.normal_(self.pt_wise.weight.data, mean=0, std=std)
            dtex_in = chan_out

        self.diff_tex = nn.Linear(dtex_in,chan_out)
        self.diff_mag = nn.Linear(2,chan_out)
        self.diff_ang = nn.Linear(2,chan_out)
        self.diff_rot = nn.Linear(2,chan_out)

        self.register_parameter('scale_1', nn.Parameter(t.ones(1,dtex_in)))
        self.register_parameter('scale_2', nn.Parameter(t.ones(1,2)))
        self.register_parameter('scale_3', nn.Parameter(t.ones(1,2)))
        self.register_parameter('scale_4', nn.Parameter(t.ones(1,2)))
        self.register_parameter('bias_1', nn.Parameter(t.zeros(1,dtex_in)))
        self.register_parameter('bias_2', nn.Parameter(t.zeros(1,2)))
        self.register_parameter('bias_3', nn.Parameter(t.zeros(1,2)))
        self.register_parameter('bias_4', nn.Parameter(t.zeros(1,2)))

        self.diff_1 = nn.Linear(chan_out, chan_out)
        self.diff_2 = nn.Linear(chan_out, chan_out)
        self.diff_3 = nn.Linear(chan_out, chan_out)
        self.diff_4 = nn.Linear(chan_out, chan_out)
        nn.init.normal_(self.diff_1.weight.data, mean=0, std=std)
        nn.init.normal_(self.diff_2.weight.data, mean=0, std=std)
        nn.init.normal_(self.diff_3.weight.data, mean=0, std=std)
        nn.init.normal_(self.diff_4.weight.data, mean=0, std=std)

    def set_debug_(self, flag): self.debug = flag

    def forward(self, data):

        textures, pts_geom, leaf_geom = data

        if hasattr(self, 'lin_radius'):
            pts_geom, leaf_geom = find_geom(pts_geom, self.lin_radius, self.scale_radius)

        orig_tex, pred_tex, target_tex, class_tex = textures
        pts, imgid, edges, batch_size = pts_geom
        diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, _ = leaf_geom

        if hasattr(self, 'pt_wise'):
            class_tex = self.pt_wise(class_tex)

        ## src textures
        src_tex, dst_tex = class_tex[edges[:, 0]], class_tex[edges[:, 1]]
        diff_tex = dst_tex.sub(src_tex).mul(self.scale_1).add(self.bias_1)
        diff_tex = self.diff_tex(diff_tex.abs())

        ## relative angles
        diff_ang = diff_ang_leaf.mul(self.scale_2).add(self.bias_2)
        diff_ang = self.diff_ang(diff_ang)

        ## relative distances
        diff_mag = diff_mag_leaf.mul(self.scale_3).add(self.bias_3)
        diff_mag = self.diff_mag(diff_mag)

        ## rotated relative vec-angles
        diff_rot = diff_rot_leaf.mul(self.scale_4).add(self.bias_4)
        diff_rot = self.diff_rot(diff_rot)

        ## chain integration
        descr = self.diff_1(diff_ang)
        descr = self.diff_2(descr * diff_mag)
        descr = self.diff_3(descr * diff_rot)
        descr = self.diff_4(descr * diff_tex)
        diff_act = descr * src_tex

        ## center of kernel
        center_act = class_tex * self.center_kern

        ## pool
        diff_act = diff_act.to(center_act.dtype)
        tex_active = center_act.index_add(0, edges[:,1], diff_act)
        class_tex = tex_active + self.kernel_bias

        ## av_pool
        counts = t.ones_like(class_tex[:,0]).index_add_(0, edges[:,1], t.ones_like(edges[:,1]).float())
        class_tex = class_tex.div(counts[:,None])

        ## bn + relu
        if hasattr(self, 'bn'): class_tex = self.bn(class_tex)
        if hasattr(self, 'relu'): class_tex = self.relu(class_tex)

        ## repack
        textures = orig_tex, pred_tex, target_tex, class_tex
        leaf_geom = diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, None
        return textures, pts_geom, leaf_geom




########################################################################################################################
########################################################################################################################
########################################################################################################################
###### OOLayer Experimental

class OOLayer_Ring(nn.Module):
    def __init__(self, opt, radius_factor, chan_in, chan_out, ptwise=True, bn=True, relu=True):
        super().__init__()
        std = 0.1

        if radius_factor is not None:
            lin_radius = radius_factor + 0.05
            self.register_buffer('lin_radius',   t.tensor(lin_radius, dtype=t.float))
            self.register_buffer('scale_radius', t.tensor(1.0, dtype=t.float))
        if ptwise:
            self.pt_wise = nn.Linear(chan_in,chan_out)
            nn.init.normal_(self.pt_wise.weight.data, mean=0, std=std)
        if bn: self.bn = nn.BatchNorm1d(chan_out, track_running_stats=False)
        if relu: self.relu = nn.ReLU()

        self.register_parameter('center_kern', nn.Parameter(t.empty([1, chan_out])) )
        nn.init.normal_(self.center_kern.data, mean=0, std=std)
        self.register_parameter('ring_kern', nn.Parameter(t.empty([1, chan_out])) )
        nn.init.normal_(self.ring_kern.data, mean=0, std=std)
        self.register_parameter('kernel_bias', nn.Parameter(t.zeros([1, chan_out])))

    def set_debug_(self, flag): self.debug = flag

    def forward(self, data):

        textures, pts_geom, leaf_geom = data
        if hasattr(self, 'lin_radius'): pts_geom, leaf_geom = find_geom(pts_geom, self.lin_radius, self.scale_radius)

        orig_tex, pred_tex, target_tex, class_tex = textures
        pts, imgid, edges, batch_size = pts_geom
        diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, _ = leaf_geom

        if hasattr(self, 'pt_wise'): class_tex = self.pt_wise(class_tex)

        ## src textures
        src_tex = class_tex[edges[:, 0]]

        ## center of kernel
        center_act = class_tex * self.center_kern

        ## ring kernel
        diff_act = src_tex * self.ring_kern

        ## pool
        tex_active = center_act.index_add(0, edges[:,1], diff_act)
        class_tex = tex_active + self.kernel_bias

        ## av_pool
        counts = t.ones_like(class_tex[:,0]).index_add_(0, edges[:,1], t.ones_like(edges[:,1]).float())
        class_tex = class_tex.div(counts[:,None])

        ## bn + relu
        if hasattr(self, 'bn'): class_tex = self.bn(class_tex)
        if hasattr(self, 'relu'): class_tex = self.relu(class_tex)

        ## repack
        textures = orig_tex, pred_tex, target_tex, class_tex
        leaf_geom = diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, None
        return textures, pts_geom, leaf_geom


## using pdesc as geometric descriptor
class OOLayer(nn.Module):
    def __init__(self, opt, radius_factor, chan_in, chan_out, pdesc_size, ptwise=True, bn=True, relu=True):
        super().__init__()
        std = 0.1

        if radius_factor is not None:
            lin_radius = radius_factor + 0.05
            self.register_buffer('lin_radius',   t.tensor(lin_radius, dtype=t.float))
            self.register_buffer('scale_radius', t.tensor(1.0, dtype=t.float))
        if ptwise:
            self.pt_wise = nn.Linear(chan_in,chan_out)
            nn.init.normal_(self.pt_wise.weight.data, mean=0, std=std)
        if bn: self.bn = nn.BatchNorm1d(chan_out, track_running_stats=False)
        if relu: self.relu = nn.ReLU()

        self.register_parameter('center_kern', nn.Parameter(t.empty([1, chan_out])) )
        nn.init.normal_(self.center_kern.data, mean=0, std=std)
        self.register_parameter('kernel_bias', nn.Parameter(t.zeros([1, chan_out])))

        self.diff_1 = nn.Linear(pdesc_size, chan_out)
        nn.init.normal_(self.diff_1.weight.data, mean=0, std=std)

    def set_debug_(self, flag): self.debug = flag

    def forward(self, data):

        class_tex, pts, imgid, batch_size, pdesc, edges = data
        if hasattr(self, 'pt_wise'): class_tex = self.pt_wise(class_tex)

        edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, self.lin_radius, self.scale_radius, batch_size)[0]
        edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])
        edges = t.cat([edges, t.arange(pts.size(0), device=edges.device)[:, None].repeat(1, 2)])

        ## src textures
        src_tex = class_tex[edges[:, 0]]

        ## center of kernel
        center_act = class_tex * self.center_kern

        ##
        diff_geom = pdesc[edges[:,1]] - pdesc[edges[:,0]]
        descr = self.diff_1(diff_geom)
        diff_act = src_tex * descr

        ## pool
        tex_active = center_act.index_add(0, edges[:,1], diff_act)
        class_tex = tex_active + self.kernel_bias

        ## av_pool
        counts = t.ones_like(class_tex[:,0]).index_add_(0, edges[:,1], t.ones_like(edges[:,1]).float())
        class_tex = class_tex.div(counts[:,None])

        ## bn + relu
        if hasattr(self, 'bn'): class_tex = self.bn(class_tex)
        if hasattr(self, 'relu'): class_tex = self.relu(class_tex)

        return class_tex, pts, imgid, batch_size, pdesc, edges



########################################################################################################################
########################################################################################################################
########################################################################################################################
###### Grouper - Classify objects into groups and merge them

class ClassGrouper(nn.Module):
    ''' Generate group hypotheses on an object-by-object basis'''

    def __init__(self, opt, chan_in, n_groups, c_out):
        super().__init__()
        self.debug = False
        self.n_groups = n_groups

        self.fc = nn.Linear(chan_in, n_groups, bias=False)
        nn.init.uniform_(self.fc.weight.data, -1, 1)

        self.bn = nn.BatchNorm1d(n_groups, track_running_stats=False)
        self.bn1 = nn.BatchNorm1d(c_out, track_running_stats=False)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()

        self.chan_up = nn.Linear(n_groups, c_out, bias=False)

    def set_debug_(self, flag): self.debug = flag

    def forward(self, data):
        textures, pts_geom, leaf_geom = data
        orig_tex, pred_tex, target_tex, class_tex = textures
        pts, imgid, edges, batch_size = pts_geom
        diff_ang_leaf, diff_mag_leaf, diff_rot_leaf, _ = leaf_geom

        g_tex = class_tex.clone()
        g_tex = self.fc(g_tex)
        g_tex = self.bn(g_tex)
        g_tex = self.relu(g_tex)
        _, groupids = t.max(g_tex.abs(), dim=1)
        g_tex = self.chan_up(g_tex)
        g_tex = self.bn1(g_tex)
        g_tex = self.relu1(g_tex)

        ## need g_pts for residual pool
        g_pts = pts.clone()
        g_imgid = imgid.clone()

        ## partition by image
        offsets = t.full((batch_size,), self.n_groups, device=class_tex.device)
        offsets = offsets.cumsum(0).sub(offsets)
        groupids.add_(offsets[imgid])

        ## partition by groupids that are also ccpt
        edges = edges[ groupids[edges[:,0]].eq(groupids[edges[:,1]]) ]
        groupids = get_conn_comp(edges, imgid).long()
        total_groups = imgid.size(0)

        ## compose group reps
        group_avgs = t.zeros([total_groups, class_tex.size(1)], device=class_tex.device).index_add(0, groupids, class_tex)
        counts = t.zeros(total_groups, device=class_tex.device).index_add_(0, groupids, t.ones_like(groupids).float() )
        counts = t.where(counts.lt(1), t.ones_like(counts), counts)
        class_tex = group_avgs.div(counts[:,None])

        imgid = t.zeros(total_groups, dtype=t.long,device=class_tex.device).sub(1).index_put_((groupids,), imgid)

        ## assumes total_groups < imgid.size(0) - i.e. using partition by ccpt
        angles = pts[:,3].clone()
        angles = t.stack([angles.sin(), angles.cos()], 1)
        angles = t.zeros_like(angles).index_add_(0, groupids, angles)
        angles = t.atan2(angles[:,0], angles[:,1] + 1e-5)

        sizes = pts[:,4].clone().pow(2)
        sizes = t.zeros_like(sizes).index_add_(0, groupids, sizes)
        sizes.sqrt_()

        pts = t.zeros([total_groups, pts.size(1)], device=class_tex.device).index_add_(0, groupids, pts)
        pts.div_(counts[:,None])

        pts[:,3] = angles
        pts[:,4] = sizes

        have_member = imgid.ne(-1)
        class_tex, pts, imgid = class_tex[have_member], pts[have_member], imgid[have_member]

        if self.debug:
            data = textures[:-1] + (class_tex,), (pts, imgid, None, batch_size), leaf_geom
            return data, (g_tex,g_pts,g_imgid,groupids,edges)
        else:
            data = textures[:-1] + (class_tex,), (pts, imgid, None, batch_size), leaf_geom
            return data, (g_tex, g_pts, g_imgid)





########################################################################################################################
########################################################################################################################
########################################################################################################################
###### Pool

class OOPool(nn.Module):
    def __init__(self, opt, radius_factor, frac):
        super().__init__()
        self.frac = frac

        if radius_factor is not None:
            lin_radius = radius_factor + 0.05

            self.register_buffer('lin_radius', t.tensor(lin_radius, dtype=t.float))
            self.register_buffer('scale_radius', t.tensor(1.0, dtype=t.float))

    def set_debug_(self, flag): self.debug = flag

    def forward(self, data):
        class_tex, pts, imgid, batch_size, pdesc, edges = data

        ## filter out self-edges. Edges remaining are bi-directional
        if hasattr(self, 'lin_radius'):
            edges = t.ops.my_ops.frnn_ts_kernel(pts, imgid, self.lin_radius, self.scale_radius, batch_size)[0]
            edges = t.cat([edges, t.stack([edges[:, 1], edges[:, 0]], 1)])
        else:
            edges = edges[edges[:, 0].ne(edges[:, 1])]

        o_act = class_tex.norm(dim=1).clone().detach()

        ## avg rt->lf; rebalance by dividing sent textures by the # of times they've sent
        neb_avg = t.zeros_like(class_tex[:, 0]).index_add_(0, edges[:, 0], o_act[edges[:, 1]])
        counts = t.zeros_like(class_tex[:, 0]).index_add_(0, edges[:, 0], t.ones_like(edges[:, 0]).float())
        counts = t.where(counts.lt(1), t.ones_like(counts), counts)
        neb_avg.div_(counts)

        ## I survive if my act is greater than the avg of my neb's acts
        live_mask = o_act > (neb_avg * self.frac)

        ## edges with dying objects at left and survivors at right
        send_mask = (live_mask.logical_not()[edges[:, 0]]) & (live_mask[edges[:, 1]])
        send_edges = edges[send_mask]

        ## pool dying-object texture into their neighbors that survive
        class_tex = class_tex.index_add(0, send_edges[:, 1], class_tex[send_edges[:, 0]])
        class_tex = class_tex.div(counts[:, None])

        class_tex, pts, imgid = class_tex[live_mask], pts[live_mask], imgid[live_mask]

        return class_tex, pts, imgid, batch_size, pdesc, edges




