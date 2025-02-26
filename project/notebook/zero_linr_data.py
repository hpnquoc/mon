# !/usr/bin/env python
# -*- coding: utf-8 -*-
import kornia

import mon

image_file    = mon.Path("data/zero_linr/a0952-kme_172_image.png")
ref_file      = mon.Path("data/zero_linr/a0952-kme_172_ref.png")
depth_file    = mon.Path("data/zero_linr/a0952-kme_172_depth.png")
edge_file     = mon.Path("data/zero_linr/a0952-kme_172_edge.png")
enhanced_file = mon.Path("data/zero_linr/a0952-kme_172_enhanced.jpg")
image_v_file  = mon.Path("data/zero_linr/a0952-kme_172_image_v.png")
ref_v_file    = mon.Path("data/zero_linr/a0952-kme_172_ref_v.png")
res_file      = mon.Path("data/zero_linr/a0952-kme_172_res.png")

image      = mon.read_image(path=image_file,    to_tensor=True, normalize=True)
ref        = mon.read_image(path=ref_file,      to_tensor=True, normalize=True)
enhanced   = mon.read_image(path=enhanced_file, to_tensor=True, normalize=True)
depth      = mon.read_image(path=depth_file,    to_tensor=True, normalize=True)
edge       = mon.BoundaryAwarePrior(eps=0.05, normalized=False)(depth)

image_v    = mon.rgb_to_v(image)
ref_v      = mon.rgb_to_v(ref)
enhanced_v = mon.rgb_to_v(enhanced)
# illu       = image_v / ref_v
illu       = image_v / enhanced_v
res        = illu - image_v

mon.write_image(image_v_file, image_v)
mon.write_image(ref_v_file,   ref_v)
mon.write_image(edge_file,    edge)
mon.write_image(res_file,     res)
