from matplotlib import pyplot as plt

import mon

from PIL import Image, ImageFilter


def draw_figure(image, text, save_path, cmap="gray"):
	font = {
		# 'family': 'serif',
		"color" : "white",
		"weight": "bold",
		"size"  : 28,
	}
	fig, axs       = plt.subplots(1, 1)
	dpi            = 100
	left, width    = 0, 1
	bottom, height = 0, 1
	right          = left   + width
	center_x       = left   + width / 2
	center_y       = bottom + height / 2
	top            = bottom + height
	p              = plt.Rectangle((left, bottom), width, height, linewidth=0, fill=False, facecolor='none', edgecolor=None)
	p.set_transform(axs.transAxes)
	p.set_clip_on(True)
	
	axs.add_patch(p)
	axs.imshow(image, cmap=cmap)
	axs.title.set_text("")
	axs.text(right - 0.01, bottom + 0.01, text,
	         horizontalalignment = "right",
	         verticalalignment   = "bottom",
	         transform           = axs.transAxes,
	         fontdict            = font)
	axs.set_xticks([])
	axs.set_yticks([])
	axs.set_axis_off()
	plt.show()
	fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
	
	
image = mon.read_image("data/112_image_dav2_vitb_c.jpg", to_tensor=True, normalize=True)
edge, gradient = mon.boundary_aware_prior(image, eps=0.05)
gradient       = mon.to_image_nparray(gradient, denormalize=False)
edge           = mon.to_image_nparray(edge, denormalize=False)
image 		   = mon.to_image_nparray(image, denormalize=False)
gradient_pil   = Image.fromarray(gradient)
gradient_pil   = gradient_pil.filter(ImageFilter.FIND_EDGES)
draw_figure(gradient_pil, "", "data/Venice_gradient_color.jpg", cmap="viridis")
mon.write_image(mon.Path("data/Venice_edge.jpg"),     edge)
mon.write_image(mon.Path("data/Venice_gradient.jpg"), gradient)
