import torch
import numpy as np

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

repo = "isl-org/ZoeDepth"
# Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Zoe_K trained on KITTI
# model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True, config_mode='eval')

# Zoe_NK
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(DEVICE)


# Local file
from PIL import Image
image = Image.open("/home/lukasnroessler/Anomaly_Datasets/AnoVox/Scenario_c8d20e26-7eaf-425b-8f86-c26bdd4ba365/RGB_IMG/RGB_IMG_271.png").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor



# Tensor 
from zoedepth.utils.misc import pil_to_batched_tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)


np.save("outputnumpy", depth_numpy)
# From URL
# from zoedepth.utils.misc import get_image_from_url

# Example URL
# URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"


# image = get_image_from_url(URL)  # fetch
# depth = zoe.infer_pil(image)
output_path = "/home/lukasnroessler/Projects/ZoeDepth/output.png"
# Save raw
# from zoedepth.utils.misc import save_raw_16bit
# fpath = "/home/lukasnroessler/Projects/ZoeDepth/output.png"
# save_raw_16bit(depth, fpath)
depth_pil.save(output_path)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth_numpy)

# save colored output
fpath_colored = "/home/lukasnroessler/Projects/ZoeDepth/output_colored.png"
Image.fromarray(colored).save(fpath_colored)
