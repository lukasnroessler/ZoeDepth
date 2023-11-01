import os
import argparse
import torch
import numpy as np
from PIL import Image
import tqdm
from zoedepth.utils.misc import colorize




parser = argparse.ArgumentParser(description='OOD Evaluation')



parser.add_argument('--dataset_path', type=str, default='/home/lukasnroessler/Anomaly_Datasets/AnoVox',
                    help=""""path to depth images""")

args = parser.parse_args()


def collect_images():
    root = args.dataset_path
    img_data = []

    for scenario in os.listdir(root):
        if scenario == 'Scenario_Configuration_Files':
            continue
        img_dir = os.path.join(root, scenario, 'RGB_IMG')
        for image in os.listdir(img_dir):
            img_data.append(os.path.join(img_dir, image))
    
    def sorter(file_paths):
        identifier = (os.path.basename(file_paths).split('.')[0]).split('_')[-1]
        return int(identifier)
    
    img_data.sort(key=sorter)
    return img_data


def main():
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_nk.to(DEVICE)

    img_data = collect_images()
    output_dir = '/home/lukasnroessler/Projects/ZoeDepth/DepthPreds'
    os.mkdir(output_dir)

    # for i, img in tqdm(enumerate(img_data), desc=f"predicting depth images"):
    for i, img in enumerate(img_data):
        image = Image.open(img).convert("RGB")
        depth_prediction = zoe.infer_pil(image)  # as numpy
        output_path = os.path.join(output_dir, "depth_prediction{}.npy".format(str(i)))
        np.save(output_path, depth_prediction)
        # colored = colorize(depth_prediction)
        # Image.fromarray(colored).save(output_path)








if __name__ == "__main__":
    main()