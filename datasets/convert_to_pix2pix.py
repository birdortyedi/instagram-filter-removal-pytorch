import os
import shutil

raw_folder = "../../../Downloads/IFFI-dataset-raw"
raw_org_folder = "../../../Downloads/IFFI-dataset-raw/Original"
dst_folder = "../../../Downloads/IFFI-dataset-pix2pix"

for org_imgs_name in os.listdir(raw_org_folder):
    for filter_idx, filter_name in enumerate(sorted(os.listdir(raw_folder))):
        for f_imgs_name in os.listdir(os.path.join(raw_folder, filter_name)):
            if f_imgs_name == org_imgs_name:
                new_org_imgs_name = org_imgs_name.split(".")[0] + "_" + str(filter_idx) + "." + org_imgs_name.split(".")[-1]
                new_f_imgs_name = f_imgs_name.split(".")[0] + "_" + str(filter_idx) + "." + f_imgs_name.split(".")[-1]
                # print(filter_name, filter_idx, new_org_imgs_name, new_f_imgs_name)
                shutil.copy(os.path.join(raw_org_folder, org_imgs_name), os.path.join(dst_folder, "B", "train", new_org_imgs_name))
                shutil.copy(os.path.join(raw_folder, filter_name, f_imgs_name), os.path.join(dst_folder, "A", "train", new_f_imgs_name))
