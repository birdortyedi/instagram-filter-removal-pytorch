import os
import shutil

subset = "train"
raw_folder = "../../../Downloads/IFFI-dataset-raw/{}".format(subset)
src_folder = "../../../Downloads/IFFI-dataset/{}/".format(subset)
dst_folder = "../../../Downloads/IFFI-dataset-cycleGAN/{}{}"

for i, filter_name in enumerate(sorted(os.listdir(raw_folder))):
    for j, img_name in enumerate(sorted(os.listdir(os.path.join(raw_folder, filter_name)))):
        new_image_name = img_name.split(".")[0] + "_" + str(i) + "." + img_name.split(".")[-1]
        corr_org = os.path.join(src_folder, str(j), str(j)+"_Original.jpg")
        assert os.path.isfile(corr_org)

        src_A = os.path.join(raw_folder, filter_name, img_name)
        src_B = os.path.join(src_folder, img_name.split(".")[0], img_name.split(".")[0]+"_Original.jpg")
        dst_A = os.path.join(dst_folder.format(subset, "A"), new_image_name)
        dst_B = os.path.join(dst_folder.format(subset, "B"), new_image_name)

        print(src_A, dst_A, src_B, dst_B)

        shutil.copy(src_A, dst_A)
        shutil.copy(src_B, dst_B)