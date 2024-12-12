import os
import shutil

if __name__ == "__main__":
    txt_file = "../util_txt/rdg_clean.txt"

    with open(txt_file) as file:
        lines = file.readlines()
        for line in lines:
            combined = os.path.join("all", "RdGG_" + line.strip() + ".png")
            if os.path.exists(combined):
                print(line)
                shutil.copyfile(combined, os.path.join("clean", "RdGG_" + line.strip() + ".png"))
