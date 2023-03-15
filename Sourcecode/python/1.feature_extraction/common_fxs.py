import os
import random
import shutil

def select_random_files(path, percent):
    all_files = os.listdir(path)
    num_files = len(all_files)
    num_selected_files = int(num_files * percent / 100)
    selected_files = random.sample(all_files, num_selected_files)
    return selected_files

def move_files(src_path, dst_path, files):
    for file in files:
        src = os.path.join(src_path, file)
        dst = os.path.join(dst_path, file)
        shutil.move(src, dst)


classes = os.listdir('data/dataset_flowers/train/')
for item in classes:

	src_path = "data/dataset_flowers/train/"+item # Replace with the path to the directory containing the files
	dst_path = "data/dataset_flowers/validation/"+item # Replace with the path to the directory where the selected files should be moved
	percent = 20 # Replace with the desired percentage of files to be selected

	selected_files = select_random_files(src_path, percent)
	move_files(src_path, dst_path, selected_files)
	print("Finished moving : "+ item+" with the file amount = "+str(len(selected_files)))
 
print("End of script")
