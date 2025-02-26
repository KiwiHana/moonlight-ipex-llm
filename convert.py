import os
import shutil
from safetensors.torch import load_file, save_file
 
src_dir = r"C:\Users\Le\Documents\Moonlight-16B-A3B-Instruct"
dst_dir = r"C:\Users\Le\Documents\Moonlight-16B-A3B-Instruct-converted"
 
os.makedirs(dst_dir, exist_ok=True)
 
for filename in os.listdir(src_dir):
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
 
    if filename.endswith(".safetensors"):
        state_dict = load_file(src_path)
        save_file(state_dict, dst_path, metadata={"format": "pt"})
    # else:
    #     shutil.copyfile(src_path, dst_path)
