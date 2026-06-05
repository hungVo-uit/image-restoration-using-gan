import warnings
warnings.filterwarnings("ignore")

import os
import os.path as osp
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# --- 1. CUSTOM DATASET DEFINITION ---
class InferenceDataset(Dataset):
    def __init__(self, img_folder):
        self.img_paths = glob.glob(osp.join(img_folder, "*.*"))
        self.img_paths = [p for p in self.img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = osp.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        return img_np, img_name

# --- 2. CUSTOM COLLATE FUNCTION FOR WINDOWS COMPATIBILITY ---
def windows_collate_fn(batch):
    return batch[0]


# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    
    model_path = "model/RealESRGAN_x4plus.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("-" * 60)
    if device.type == "cuda":
        print(f"NOTIFICATION: Model is running on [ GPU ]")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print(" WARNING: Valid GPU not found or proper drivers are not installed.")
        print(" Model is running on [ CPU ] -> Processing speed will be significantly slower!")
    print("-" * 60)

    state_dict = torch.load(model_path, map_location=device)['params_ema']

    output_dir = "../data/real-esrgan"
    os.makedirs(output_dir, exist_ok=True)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(state_dict, strict=True)
    half_precision = True if torch.cuda.is_available() else False

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    upsample = RealESRGANer(
        scale=4,
        model=model, 
        model_path=model_path,
        tile=600,          
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
        device=device
    )
    upsample.model.eval()

    # Initialize DataLoader
    test_img_folder = "../data/test"
    dataset = InferenceDataset(test_img_folder)

    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False, 
        collate_fn=windows_collate_fn  
    )

    print(f"\nStart processing {len(dataset)} images...")

    with torch.no_grad():
        for img_np, img_name in data_loader:
            output, _ = upsample.enhance(img_np, outscale=4)    
            output_img = Image.fromarray(np.uint8(output))
            save_path = osp.join(output_dir, img_name)
            output_img.save(save_path, compress_level=1)

    print("\nInference pipeline completed successfully!")