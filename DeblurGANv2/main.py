import warnings
warnings.filterwarnings("ignore")

import os
import os.path as osp
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


from models.networks import get_generator
from config.config import get_config


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

def windows_collate_fn(batch):
    return batch[0]


# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    

    deblur_config_path = "config/config_deblur.yaml"
    deblur_weight_path = "model/fpn_inception.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("-" * 60)
    if device.type == "cuda":
        print(f"NOTIFICATION: DeblurGAN-v2 is running on [ GPU ]")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print(" WARNING: Valid GPU not found or proper drivers are not installed.")
        print(" Model is running on [ CPU ] -> Processing speed will be significantly slower!")
    print("-" * 60)

    os.makedirs("results", exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("Loading DeblurGAN-v2...")
    deblur_cfg = get_config(deblur_config_path)
    model = get_generator(deblur_cfg)
    
    
    checkpoint = torch.load(deblur_weight_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()

    test_img_folder = "test_img"
    dataset = InferenceDataset(test_img_folder)

    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True if torch.cuda.is_available() else False, 
        collate_fn=windows_collate_fn  
    )

    print(f"\nStart processing {len(dataset)} images (Deblurring only)...")

    with torch.no_grad():
        for img_np, img_name in data_loader:
            
        
            img_tensor = (img_np / 127.5) - 1.0
            img_tensor = torch.from_numpy(img_tensor).float().permute(2, 0, 1).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            
            output_tensor = model(img_tensor)
            
           
            output_np = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_np = (output_np + 1.0) * 127.5
            output_np = np.clip(output_np, 0, 255).astype(np.uint8)
            
            
            output_img = Image.fromarray(output_np)
            output_img.save(osp.join("results", img_name), compress_level=1)

    print("\nInference pipeline completed successfully!")