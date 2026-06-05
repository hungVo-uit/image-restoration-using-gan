import os
import os.path as osp
import glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torchvision.transforms as transforms
import lpips
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

loss_fn = None
transform_lpips = None
device = torch.device('cpu')

def init_worker():
    """Hàm này sẽ chạy DUY NHẤT một lần khi mỗi tiến trình (Process) được tạo ra"""
    global loss_fn, transform_lpips
    torch.set_num_threads(1) 
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()
    transform_lpips = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def evaluate_single_pair(gt_path, pred_path):
    """Hàm xử lý một cặp ảnh duy nhất"""
    global loss_fn, transform_lpips
    img_name = osp.basename(gt_path)
    
    if not osp.exists(pred_path):
        return None, f"Cảnh báo: Không tìm thấy ảnh đối ứng cho {img_name}"
        
    try:
        with Image.open(gt_path).convert('RGB') as img_gt, Image.open(pred_path).convert('RGB') as img_pred:
            if img_gt.size != img_pred.size:
                img_gt = img_gt.resize(img_pred.size, Image.BICUBIC)
            
        
            img_gt_np = np.array(img_gt)
            img_pred_np = np.array(img_pred)
            data_range = 255 if img_gt_np.max() > 1.0 else 1.0
            
            if len(img_gt_np.shape) == 3 and img_gt_np.shape[-1] in [3, 4]:
                ssim_val = ssim(img_gt_np, img_pred_np, data_range=data_range, channel_axis=-1)
            else:
                ssim_val = ssim(img_gt_np, img_pred_np, data_range=data_range)
                
            psnr_val = psnr(img_gt_np, img_pred_np, data_range=data_range)
            
            img_gt_lpips = img_gt.resize((512, 512), Image.BILINEAR)
            img_pred_lpips = img_pred.resize((512, 512), Image.BILINEAR)
            
            with torch.no_grad():
                tensor_gt = transform_lpips(img_gt_lpips).unsqueeze(0).to(device)
                tensor_pred = transform_lpips(img_pred_lpips).unsqueeze(0).to(device)
                lpips_val = loss_fn(tensor_gt, tensor_pred).item()
                
        return (psnr_val, ssim_val, lpips_val, img_name), None
        
    except Exception as e:
        return None, f"Lỗi khi xử lý ảnh {img_name}: {str(e)}"

def load_and_evaluate(gt_folder, pred_folder):
    gt_paths = glob.glob(osp.join(gt_folder, "*.*"))
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    gt_paths = [p for p in gt_paths if p.lower().endswith(valid_extensions)]
    
    print(f"Tìm thấy {len(gt_paths)} ảnh. Bắt đầu tính toán song song...")
    
    num_workers = 4
    print(f"Đang sử dụng {num_workers} lõi CPU để xử lý.")

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = {}
        for gt_path in gt_paths:
            img_name = osp.basename(gt_path)
            pred_path = osp.join(pred_folder, img_name)
            futures[executor.submit(evaluate_single_pair, gt_path, pred_path)] = img_name
            
        for idx, future in enumerate(as_completed(futures)):
            result, error = future.result()
            
            if error:
                print(error)
                continue
                
            if result:
                psnr_val, ssim_val, lpips_val, img_name = result
                total_psnr += psnr_val
                total_ssim += ssim_val
                total_lpips += lpips_val
                count += 1
                
                print(f"[{count}/{len(gt_paths)}] {img_name} | PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f}")
            
            
            if count % 10 == 0:
                gc.collect()

    if count == 0:
        print("Không có cặp ảnh nào khớp để đánh giá!")
        return

    mean_psnr = total_psnr / count
    mean_ssim = total_ssim / count
    mean_lpips = total_lpips / count

    print("\n" + "="*45)
    print("RESULT:")
    print(f"Mean PSNR : {mean_psnr:.4f} dB")
    print(f"Mean SSIM : {mean_ssim:.4f}")
    print(f"Mean LPIPS: {mean_lpips:.4f}")
    print("="*45)

if __name__ == '__main__':
    FOLDER_GROUND_TRUTH = "ground"
    FOLDER_PREDICT = "deblur"
    
    load_and_evaluate(FOLDER_GROUND_TRUTH, FOLDER_PREDICT)