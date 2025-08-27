from matplotlib import gridspec
import torch
import cv2
from model.SemLA import SemLA
import matplotlib.pyplot as plt
import numpy as np
from einops.einops import rearrange
from model.utils import YCbCr2RGB, RGB2YCrCb, make_matching_figure
import os
from tqdm import tqdm
import time
import csv
import json

def calc_homography_euclidean_error(H1, H2, w, h):
    """
    計算兩個 homography 矩陣的歐幾里得誤差
    
    Args:
        H1: 第一個 homography 矩陣 (3x3)
        H2: 第二個 homography 矩陣 (3x3)  
        w: 圖像寬度
        h: 圖像高度
    
    Returns:
        float: 平均歐幾里得誤差，如果輸入無效則返回 -1.0
    """
    if H1 is None or H2 is None:
        return -1.0
    
    # 定義四個角點
    corners = np.array([
        [0, 0],
        [w, 0], 
        [0, h],
        [w, h]
    ], dtype=np.float32)
    
    try:
        # 使用兩個 homography 矩陣變換角點
        pts1 = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H1).reshape(-1, 2)
        pts2 = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H2).reshape(-1, 2)
        
        # 計算歐幾里得距離
        err = 0.0
        for i in range(4):
            dx = pts1[i][0] - pts2[i][0]
            dy = pts1[i][1] - pts2[i][1]
            err += np.sqrt(dx * dx + dy * dy)
        
        return err / 4.0
    except:
        return -1.0

if __name__ == '__main__':
    # config
    device = torch.device("cuda")
    dir = '/yh/Circ/circ_video/Version3'
    gt_dir = '/yh/Circ/Homo/SemLA/Version3'  # Ground truth JSON 文件目錄
    result_path = "/yh/Circ/Homo/SemLA/Ver3_Allimg/"
    os.makedirs(result_path, exist_ok=True)

    reg_weight_path = "/yh/Circ/Homo/SemLAold/reg.ckpt"
    fusion_weight_path = "/yh/Circ/Homo/SemLAold/fusion75epoch.ckpt"
    match_mode = 'scene'  # 'semantic' or 'scene'

    # matcher = SemLA(device).cuda()
    matcher = SemLA(device).to(device)
    matcher.load_state_dict(torch.load(reg_weight_path, map_location=device), strict=False)
    matcher.load_state_dict(torch.load(fusion_weight_path, map_location=device), strict=False)
    matcher.eval()

    eo_images = sorted([f for f in os.listdir(dir) if f.endswith("EO.jpg")])
    ir_images = sorted([f for f in os.listdir(dir) if f.endswith("IR.jpg")])
    ir_map = {os.path.splitext(f)[0].replace('IR', ''): f for f in ir_images}

    # 用於記錄誤差的列表
    error_results = []

    for eo_img_name in tqdm(eo_images, desc="Processing images"):
        base_name = os.path.splitext(eo_img_name)[0].replace('EO', '')
        ir_img_name = ir_map.get(base_name)
        if ir_img_name is None:
            continue
            
        # 查找對應的 GT JSON 文件
        # 從 "2024-07-10_15-38-54_" 提取時間戳，對應到 "IR_2024-07-10_15-38-54.json"
        timestamp = base_name.rstrip('_')  # 移除末尾的下劃線
        gt_json_name = f"IR_{timestamp}.json"
        gt_json_path = os.path.join(gt_dir, gt_json_name)
        
        if not os.path.exists(gt_json_path):
            print(f"Warning: GT file {gt_json_name} not found, skipping...")
            continue
            
        img0_pth = os.path.join(dir, eo_img_name)
        img1_pth = os.path.join(dir, ir_img_name)

        # 讀取EO和IR圖片
        h=240
        w=320
        img0_raw = cv2.imread(img0_pth)
        img0_raw = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2RGB)
        img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
        img0_raw = cv2.resize(img0_raw, (w, h))
        img1_raw = cv2.resize(img1_raw, (w, h))

        # 將vi圖片轉成灰階再做Canny
        # TODO: Start
        img0_gray = cv2.cvtColor(img0_raw, cv2.COLOR_RGB2GRAY)

        # 1. Gaussian blur
        blur = cv2.GaussianBlur(img0_gray, (5, 5), 0)

        # 2. Sobel gradients
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

        # 3. Edge_x and edge_y with border roll logic
        border = 1  # You can adjust this as needed
        edge_x = np.zeros_like(sobel_x)
        edge_y = np.zeros_like(sobel_y)

        # Roll along width (axis=1) for x, height (axis=0) for y
        edge_x = np.where(sobel_x < 1.0, np.roll(sobel_x, border, axis=1), edge_x)
        edge_x = np.where(sobel_x > 1.0, np.roll(-sobel_x, -border, axis=1), edge_x)
        edge_y = np.where(sobel_y < 1.0, np.roll(sobel_y, border, axis=0), edge_y)
        edge_y = np.where(sobel_y > 1.0, np.roll(-sobel_y, -border, axis=0), edge_y)

        # 4. Edge magnitude
        edge = np.sqrt(edge_x ** 2 + edge_y ** 2)

        # 5. Sobel magnitude
        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # 6. Final edge
        edge = sobel - edge
        edge = np.clip(edge, -255, 255)  # Clip to valid range
        edge = np.expand_dims(edge, axis=-1)  # Add channel dimension
        edge = np.repeat(edge, 3, axis=-1)  # Convert to 3-channel image
        # TODO: End

        img0 = torch.from_numpy(img0_raw)[None].to(device) / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].to(device) / 255.
        img0 = rearrange(img0, 'n h w c ->  n c h w')
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img0)

        # 特徵點匹配
        
        start_time = time.time()
        mkpts0, mkpts1, leng1, leng2 = matcher(vi_Y, img1, matchmode=match_mode)
        infer_time = time.time() - start_time  # infer_time 單位是秒
        
        csv_file = os.path.join(result_path, "infer_times.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "   infer_time_seconds","   leng"])

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([eo_img_name, f"  {infer_time:.6f}   " , leng1])
        print(f"Inference time for {eo_img_name}: {infer_time:.6f} seconds")
        # 只取實際有效的特徵點
        mkpts0 = mkpts0[:leng1].cpu().numpy()
        mkpts1 = mkpts1[:leng2].cpu().numpy()
        
        # 畫上特徵點
        vi_Y_np = vi_Y.cpu().numpy()[0, 0]
        vi_Y_np = (vi_Y_np * 255).astype(np.uint8)
        temp_pair = np.hstack((img0_gray, img1_raw))
        temp_pair = cv2.cvtColor(temp_pair, cv2.COLOR_GRAY2BGR)

        
        # 直接對齊，不用tps
        img1_tensor = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        # 計算Homography
        if len(mkpts0) >= 4 and len(mkpts1) >= 4:
            H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 10.0, maxIters=800, confidence=0.99)
            mask = mask.ravel().astype(bool)
            
            # 讀取 Ground Truth Homography
            try:
                with open(gt_json_path, 'r') as f:
                    gt_data = json.load(f)
                gt_H = np.array(gt_data['H'], dtype=np.float32)
                
                # 計算誤差
                error = calc_homography_euclidean_error(H, gt_H, w, h)
                error_results.append({
                    'image_name': timestamp,
                    'error': error
                })
                print(f"Image: {timestamp}, Homography Error: {error:.4f}")
                
            except Exception as e:
                print(f"Error reading GT file {gt_json_name}: {e}")
                error_results.append({
                    'image_name': timestamp,
                    'error': -1.0
                })
            
            for pt_ir, pt_eo in zip(mkpts1[mask], mkpts0[mask]):
                x_ir, y_ir = pt_ir
                x_eo, y_eo = pt_eo
                x_ir += img0_gray.shape[1]  # 將EO點的x座標偏移到右側

                cv2.circle(temp_pair, (x_ir, y_ir), 2, (0, 255, 0), -1)
                cv2.circle(temp_pair, (x_eo, y_eo), 2, (0, 0, 255), -1)
                cv2.line(temp_pair, (x_ir, y_ir), (x_eo, y_eo), (255, 0, 0), 1)
        else:
            H = None
            error_results.append({
                'image_name': timestamp,
                'error': -1.0
            })
            print(f"Image: {timestamp}, Insufficient keypoints for homography")

        # 對齊EO原圖到IR圖片空間
        if H is not None:
            # 1. 邊緣檢測的對齊（用於第三張圖）
            aligned_eo_edge = cv2.warpPerspective(edge, H, (img1_raw.shape[1], img1_raw.shape[0]))
            
            # 2. 整個EO圖片的對齊（用於第四張圖，只顯示重疊區域）
            aligned_eo_rgb = cv2.warpPerspective(img0_raw, H, (img1_raw.shape[1], img1_raw.shape[0]))
            
            # 創建mask來標示有效的重疊區域
            mask_eo = np.ones((img0_raw.shape[0], img0_raw.shape[1]), dtype=np.uint8) * 255
            aligned_mask = cv2.warpPerspective(mask_eo, H, (img1_raw.shape[1], img1_raw.shape[0]))
            
            # 將mask轉換為3通道
            aligned_mask_3ch = cv2.cvtColor(aligned_mask, cv2.COLOR_GRAY2RGB)
            
            # 只保留重疊區域：使用mask來遮蔽非重疊區域
            aligned_eo_masked = np.where(aligned_mask_3ch > 0, aligned_eo_rgb, 0)
        else:
            aligned_eo_edge = edge
            aligned_eo_masked = img0_raw

        # 將IR圖轉成3通道以便重疊顯示
        ir_rgb = cv2.cvtColor(img1_raw, cv2.COLOR_GRAY2RGB)

        # 第三張圖：邊緣檢測融合（保持原有效果）
        overlap_edge = ir_rgb + aligned_eo_edge
        overlap_edge = np.clip(overlap_edge, 0, 255).astype(np.uint8)

        # 新版：顯示原始EO、IR和對齊後的重疊結果
        fig = plt.figure(figsize=(18, 6), dpi=300)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        # ax4 = fig.add_subplot(gs[3])

        ax1.imshow(img1_raw, cmap='gray')
        ax1.set_title('Original IR')
        ax1.axis('off')
        
        ax2.imshow(aligned_eo_masked)
        ax2.set_title('Aligned EO (Overlap Only)')
        ax2.axis('off')
        
        ax3.imshow(overlap_edge)
        ax3.set_title('Aligned EO Edge + IR')
        ax3.axis('off')
        
        plt.tight_layout()

        # 儲存
        save_name = os.path.splitext(eo_img_name)[0].replace('EO', '') + '.png'
        plt.savefig(os.path.join(result_path, save_name))
        plt.close(fig)

    # 計算並保存統計結果
    valid_errors = [r['error'] for r in error_results if r['error'] >= 0]
    total_images = len(error_results)
    successful_homographies = len(valid_errors)
    
    print("\n=== Homography Error Statistics ===")
    print(f"Total images processed: {total_images}")
    print(f"Successfully computed homography: {successful_homographies}")
    print(f"Success rate: {successful_homographies/total_images*100:.2f}%")
    
    if valid_errors:
        mean_error = np.mean(valid_errors)
        std_error = np.std(valid_errors)
        min_error = np.min(valid_errors)
        max_error = np.max(valid_errors)
        
        print(f"Mean error: {mean_error:.4f} pixels")
        print(f"Std error: {std_error:.4f} pixels")
        print(f"Min error: {min_error:.4f} pixels")
        print(f"Max error: {max_error:.4f} pixels")
    else:
        print("No valid errors computed!")
    
    # 保存詳細結果到 CSV
    error_csv_path = os.path.join(result_path, "homography_errors.csv")
    with open(error_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['image_name', '  error'])  # 標題行也加空格
        for result in error_results:
            writer.writerow([result['image_name'], f"  {result['error']:.6f}"])
    
    print(f"\nDetailed results saved to: {error_csv_path}")
    print('done!')
