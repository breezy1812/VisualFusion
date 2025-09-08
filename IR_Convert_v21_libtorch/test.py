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

def crop_image(source_pic, x, y, w, h):
    """
    圖片裁切函數，與C++ cropImage完全一致
    """
    # 邊界檢查，確保不超出原圖
    crop_x = max(0, x)
    crop_y = max(0, y)
    crop_w = w
    crop_h = h

    if w < 0:
        crop_w = source_pic.shape[1] - crop_x
    if h < 0:
        crop_h = source_pic.shape[0] - crop_y

    crop_w = min(crop_w, source_pic.shape[1] - crop_x)
    crop_h = min(crop_h, source_pic.shape[0] - crop_y)

    return source_pic[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()

import numpy as np

def calc_keypoints_mse_error(eo_homo_pred, homo_gt, eo_keypoints):
    """
    計算特徵點轉換後的 MSE 誤差（索引一對一比較，不做最近鄰匹配）

    Args:
        eo_homo_pred: (3,3) 預測單應性矩陣
        homo_gt: (3,3) GT 單應性矩陣
        eo_keypoints: (N,2) EO 特徵點

    Returns:
        mse_error: float，平均 MSE 誤差
    """
    if eo_homo_pred is None or homo_gt is None or len(eo_keypoints) == 0:
        return -1.0

    try:
        # 1. 齊次座標
        eo_kpts_homo = np.hstack([eo_keypoints, np.ones((len(eo_keypoints), 1))])  # (N,3)

        # 2. 使用預測 H 做轉換
        kpts_pred_homo = (eo_homo_pred @ eo_kpts_homo.T).T  # (N,3)
        kpts_pred = kpts_pred_homo[:, :2] / kpts_pred_homo[:, 2:3]  # (N,2)

        # 3. 使用 GT H 做轉換
        kpts_gt_homo = (homo_gt @ eo_kpts_homo.T).T  # (N,3)
        kpts_gt = kpts_gt_homo[:, :2] / kpts_gt_homo[:, 2:3]  # (N,2)

        # 4. 一對一 MSE（索引對齊）
        diff = kpts_pred - kpts_gt          # (N,2)
        squared_distances = np.sum(diff**2, axis=1)  # (N,)
        mse_error = np.mean(squared_distances)

        # --- 診斷輸出 ---
        mean_disp = np.mean(np.sqrt(squared_distances))
        max_disp = np.max(np.sqrt(squared_distances))
        print(f"特徵點數量: {len(eo_keypoints)}")
        print(f"平均位移: {mean_disp:.2f} pixels")
        print(f"最大位移: {max_disp:.2f} pixels")
        print(f"MSE: {mse_error:.4f}")

        return mse_error

    except Exception as e:
        print(f"Error in keypoints MSE calculation: {e}")
        return -1.0
 

if __name__ == '__main__':
    # config
    device = torch.device("cuda")
    # device = torch.device("cpu")
    dir = '/yh/Circ/circ_video/Version3'
    gt_dir = '/yh/Circ/Homo/SemLA/Version3'  # Ground truth JSON 文件目錄
    result_path = "/yh/Circ/Homo/SemLA/Ver3_Allimg/"
    os.makedirs(result_path, exist_ok=True)

    reg_weight_path = "/yh/Circ/Homo/SemLAold/reg.ckpt"
    fusion_weight_path = "/yh/Circ/Homo/SemLAold/fusion75epoch.ckpt"
    match_mode = 'scene'  # 'semantic' or 'scene'

    # fpMode = torch.float16
    fpMode = torch.float32
    matcher = SemLA(device=device, fp=fpMode).to(device)
    matcher.load_state_dict(torch.load(reg_weight_path, map_location=device), strict=False)
    # matcher.load_state_dict(torch.load(fusion_weight_path, map_location=device), strict=False)
    matcher = matcher.eval().to(device, dtype=fpMode)

    # Load config for warm up
    config_path = "/yh/Circ/Homo/SemLA/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    pred_w, pred_h = config["pred_width"], config["pred_height"]
    
    # Warm Up (參考C++版本的warm_up函數)
    print("Warm up...")
    ts_st = time.time()
    for i in range(10):
        # 創建與pred尺寸相同的全白張量 (類似C++ cv::Mat::ones * 255)
        eo_warmup = torch.ones(1, 1, pred_h, pred_w, dtype=fpMode, device=device)
        ir_warmup = torch.ones(1, 1, pred_h, pred_w, dtype=fpMode, device=device)
        with torch.no_grad():
            matcher(eo_warmup, ir_warmup)
    ts_ed = time.time()
    print(f"Warm up done in {ts_ed - ts_st:.2f} s")

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

        print('device:',device)
        print('fpMode:',fpMode)
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

        # 讀取EO和IR圖片 - 完全按照C++版本的處理方式
        # Load config to get actual dimensions - 與C++版本完全一致
        config_path = "/yh/Circ/Homo/SemLA/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        out_w, out_h = config["output_width"], config["output_height"]  # 與C++ main.cpp一致
        pred_w, pred_h = config["pred_width"], config["pred_height"]  # 與C++ config一致

        # 檢查是否需要裁切（與C++版本一致）
        is_picture_cut = config.get("PictureCut", False)
        pcut_x = config.get("Pcut_x", 0)
        pcut_y = config.get("Pcut_y", 0)
        pcut_w = config.get("Pcut_w", -1)
        pcut_h = config.get("Pcut_h", -1)

        # 1. 讀取原始圖片 (與C++完全相同的方式)
        eo_bgr = cv2.imread(img0_pth)  # BGR格式，與C++ cv::imread一致
        ir_bgr = cv2.imread(img1_pth)  # BGR格式，與C++ cv::imread一致

        # 2. 圖片裁切 (與C++ main.cpp完全一致)
        if is_picture_cut:
            print(f"裁切EO圖片: x={pcut_x}, y={pcut_y}, w={pcut_w}, h={pcut_h}")
            eo_bgr = crop_image(eo_bgr, pcut_x, pcut_y, pcut_w, pcut_h)
        # IR圖片不裁切（與C++版本一致）

        # 3. 第一次Resize到輸出尺寸 (與C++ main.cpp完全一致，使用INTER_LINEAR)
        eo_resized_for_align = cv2.resize(eo_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        ir_resized_for_align = cv2.resize(ir_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # 4. 轉灰階 (與C++ main.cpp: cv::cvtColor(eo_resized_for_align, gray_eo, cv::COLOR_BGR2GRAY)一致)
        gray_eo = cv2.cvtColor(eo_resized_for_align, cv2.COLOR_BGR2GRAY)
        gray_ir = cv2.cvtColor(ir_resized_for_align, cv2.COLOR_BGR2GRAY)

        # 6. 再次檢查灰階（與C++版本一致）
        if len(gray_eo.shape) == 3:
            gray_eo = cv2.cvtColor(gray_eo, cv2.COLOR_BGR2GRAY)
        if len(gray_ir.shape) == 3:
            gray_ir = cv2.cvtColor(gray_ir, cv2.COLOR_BGR2GRAY)

        # 7. 準備模型輸入張量 (與LibTorch C++版本完全一致)
        # C++版本的tensor建立和normalize:
        # torch::from_blob(eo.data, {1, 1, pred_height, pred_width}, torch::kUInt8).to(device).to(torch::kFloat32) / 255.0f
        img0_tensor = torch.from_numpy(gray_eo).unsqueeze(0).unsqueeze(0).to(device, dtype=fpMode) / 255.0
        img1_tensor = torch.from_numpy(gray_ir).unsqueeze(0).unsqueeze(0).to(device, dtype=fpMode) / 255.0

        # 確保張量的形狀正確 [1, 1, H, W]
        assert img0_tensor.shape == (1, 1, out_h, out_w), f"EO tensor shape mismatch: {img0_tensor.shape} vs expected (1, 1, {out_h}, {out_w})"
        assert img1_tensor.shape == (1, 1, out_h, out_w), f"IR tensor shape mismatch: {img1_tensor.shape} vs expected (1, 1, {out_h}, {out_w})"

        # 8. 特徵點匹配推理 - 使用灰階圖片 (與C++版本一致)
        start_time = time.time()
        mkpts0, mkpts1, leng1, leng2 = matcher(img0_tensor, img1_tensor)
        infer_time = time.time() - start_time  # infer_time 單位是秒

        # 9. 只取實際有效的特徵點 (與LibTorch C++版本完全一致：使用leng作為有效特徵點數)
        # 在C++版本中，模型返回的是 leng (int pred_[2].toInt())，對應最終的有效特徵點數量
        # 這裡使用 leng1 作為實際的特徵點數量，與C++版本的 int leng=pred_[2].toInt() 一致
        actual_length = int(leng1)  # 確保是整數，與C++版本一致
        mkpts0 = mkpts0[:actual_length].cpu().numpy()
        mkpts1 = mkpts1[:actual_length].cpu().numpy()

        print(f"模型推理提取到 {actual_length} 個特徵點對")

        csv_file = os.path.join(result_path, "infer_times.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "   infer_time_seconds","   leng"])

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([eo_img_name, f"  {infer_time:.6f}   " , actual_length])
        print(f"Inference time for {eo_img_name}: {infer_time:.6f} seconds")

        # 10. **重要：特徵點縮放處理 (與LibTorch C++版本完全一致)**
        # C++ core_image_align_libtorch.cpp 中的條件檢查：
        # if (std::abs(param_.out_width_scale - 1.0) > 1e-6 || std::abs(param_.out_height_scale - 1.0) > 1e-6 || param_.bias_x > 0 || param_.bias_y > 0)
        # 縮放公式：i.x = i.x * param_.out_width_scale; i.y = i.y * param_.out_height_scale;
        out_width_scale = out_w / pred_w if pred_w != 0 else 1.0
        out_height_scale = out_h / pred_h if pred_h != 0 else 1.0
        bias_x = 0  # C++ main.cpp設定為0
        bias_y = 0  # C++ main.cpp設定為0

        # 完全按照C++版本的條件檢查（使用1e-6閾值）
        if abs(out_width_scale - 1.0) > 1e-6 or abs(out_height_scale - 1.0) > 1e-6 or bias_x > 0 or bias_y > 0:
            # 縮放特徵點座標（與C++版本完全一致，不加bias因為C++中bias=0）
            mkpts0[:, 0] = mkpts0[:, 0] * out_width_scale  # x座標
            mkpts0[:, 1] = mkpts0[:, 1] * out_height_scale  # y座標
            mkpts1[:, 0] = mkpts1[:, 0] * out_width_scale  # x座標
            mkpts1[:, 1] = mkpts1[:, 1] * out_height_scale  # y座標
            print(f"特徵點縮放應用: pred({pred_w}x{pred_h}) -> out({out_width_scale * pred_w:.0f}x{out_height_scale * pred_h:.0f}), scale=({out_width_scale:.6f}, {out_height_scale:.6f}), bias=({bias_x}, {bias_y})")
        else:
            print(f"無需特徵點縮放: scale=({out_width_scale:.6f}, {out_height_scale:.6f}), bias=({bias_x}, {bias_y})")
        print(f"最終特徵點數量: {len(mkpts0)}")

        # 準備視覺化用的組合圖片
        temp_pair = np.hstack((gray_eo, gray_ir))
        temp_pair = cv2.cvtColor(temp_pair, cv2.COLOR_GRAY2BGR)

        # ========== RANSAC 濾除 outlier，提升精度 (與C++版本完全一致) ==========
        # 這是唯一的RANSAC處理，與C++ main.cpp中的處理完全一致
        refined_H = None
        if len(mkpts0) >= 4 and len(mkpts1) >= 4:
            # RANSAC計算homography - 修正：從EO到IR的映射
            # mkpts0是EO的特徵點，mkpts1是IR的特徵點
            H, mask = cv2.findHomography(
                mkpts0.astype(np.float32),  # EO特徵點作為source
                mkpts1.astype(np.float32),  # IR特徵點作為target
                cv2.RANSAC, 8.0, mask=None, maxIters=1000, confidence=0.99
            )

            if H is not None and mask is not None:
                inliers = cv2.countNonZero(mask)
                det_H = cv2.determinant(H)  # 計算整個3x3矩陣的行列式，與C++版本一致

                if inliers >= 4 and det_H > 1e-6 and det_H < 1e6:
                    refined_H = H.copy()
                    print(f"RANSAC成功: {len(mkpts0)} -> {inliers} inliers, det={det_H:.2e}")

                    # 保存原始特徵點用於視覺化
                    mkpts0_original = mkpts0.copy()
                    mkpts1_original = mkpts1.copy()
                    
                    # 過濾 inlier 特徵點（與C++版本完全一致）
                    mask_bool = mask.ravel().astype(bool)
                    original_mask = mask_bool.copy()
                    
                    filtered_eo_pts = mkpts0[mask_bool]
                    filtered_ir_pts = mkpts1[mask_bool]

                    # 更新特徵點為inliers（與C++版本一致）
                    mkpts0 = filtered_eo_pts
                    mkpts1 = filtered_ir_pts
                    
                else:
                    print(f"Homography品質檢查失敗: inliers={inliers}, det={det_H:.2e}")
                    refined_H = None
                    mkpts0_original = mkpts0.copy()
                    mkpts1_original = mkpts1.copy()
                    original_mask = np.ones(len(mkpts0), dtype=bool)  # 全部視為outlier
            else:
                print("RANSAC failed to find homography")
                refined_H = None
                mkpts0_original = mkpts0.copy()
                mkpts1_original = mkpts1.copy()
                original_mask = np.ones(len(mkpts0), dtype=bool)  # 全部視為outlier
        else:
            print(f"特徵點不足，無法執行RANSAC: EO={len(mkpts0)}, IR={len(mkpts1)}")
            refined_H = None
        # 使用 refined homography (與C++版本一致)
        # C++版本: M = refined_H.empty() ? cv::Mat::eye(3, 3, CV_64F) : refined_H.clone();
        if refined_H is not None:
            H = refined_H.copy()
        else:
            print("無有效的homography，使用單位矩陣")
            H = np.eye(3, dtype=np.float32)
            # 如果沒有有效的homography，所有特徵點都視為原始特徵點
            if 'mkpts0_original' not in locals():
                mkpts0_original = mkpts0.copy()
                mkpts1_original = mkpts1.copy()
                original_mask = np.ones(len(mkpts0), dtype=bool) if len(mkpts0) > 0 else np.array([], dtype=bool)

        # 使用 refined homography (與C++版本一致)
        if refined_H is not None:
            H = refined_H
        else:
            raise ValueError("Homography is None after RANSAC")
        aligned_eo_gray = gray_eo
        blur = cv2.GaussianBlur(aligned_eo_gray, (3, 3), 0)
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        border = 1
        edge_x= np.zeros_like(sobel_x)
        edge_y= np.zeros_like(sobel_y)

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

        # 讀取 Ground Truth Homography並計算誤差
        try:
            with open(gt_json_path, 'r') as f:
                gt_data = json.load(f)
            gt_H = np.array(gt_data['H'], dtype=np.float32)

            # 計算MSE誤差 - 使用RANSAC過濾後的inlier特徵點
            if H is not None and not np.allclose(H, np.eye(3)) and len(mkpts0) > 0:
                # 使用RANSAC過濾後的EO inlier特徵點進行誤差計算（更準確）
                error = calc_keypoints_mse_error(H, gt_H, mkpts0)
                feature_count_used = len(mkpts0)  # 使用的是inlier特徵點數量
            else:
                error = -1.0
                feature_count_used = 0

            error_results.append({
                'image_name': timestamp,
                'error': error,
                'feature_count': feature_count_used  # 記錄實際使用的特徵點數量
            })
            print(f"Image: {timestamp}, MSE Error: {error:.4f}, Inlier Features: {feature_count_used}")

        except Exception as e:
            print(f"Error reading GT file {gt_json_name}: {e}")
            error_results.append({
                'image_name': timestamp,
                'error': -1.0,
                'feature_count': 0  # 錯誤情況下特徵點數量設為0
            })

        # 繪製特徵點匹配 (使用原始特徵點和mask)
        if len(mkpts0_original) > 0 and len(mkpts1_original) > 0:
            for i, (pt_ir, pt_eo) in enumerate(zip(mkpts1_original, mkpts0_original)):
                x_ir, y_ir = int(pt_ir[0]), int(pt_ir[1])
                x_eo, y_eo = int(pt_eo[0]), int(pt_eo[1])
                x_ir_display = x_ir + gray_eo.shape[1]  # 將IR點的x座標偏移到右側

                # 根據mask決定顏色：綠色為inlier，紅色為outlier
                if i < len(original_mask) and original_mask[i]:
                    color = (0, 255, 0)  # 綠色 - inlier
                else:
                    color = (0, 0, 255)  # 紅色 - outlier

                cv2.circle(temp_pair, (x_ir_display, y_ir), 2, color, -1)
                cv2.circle(temp_pair, (x_eo, y_eo), 2, color, -1)
                cv2.line(temp_pair, (x_ir_display, y_ir), (x_eo, y_eo), (255, 0, 0), 1)

        # 對齊EO原圖到IR圖片空間
        if H is not None:
            # 1. EO灰階圖的對齊（用於第三張圖）- 改為使用灰階圖
            aligned_eo_edge = cv2.warpPerspective(edge, H, (gray_ir.shape[1], gray_ir.shape[0]))

            # 2. 整個EO圖片的對齊（用於第四張圖，只顯示重疊區域）
            aligned_eo_rgb = cv2.warpPerspective(eo_resized_for_align, H, (ir_resized_for_align.shape[1], ir_resized_for_align.shape[0]))

            # 創建mask來標示有效的重疊區域
            mask_eo = np.ones((eo_resized_for_align.shape[0], eo_resized_for_align.shape[1]), dtype=np.uint8) * 255
            aligned_mask = cv2.warpPerspective(mask_eo, H, (ir_resized_for_align.shape[1], ir_resized_for_align.shape[0]))

            # 將mask轉換為3通道
            aligned_mask_3ch = cv2.cvtColor(aligned_mask, cv2.COLOR_GRAY2RGB)


            # 只保留重疊區域：使用mask來遮蔽非重疊區域
            aligned_eo_masked = np.where(aligned_mask_3ch > 0, aligned_eo_rgb, 0)
            aligned_eo_masked=cv2.cvtColor(aligned_eo_masked, cv2.COLOR_BGR2RGB)  # BGR轉RGB以便顯示
        else:

            aligned_eo_edge=edge

            aligned_eo_masked = eo_resized_for_align
            aligned_eo_masked=cv2.cvtColor(aligned_eo_masked, cv2.COLOR_BGR2RGB)  # BGR轉RGB以便顯示
        # 將IR圖轉成3通道以便重疊顯示
        ir_rgb = cv2.cvtColor(gray_ir, cv2.COLOR_GRAY2RGB)

        # 第三張圖：灰階圖融合（改為使用灰階圖而非邊緣檢測）
        # aligned_eo_gray_3ch = cv2.cvtColor(aligned_eo_gray, cv2.COLOR_GRAY2RGB)
        # overlap_gray = np.clip((ir_rgb.astype(np.float32) + aligned_eo_gray_3ch.astype(np.float32)) / 2, 0, 255).astype(np.uint8)

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

        ax1.imshow(ir_bgr, cmap='gray')
        ax1.set_title('Original IR')
        ax1.axis('off')

        ax2.imshow(aligned_eo_masked)
        ax2.set_title('Aligned EO (Overlap Only)')
        ax2.axis('off')

        ax3.imshow(overlap_edge)
        ax3.set_title(' EO + IR')
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

    print("\n=== Keypoints MSE Error Statistics ===")
    print(f"Total images processed: {total_images}")
    print(f"Successfully computed homography: {successful_homographies}")
    print(f"Success rate: {successful_homographies/total_images*100:.2f}%")

    if valid_errors:
        mean_error = np.mean(valid_errors)
        std_error = np.std(valid_errors)
        min_error = np.min(valid_errors)
        max_error = np.max(valid_errors)

        print(f"Mean MSE error: {mean_error:.4f}")
        print(f"Std MSE error: {std_error:.4f}")
        print(f"Min MSE error: {min_error:.4f}")
        print(f"Max MSE error: {max_error:.4f}")
    else:
        print("No valid errors computed!")

    # 保存詳細結果到 CSV
    error_csv_path = os.path.join(result_path, "keypoints_mse_errors.csv")
    with open(error_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['image_name', '  mse_error'])  # 標題行也加空格
        for result in error_results:
            writer.writerow([result['image_name'], f"  {result['error']:.6f}"])

    print(f"\nDetailed results saved to: {error_csv_path}")
    print('done!')
