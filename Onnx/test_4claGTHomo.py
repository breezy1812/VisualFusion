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
    # device = torch.device("cuda")
    device = torch.device("cpu")
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
    matcher.load_state_dict(torch.load(fusion_weight_path, map_location=device), strict=False)
    matcher = matcher.eval().to(device, dtype=fpMode)


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
        img0_raw = cv2.imread(img0_pth)  # BGR格式，與C++ cv::imread一致
        img1_raw = cv2.imread(img1_pth)  # BGR格式，與C++ cv::imread一致

        # 2. 圖片裁切 (與C++ main.cpp完全一致)
        if is_picture_cut:
            print(f"裁切EO圖片: x={pcut_x}, y={pcut_y}, w={pcut_w}, h={pcut_h}")
            img0_raw = crop_image(img0_raw, pcut_x, pcut_y, pcut_w, pcut_h)
        # IR圖片不裁切（與C++版本一致）

        # 3. 第一次Resize到輸出尺寸 (與C++ main.cpp完全一致，使用INTER_LINEAR)
        eo_resized = cv2.resize(img0_raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        ir_resized = cv2.resize(img1_raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # 4. 轉灰階 (與C++ main.cpp: cv::cvtColor(eo_resized, gray_eo, cv::COLOR_BGR2GRAY)一致)
        gray_eo = cv2.cvtColor(eo_resized, cv2.COLOR_BGR2GRAY)
        gray_ir = cv2.cvtColor(ir_resized, cv2.COLOR_BGR2GRAY)

        # 5. 第二次resize到預測尺寸 (與C++ core_image_align_onnx.cpp一致，使用默認INTER_LINEAR)
        eo_pred = cv2.resize(gray_eo, (pred_w, pred_h), interpolation=cv2.INTER_LINEAR)
        ir_pred = cv2.resize(gray_ir, (pred_w, pred_h), interpolation=cv2.INTER_LINEAR)

        # 6. 再次檢查灰階（與C++版本一致）
        if len(eo_pred.shape) == 3:
            eo_pred = cv2.cvtColor(eo_pred, cv2.COLOR_BGR2GRAY)
        if len(ir_pred.shape) == 3:
            ir_pred = cv2.cvtColor(ir_pred, cv2.COLOR_BGR2GRAY)

        # 7. 準備模型輸入張量 (與C++ normalize操作一致: / 255.0)
        img0_tensor = torch.from_numpy(eo_pred)[None][None].to(device, dtype=fpMode) / 255.
        img1_tensor = torch.from_numpy(ir_pred)[None][None].to(device, dtype=fpMode) / 255.

        # 8. 特徵點匹配推理 - 使用灰階圖片 (與C++版本一致)
        start_time = time.time()
        mkpts0, mkpts1, leng1, leng2 = matcher(img0_tensor, img1_tensor)
        infer_time = time.time() - start_time  # infer_time 單位是秒

        # 9. 只取實際有效的特徵點 (與C++版本一致：使用leng1作為有效特徵點數)
        mkpts0 = mkpts0[:leng1].cpu().numpy()
        mkpts1 = mkpts1[:leng1].cpu().numpy()  # 注意：C++版本使用leng1，不是leng2

        print(f"模型推理提取到 {leng1} 個特徵點對")

        csv_file = os.path.join(result_path, "infer_times.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "   infer_time_seconds","   leng"])

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([eo_img_name, f"  {infer_time:.6f}   " , leng1])
        print(f"Inference time for {eo_img_name}: {infer_time:.6f} seconds")

        # 10. **重要：特徵點縮放處理 (與C++版本完全一致)**
        # C++ core_image_align_onnx.cpp中進行特徵點縮放：
        # pt.x = pt.x * param_.out_width_scale + param_.bias_x;
        # pt.y = pt.y * param_.out_height_scale + param_.bias_y;
        # 其中：out_width_scale = out_w / pred_w, out_height_scale = out_h / pred_h
        out_width_scale = out_w / pred_w if pred_w != 0 else 1.0
        out_height_scale = out_h / pred_h if pred_h != 0 else 1.0
        bias_x = 0  # C++ main.cpp設定為0
        bias_y = 0  # C++ main.cpp設定為0

        # 只有當縮放係數不等於1或bias不為0時才進行處理
        if abs(out_width_scale - 1.0) > 1e-6 or abs(out_height_scale - 1.0) > 1e-6 or bias_x > 0 or bias_y > 0:
            # 縮放和偏移特徵點座標（與C++版本完全一致）
            mkpts0[:, 0] = mkpts0[:, 0] * out_width_scale + bias_x  # x座標
            mkpts0[:, 1] = mkpts0[:, 1] * out_height_scale + bias_y  # y座標
            mkpts1[:, 0] = mkpts1[:, 0] * out_width_scale + bias_x  # x座標
            mkpts1[:, 1] = mkpts1[:, 1] * out_height_scale + bias_y  # y座標
            print(f"特徵點縮放: pred({pred_w}x{pred_h}) -> out({out_w}x{out_h}), scale=({out_width_scale:.2f}, {out_height_scale:.2f}), bias=({bias_x}, {bias_y})")
        else:
            print(f"特徵點處理: pred({pred_w}x{pred_h}) == out({out_w}x{out_h}), scale=({out_width_scale:.2f}, {out_height_scale:.2f}), 無需縮放或偏移")
        print(f"縮放後特徵點數量: EO={len(mkpts0)}, IR={len(mkpts1)}")

        # 準備視覺化用的組合圖片
        temp_pair = np.hstack((gray_eo, gray_ir))
        temp_pair = cv2.cvtColor(temp_pair, cv2.COLOR_GRAY2BGR)

        # ========== RANSAC 濾除 outlier，提升精度 (與C++版本完全一致) ==========
        # 這是唯一的RANSAC處理，與C++ main.cpp中的處理完全一致
        refined_H = None
        if len(mkpts0) >= 4 and len(mkpts1) >= 4:
            # RANSAC計算homography (參數與C++版本完全一致)
            H, mask = cv2.findHomography(
                mkpts0.astype(np.float32),
                mkpts1.astype(np.float32),
                cv2.RANSAC, 8.0, mask=None, maxIters=800, confidence=0.98
            )

            if H is not None and mask is not None:
                inliers = cv2.countNonZero(mask)
                det_H = cv2.determinant(H)  # 計算整個3x3矩陣的行列式，與C++版本一致

                if inliers >= 4 and det_H > 1e-6 and det_H < 1e6:
                    refined_H = H.copy()
                    print(f"RANSAC成功: {len(mkpts0)} -> {inliers} inliers, det={det_H:.2e}")

                    # 過濾 inlier 特徵點 (與C++版本一致)
                    mask_bool = mask.ravel().astype(bool)
                    filtered_eo_pts = mkpts0[mask_bool]
                    filtered_ir_pts = mkpts1[mask_bool]

                    # 保存原始特徵點用於視覺化
                    mkpts0_original = mkpts0.copy()
                    mkpts1_original = mkpts1.copy()
                    original_mask = mask_bool.copy()

                    # 更新特徵點為inliers (與C++版本一致)
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
            mkpts0_original = mkpts0.copy()
            mkpts1_original = mkpts1.copy()
            original_mask = np.ones(len(mkpts0), dtype=bool) if len(mkpts0) > 0 else np.array([], dtype=bool)

        # 使用 refined homography (與C++版本一致)
        H = refined_H if refined_H is not None else np.eye(3, dtype=np.float64)
        aligned_eo_gray = eo_pred
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

            # 計算誤差 - 使用輸出尺寸
            if refined_H is not None:
                error = calc_homography_euclidean_error(refined_H, gt_H, out_w, out_h)
            else:
                error = -1.0

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
            aligned_eo_rgb = cv2.warpPerspective(eo_resized, H, (ir_resized.shape[1], ir_resized.shape[0]))

            # 創建mask來標示有效的重疊區域
            mask_eo = np.ones((eo_resized.shape[0], eo_resized.shape[1]), dtype=np.uint8) * 255
            aligned_mask = cv2.warpPerspective(mask_eo, H, (ir_resized.shape[1], ir_resized.shape[0]))

            # 將mask轉換為3通道
            aligned_mask_3ch = cv2.cvtColor(aligned_mask, cv2.COLOR_GRAY2RGB)

            # 只保留重疊區域：使用mask來遮蔽非重疊區域
            aligned_eo_masked = np.where(aligned_mask_3ch > 0, aligned_eo_rgb, 0)
            aligned_eo_masked=cv2.cvtColor(aligned_eo_masked, cv2.COLOR_BGR2RGB)  # BGR轉RGB以便顯示
        else:

            aligned_eo_edge=edge

            aligned_eo_masked = eo_resized
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

        ax1.imshow(img1_raw, cmap='gray')
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
