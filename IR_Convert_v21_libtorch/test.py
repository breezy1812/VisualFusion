import os
import cv2
import glob
import torch
import numpy as np
import json

from model.SemLA import SemLA

import warnings
warnings.filterwarnings('ignore')

# 載入裁剪配置
def load_crop_config():
    """載入EO影像裁剪配置"""
    try:
        # 讀取當前目錄下的配置文件
        config_path = "/yh/Circ/Homo/SemLA/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                _cfg = json.load(f)
            return _cfg["Vcut_x"], _cfg["Vcut_y"], _cfg["Vcut_w"], _cfg["Vcut_h"]
        else:
            print("配置文件不存在，使用默認裁剪參數")
            return 850, 200, 2040, 1700  # 您要求的配置
    except Exception as e:
        print(f"讀取配置文件失敗: {e}，使用默認裁剪參數")
        return 850, 200, 2040, 1700  # 您要求的配置

# 平滑Homography管理器
class SmoothHomographyManager:
    def __init__(self, max_translation_diff=20.0, max_rotation_diff=0.1, smooth_alpha=0.3):
        """
        max_translation_diff: 最大允許的平移差異 (像素)
        max_rotation_diff: 最大允許的旋轉差異 (弧度)
        smooth_alpha: 平滑係數 (0-1, 越小越平滑)
        """
        self.max_translation_diff = max_translation_diff
        self.max_rotation_diff = max_rotation_diff
        self.smooth_alpha = smooth_alpha
        self.previous_homo = None
        
    def calculate_homography_difference(self, homo1, homo2):
        """計算兩個homography矩陣的差異"""
        if homo1 is None or homo2 is None:
            return float('inf'), float('inf')
        
        # 計算平移差異
        translation_diff = np.sqrt((homo1[0, 2] - homo2[0, 2])**2 + 
                                 (homo1[1, 2] - homo2[1, 2])**2)
        
        # 計算旋轉差異 (通過矩陣的2x2左上角部分)
        rotation_matrix1 = homo1[:2, :2]
        rotation_matrix2 = homo2[:2, :2]
        
        # 計算旋轉角度差異
        angle1 = np.arctan2(rotation_matrix1[1, 0], rotation_matrix1[0, 0])
        angle2 = np.arctan2(rotation_matrix2[1, 0], rotation_matrix2[0, 0])
        rotation_diff = abs(angle1 - angle2)
        
        # 處理角度循環問題
        if rotation_diff > np.pi:
            rotation_diff = 2 * np.pi - rotation_diff
            
        return translation_diff, rotation_diff
    
    def should_update_homography(self, new_homo):
        """判斷是否應該更新homography"""
        if self.previous_homo is None:
            return True
        
        trans_diff, rot_diff = self.calculate_homography_difference(self.previous_homo, new_homo)
        
        # 如果差異太大，不更新
        if (trans_diff > self.max_translation_diff or 
            rot_diff > self.max_rotation_diff):
            return False
        
        return True
    
    def update_homography(self, new_homo):
        """更新homography with smooth blending"""
        if new_homo is None:
            return self.previous_homo
        
        # 如果這是第一次更新，直接使用新的
        if self.previous_homo is None:
            self.previous_homo = new_homo.copy()
            return new_homo
        
        # 如果應該更新，使用平滑混合
        if self.should_update_homography(new_homo):
            # 平滑混合: smooth_alpha * 新的 + (1-smooth_alpha) * 舊的
            smoothed_homo = (self.smooth_alpha * new_homo + 
                           (1 - self.smooth_alpha) * self.previous_homo)
            self.previous_homo = smoothed_homo.copy()
            return smoothed_homo
        else:
            # 差異太大，保持前一次的homography
            return self.previous_homo
    
    def get_current(self):
        """獲取當前homography"""
        return self.previous_homo
# 可以接受的副檔名
extends = ['mp4', 'mov', 'avi', 'mkv']

# 來源影片與目標資料夾路徑
source_path = "/yh/Circ/circ_video/Version1/"
target_path = "/yh/Circ/Homo/SemLA/video/"

# 從來源資料夾中找影片，並且是屬於 IR 的影片
files = []
for ext in extends:
    files += glob.glob(f"{source_path}/*_IR.{ext}")
    files += glob.glob(f"{source_path}/*_IR.{ext.upper()}")

# 載入EO裁剪配置
Vcut_x, Vcut_y, Vcut_w, Vcut_h = load_crop_config()
print(f"EO裁剪配置: x={Vcut_x}, y={Vcut_y}, w={Vcut_w}, h={Vcut_h}")

# 預計辨識的尺度
TARGET_W, TARGET_H = 320, 240

# 每幾幀做一次計算
compute_per_frame = 50

# 時間延遲
time_delay = {
    "20231117_140552_523_IR": 95,
    "20231117_140617_461_IR": 68,
    "20231117_140723_990_IR": 72,
    "20231117_140746_894_IR": 68,
    "20231117_140758_655_IR": 68,
    "20231117_140815_118_IR": 67.5,
    "20231117_140857_164_IR": 66,
    "20231117_140917_210_IR": 68,
    "20231117_140934_116_IR": 66
}

# 載入模型
reg_weight_path = "/yh/Circ/Homo/SemLAold/reg.ckpt"
fusion_weight_path = "/yh/Circ/Homo/SemLAold/fusion75epoch.ckpt"
matcher = SemLA().cuda()
matcher.load_state_dict(torch.load(reg_weight_path), strict=False)
matcher.load_state_dict(torch.load(fusion_weight_path), strict=False)
matcher.eval()
files=sorted(files)
i=3
# for file in files[i-1:i]:
for file in files:
    
    # 初始化平滑Homography管理器
    homo_manager = SmoothHomographyManager(
        max_translation_diff=80.0,  # 最大平移差異 (像素)
        max_rotation_diff=0.05,     # 最大旋轉差異 (弧度)
        smooth_alpha=0.1            # 平滑係數 (越小越平滑)
    )
    # 取得檔名跟副檔名
    filename, ext = os.path.basename(file).split(".")

    # 取得目錄位置
    parent_dir = os.path.dirname(file)

    # 取得 IR 與 EO 的影片路徑，並且檢查是否存在
    # 會用撈取是因為可能目標與來源不同副檔名
    path_ir = file
    path_eo = glob.glob(os.path.join(parent_dir, f"{filename.replace('_IR', '_EO')}.*"))
    if len(path_eo) == 0:
        assert False, f"EO file not found for {file}"
    path_eo = path_eo[0]

    # 取得 EO 和 IR 的影片
    cap_ir, cap_eo = cv2.VideoCapture(path_ir), cv2.VideoCapture(path_eo)

    # 取得影片的幀率、幀數、寬度、高度，並計算兩部影片的幀率比率
    fps_ir, fps_eo = cap_ir.get(cv2.CAP_PROP_FPS), cap_eo.get(cv2.CAP_PROP_FPS)
    H_ir, W_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_eo, W_eo = int(cap_eo.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap_eo.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count_ir, frame_count_eo = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_eo.get(cv2.CAP_PROP_FRAME_COUNT))

    # 計算 EO 的新尺寸、EO 的偏移量、幀率比率
    # 以 IR 為主，EO 才需要處理
    rate = int(fps_ir) // int(fps_eo)
    print(f"* {filename}")
    print(f"  - IR: {int(fps_ir)} fps, {frame_count_ir} frames, {W_ir}x{H_ir}")
    print(f"  - EO: {int(fps_eo)} fps, {frame_count_eo} frames, {W_eo}x{H_eo}")
    print(f"  - Rate: {rate} (IR:EO)")
    print(f"  - Target Size: {TARGET_W}x{TARGET_H}")

    # 若影片屬於需要時間延遲的影片，則需要跳過一些幀數
    if filename in time_delay.keys():
        delay = int(time_delay[filename])
        for _ in range(delay):
            ret_ir, frame_ir = cap_ir.read()
            frame_count_ir -= 1
    
    # 建立影片
    writer = cv2.VideoWriter(os.path.join(target_path, f"{filename}_noShake.mp4"), 
                           cv2.VideoWriter_fourcc(*'mp4v'), fps_ir, (TARGET_W * 3, TARGET_H))


    # 開始影片處理
    cnt = 0
    M = None
    temp_pair = np.zeros((TARGET_H, TARGET_W*2, 3), dtype=np.uint8)

    print(f"  - 平滑參數: 最大平移差異={homo_manager.max_translation_diff}px, 最大旋轉差異={homo_manager.max_rotation_diff}rad, 平滑係數={homo_manager.smooth_alpha}")
    while True:
        # 取得單幀
        ret_ir, frame_ir = cap_ir.read()
        ret_eo, frame_eo = cap_eo.read()
        if not ret_ir or not ret_eo:
            break

        # 步驟1: 讀取並裁剪EO影像
        eo_cropped = frame_eo[Vcut_y:Vcut_y+Vcut_h, Vcut_x:Vcut_x+Vcut_w]
        
        # 調整尺寸
        img_ir = cv2.resize(frame_ir, (TARGET_W, TARGET_H))
        img_eo = cv2.resize(eo_cropped, (TARGET_W, TARGET_H))
        
        # 轉換為灰階
        gray_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
        gray_eo = cv2.cvtColor(img_eo, cv2.COLOR_BGR2GRAY)

        # 每 compute_per_frame 幀計算一次轉換矩陣
        if cnt % compute_per_frame == 0:
            # 步驟2: 將EO(已裁剪)和IR給model得出雙方的特徵點
            ten_ir = torch.from_numpy(gray_ir)[None, None].cuda().float() / 255.0
            ten_eo = torch.from_numpy(gray_eo)[None, None].cuda().float() / 255.0

            # 取得 IR 與 EO 的特徵點
            with torch.no_grad():
                mkpts_eo, mkpts_ir, _, _, _ = matcher(ten_eo, ten_ir, "scene")
                mkpts_ir = mkpts_ir.cpu().numpy()
                mkpts_eo = mkpts_eo.cpu().numpy()

            # 步驟3: 雙方特徵點去使用homo轉換eo圖片去對齊ir
            if len(mkpts_ir) >= 4 and len(mkpts_eo) >= 4:
                M_candidate, mask = cv2.findHomography(mkpts_eo, mkpts_ir, cv2.RANSAC, 10.0, maxIters=5000, confidence=0.5)
                if M_candidate is not None:
                    mask = mask.ravel().astype(bool)
                    mkpts_ir = mkpts_ir[mask]
                    mkpts_eo = mkpts_eo[mask]
                    
                    # 步驟4: 檢查homo差異並決定是否更新
                    if homo_manager.previous_homo is not None:
                        trans_diff, rot_diff = homo_manager.calculate_homography_difference(
                            homo_manager.previous_homo, M_candidate)
                        print(f"Frame {cnt}: 平移差異={trans_diff:.2f}px, 旋轉差異={rot_diff:.4f}rad")
                        
                        if trans_diff > homo_manager.max_translation_diff or rot_diff > homo_manager.max_rotation_diff:
                            print(f"  -> 差異太大，保持前一次homography")
                            M = homo_manager.get_current()
                        else:
                            print(f"  -> 差異適中，平滑更新homography (混合係數={homo_manager.smooth_alpha})")
                            M = homo_manager.update_homography(M_candidate)
                    else:
                        print(f"Frame {cnt}: 首次homography更新")
                        M = homo_manager.update_homography(M_candidate)
                else:
                    # 如果無法計算homography，使用之前的
                    M = homo_manager.get_current()
                    print(f"Frame {cnt}: 無法計算homography，使用之前的")
            else:
                # 如果特徵點不足，使用之前的homography
                M = homo_manager.get_current()
                print(f"Frame {cnt}: 特徵點不足，使用之前的homography")

        # 邊緣處理
        blur = cv2.GaussianBlur(gray_eo, (5, 5), 0)
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

        border = 2
        edge_x = np.zeros_like(sobel_x)
        edge_y = np.zeros_like(sobel_y)

        edge_x = np.where(sobel_x < 1.0, np.roll(sobel_x, border, axis=1), edge_x)
        edge_x = np.where(sobel_x > 1.0, np.roll(-sobel_x, -border, axis=1), edge_x)
        edge_y = np.where(sobel_y < 1.0, np.roll(sobel_y, border, axis=0), edge_y)
        edge_y = np.where(sobel_y > 1.0, np.roll(-sobel_y, -border, axis=0), edge_y)

        edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge = sobel - edge
        edge = np.clip(edge, -255, 255)
        edge = np.expand_dims(edge, axis=-1)
        edge = np.repeat(edge, 3, axis=-1)

        # 將 EO 影像轉換到 IR 的座標系統
        if M is not None:
            edge = cv2.warpPerspective(edge, M, (TARGET_W, TARGET_H))

        # 融合影像 
        img_combined = img_ir + edge
        img_combined = np.clip(img_combined, 0, 255).astype(np.uint8)

        # 畫上特徵點
        if (cnt % compute_per_frame == 0) and (M is not None) and 'mkpts_ir' in locals() and 'mkpts_eo' in locals():
            temp_pair = np.hstack((img_ir, img_eo))

            for pt_ir, pt_eo in zip(mkpts_ir, mkpts_eo):
                x_ir, y_ir = pt_ir
                x_eo, y_eo = pt_eo
                x_eo += TARGET_W

                cv2.circle(temp_pair, (int(x_ir), int(y_ir)), 5, (0, 255, 0), -1)
                cv2.circle(temp_pair, (int(x_eo), int(y_eo)), 5, (0, 0, 255), -1)
                cv2.line(temp_pair, (int(x_ir), int(y_ir)), (int(x_eo), int(y_eo)), (255, 0, 0), 1)

        # 輸出影像
        img = np.hstack((temp_pair, img_combined))
        writer.write(img)

        # 將 EO 影片的幀數調整到 IR 的幀率
        for _ in range(rate - 1):
            ret_ir, frame_ir = cap_ir.read()

        # 迭代
        cnt += 1

    writer.release()
    cap_ir.release()
    cap_eo.release()
    print("Finished processing:", filename)
    