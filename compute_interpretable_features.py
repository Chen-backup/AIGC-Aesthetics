import os
import csv
import cv2
import numpy as np
from tqdm import tqdm

# ===================== 路径 =====================
IMAGE_DIR = r"G:\E\CJH-SJTU\课题组\图像美学\Dataset\face_dataset"
INPUT_CSV = r"G:\E\CJH-SJTU\课题组\图像美学\code_3\face_features.csv"
OUTPUT_CSV = r"G:\E\CJH-SJTU\课题组\图像美学\code_3\interpretable_face_features.csv"


# ===================== 辅助函数 =====================
def calculate_angle(p1, p2, p3):
    """计算三个点之间的角度"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) /
                                         (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1.0, 1.0)))
    return angle


# ===================== 主程序 =====================
def main():
    # 读取face_features.csv
    face_data = []
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            face_data.append(row)

    print(f"成功读取 {len(face_data)} 条人脸数据")

    rows = []

    for face_row in tqdm(face_data):
        image_name = face_row['image_name']
        status = face_row['status']

        if status != 'ok':
            rows.append({
                "image_name": image_name,
                "status": status
            })
            continue

        # 读取边界框
        x1 = float(face_row['bbox_x1'])
        y1 = float(face_row['bbox_y1'])
        x2 = float(face_row['bbox_x2'])
        y2 = float(face_row['bbox_y2'])
        w = x2 - x1
        h = y2 - y1

        # 读取5个关键点
        le_x = float(face_row['kps0_x'])  # 左眼
        le_y = float(face_row['kps0_y'])
        re_x = float(face_row['kps1_x'])  # 右眼
        re_y = float(face_row['kps1_y'])
        nose_x = float(face_row['kps2_x'])  # 鼻尖
        nose_y = float(face_row['kps2_y'])
        ml_x = float(face_row['kps3_x'])  # 左嘴角
        ml_y = float(face_row['kps3_y'])
        mr_x = float(face_row['kps4_x'])  # 右嘴角
        mr_y = float(face_row['kps4_y'])

        det_score = float(face_row['det_score'])

        # 计算关键点的中间点
        eye_center_x = (le_x + re_x) / 2
        eye_center_y = (le_y + re_y) / 2
        mouth_center_x = (ml_x + mr_x) / 2
        mouth_center_y = (ml_y + mr_y) / 2

        # ========= 基础特征 =========
        face_w = w
        face_h = h
        face_hw_ratio = h / w if w > 0 else 0
        face_area = w * h

        # ========= 眼睛特征 =========
        eye_dist = np.hypot(le_x - re_x, le_y - re_y)
        eye_face_w_ratio = eye_dist / face_w if face_w > 0 else 0
        eye_face_h_ratio = eye_dist / face_h if face_h > 0 else 0
        eye_y_ratio = eye_center_y / face_h if face_h > 0 else 0

        # ========= 鼻子特征 =========
        nose_eye_dist = np.hypot(nose_x - eye_center_x, nose_y - eye_center_y)
        nose_eye_ratio = nose_eye_dist / face_h if face_h > 0 else 0
        nose_eye_h_ratio = (nose_y - eye_center_y) / face_h if face_h > 0 else 0

        # ========= 嘴巴特征 =========
        mouth_w = np.hypot(ml_x - mr_x, ml_y - mr_y)
        mouth_face_w_ratio = mouth_w / face_w if face_w > 0 else 0
        mouth_eye_dist = np.hypot(mouth_center_x - eye_center_x, mouth_center_y - eye_center_y)
        mouth_eye_ratio = mouth_eye_dist / face_h if face_h > 0 else 0
        mouth_nose_dist = np.hypot(mouth_center_x - nose_x, mouth_center_y - nose_y)
        mouth_nose_ratio = mouth_nose_dist / face_h if face_h > 0 else 0
        mouth_y_ratio = mouth_center_y / face_h if face_h > 0 else 0

        # ========= 面部比例特征（三庭） =========
        upper_face_h = eye_center_y - y1
        middle_face_h = nose_y - eye_center_y
        lower_face_h = y2 - nose_y

        upper_mid_ratio = upper_face_h / middle_face_h if middle_face_h > 0 else 0
        mid_lower_ratio = middle_face_h / lower_face_h if lower_face_h > 0 else 0
        upper_lower_ratio = upper_face_h / lower_face_h if lower_face_h > 0 else 0

        ideal_ratio = 1.0
        three_courts_balance = 1.0 - (
                abs(upper_mid_ratio - ideal_ratio) +
                abs(mid_lower_ratio - ideal_ratio) +
                abs(upper_lower_ratio - ideal_ratio)
        ) / 3

        # ========= 对称性特征 =========
        eye_y_diff = abs(le_y - re_y)
        eye_sym_score = 1.0 - (eye_y_diff / (face_h + 1e-6))

        nose_center_diff = abs(nose_x - eye_center_x)
        nose_sym_score = 1.0 - (nose_center_diff / (face_w + 1e-6))

        mouth_center_diff = abs(mouth_center_x - nose_x)
        mouth_sym_score = 1.0 - (mouth_center_diff / (face_w + 1e-6))

        total_symmetry = (eye_sym_score + nose_sym_score + mouth_sym_score) / 3

        mouth_left_y_diff = abs(ml_y - mouth_center_y)
        mouth_right_y_diff = abs(mr_y - mouth_center_y)
        mouth_vertical_sym = 1.0 - (abs(mouth_left_y_diff - mouth_right_y_diff) / (face_h + 1e-6))

        # ========= 角度特征 =========
        le_nose_re_angle = calculate_angle((le_x, le_y), (nose_x, nose_y), (re_x, re_y))
        ml_nose_mr_angle = calculate_angle((ml_x, ml_y), (nose_x, nose_y), (mr_x, mr_y))
        le_ml_nose_angle = calculate_angle((le_x, le_y), (ml_x, ml_y), (nose_x, nose_y))
        re_mr_nose_angle = calculate_angle((re_x, re_y), (mr_x, mr_y), (nose_x, nose_y))

        # ========= 表情相关特征 =========
        mouth_eye_ratio_width = mouth_w / eye_dist if eye_dist > 0 else 0
        mouth_nose_y_diff = mouth_center_y - nose_y
        mouth_nose_y_ratio = mouth_nose_y_diff / face_h if face_h > 0 else 0

        # ========= 图像质量特征 =========
        img_path = os.path.join(IMAGE_DIR, image_name)
        brightness = 0
        brightness_normalized = 0
        contrast = 0
        contrast_normalized = 0
        clarity = 0
        saturation = 0
        saturation_normalized = 0
        hue = 0
        edge_density = 0
        face_brightness = 0
        face_contrast = 0
        face_clarity = 0

        if os.path.exists(img_path):
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()
                brightness_normalized = brightness / 255.0
                contrast = gray.std()
                contrast_normalized = contrast / 128.0
                laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
                clarity = laplacian

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                saturation = hsv[:, :, 1].mean()
                saturation_normalized = saturation / 255.0
                hue = hsv[:, :, 0].mean()

                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

                # 人脸区域质量
                y1_int = max(0, int(y1))
                y2_int = min(img.shape[0], int(y2))
                x1_int = max(0, int(x1))
                x2_int = min(img.shape[1], int(x2))

                if y2_int > y1_int and x2_int > x1_int:
                    face_img = img[y1_int:y2_int, x1_int:x2_int]
                    if face_img.size > 0:
                        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        face_brightness = face_gray.mean()
                        face_contrast = face_gray.std()
                        face_laplacian = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                        face_clarity = face_laplacian

        # ========= 保存 =========
        row = {
            "image_name": image_name,
            "status": "ok",

            # 基础特征
            "face_w": round(float(face_w), 2),
            "face_h": round(float(face_h), 2),
            "face_hw_ratio": round(float(face_hw_ratio), 3),
            "face_area": round(float(face_area), 2),

            # 眼睛特征
            "eye_dist": round(float(eye_dist), 2),
            "eye_face_w_ratio": round(float(eye_face_w_ratio), 3),
            "eye_face_h_ratio": round(float(eye_face_h_ratio), 3),
            "eye_y_ratio": round(float(eye_y_ratio), 3),

            # 鼻子特征
            "nose_eye_dist": round(float(nose_eye_dist), 2),
            "nose_eye_ratio": round(float(nose_eye_ratio), 3),
            "nose_eye_h_ratio": round(float(nose_eye_h_ratio), 3),

            # 嘴巴特征
            "mouth_w": round(float(mouth_w), 2),
            "mouth_face_w_ratio": round(float(mouth_face_w_ratio), 3),
            "mouth_eye_dist": round(float(mouth_eye_dist), 2),
            "mouth_eye_ratio": round(float(mouth_eye_ratio), 3),
            "mouth_nose_dist": round(float(mouth_nose_dist), 2),
            "mouth_nose_ratio": round(float(mouth_nose_ratio), 3),
            "mouth_y_ratio": round(float(mouth_y_ratio), 3),

            # 三庭比例特征
            "upper_face_h": round(float(upper_face_h), 2),
            "middle_face_h": round(float(middle_face_h), 2),
            "lower_face_h": round(float(lower_face_h), 2),
            "upper_mid_ratio": round(float(upper_mid_ratio), 3),
            "mid_lower_ratio": round(float(mid_lower_ratio), 3),
            "upper_lower_ratio": round(float(upper_lower_ratio), 3),
            "three_courts_balance": round(float(three_courts_balance), 3),

            # 对称性特征
            "eye_sym": round(float(eye_sym_score), 3),
            "nose_sym": round(float(nose_sym_score), 3),
            "mouth_sym": round(float(mouth_sym_score), 3),
            "mouth_vertical_sym": round(float(mouth_vertical_sym), 3),
            "total_symmetry": round(float(total_symmetry), 3),

            # 角度特征
            "le_nose_re_angle": round(float(le_nose_re_angle), 2),
            "ml_nose_mr_angle": round(float(ml_nose_mr_angle), 2),
            "le_ml_nose_angle": round(float(le_ml_nose_angle), 2),
            "re_mr_nose_angle": round(float(re_mr_nose_angle), 2),

            # 表情特征
            "mouth_eye_ratio_width": round(float(mouth_eye_ratio_width), 3),
            "mouth_nose_y_diff": round(float(mouth_nose_y_diff), 2),
            "mouth_nose_y_ratio": round(float(mouth_nose_y_ratio), 3),

            # 图像质量特征
            "brightness": round(float(brightness), 2),
            "brightness_normalized": round(float(brightness_normalized), 3),
            "contrast": round(float(contrast), 2),
            "contrast_normalized": round(float(contrast_normalized), 3),
            "clarity": round(float(clarity), 2),
            "saturation": round(float(saturation), 2),
            "saturation_normalized": round(float(saturation_normalized), 3),
            "hue": round(float(hue), 2),
            "edge_density": round(float(edge_density), 5),

            # 人脸区域图像质量
            "face_brightness": round(float(face_brightness), 2),
            "face_contrast": round(float(face_contrast), 2),
            "face_clarity": round(float(face_clarity), 2),

            # 检测信息
            "det_score": round(float(det_score), 3),
        }
        rows.append(row)

    # 保存
    if rows:
        fieldnames = list(rows[0].keys())

        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    print("✅ 可解释人脸特征计算完成！")
    print(f"文件保存：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
