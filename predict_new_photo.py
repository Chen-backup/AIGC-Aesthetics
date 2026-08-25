import os
import cv2
import math
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import bambi as bmb
import arviz as az
from insightface.app import FaceAnalysis
from transformers import AutoImageProcessor, AutoModel

# =====================================================================
# 一、 全局路径配置 (⚠️ 运行前请务必核对以下路径！)
# =====================================================================
# 1. 待测图片路径
TEST_IMAGE_PATH = r"test_predict_photo/hch.png"  # 替换你想测试的照片路径

# 2. 模型依赖路径
INSIGHTFACE_ROOT = r"G:\E\CJH-SJTU\课题组\图像美学\code_3\insightface-base-local"  # ⚠️ 注意：不要带 \models\buffalo_l
PCA_MODEL_PATH = r"BYS_Fusion_28D_DINOv2_result/dinov2_pca_14d.pkl"
NC_TRACE_PATH = r"BYS_Fusion_28D_DINOv2_result/Fusion_28D_DINOv2_model_trace.nc"
LOCAL_DINO_DIR = r"G:\E\CJH-SJTU\课题组\图像美学\code_3\dinov2-base-local"

# 3. 原始训练集数据路径（用于提取标准化“旧尺子”）
EXCEL_RATINGS = r"ratings_for_bayesian_model.xlsx"
EXCEL_MAPPING = r"renumber&gender.xlsx"
CSV_GEOM = r"interpretable_face_features.csv"
CSV_DINO = r"PCA_14_dinov2.csv"

# 特征列名定义
GEOM_FEATURES = ['face_hw_ratio', 'eye_face_w_ratio', 'mouth_face_w_ratio', 'three_courts_balance',
                 'upper_lower_ratio', 'eye_y_ratio', 'total_symmetry', 'le_nose_re_angle',
                 'mouth_nose_ratio', 'face_brightness', 'face_contrast', 'face_clarity',
                 'saturation', 'edge_density']
DINO_FEATURES = [f'PC{i}' for i in range(1, 15)]
ALL_28_FEATURES = GEOM_FEATURES + DINO_FEATURES

# =====================================================================
# 二、 核心引擎预加载 (仅执行一次，避免重复耗时)
# =====================================================================
print("\n" + "=" * 50)
print("🚀 启动 28 维人脸美学终极评估系统...")
print("=" * 50)

# 1. 加载 InsightFace (14维可解释几何特征提取器)
print("⏳ [1/4] 正在唤醒 InsightFace 几何测量师...")
face_app = FaceAnalysis(name='buffalo_l', root=INSIGHTFACE_ROOT, allowed_modules=['detection'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 2. 加载 DINOv2 (使用 transformers 读取本地 HuggingFace 格式)
print("⏳ [2/4] 正在从本地安全唤醒 DINOv2 视觉大模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载配套的图像处理器
dino_processor = AutoImageProcessor.from_pretrained(LOCAL_DINO_DIR)
# 加载模型本体
dinov2_model = AutoModel.from_pretrained(LOCAL_DINO_DIR).to(device)
dinov2_model.eval()

# 3. 加载 PCA 降维模具
print("⏳ [3/4] 正在加载 PCA 降维模具...")
pca_model = joblib.load(PCA_MODEL_PATH)

# 4. 加载贝叶斯大模型与标准化参数
print("⏳ [4/4] 正在读取训练集旧尺子并装载贝叶斯大脑...")
trace = az.from_netcdf(NC_TRACE_PATH)

# 数据合并以计算旧尺子
df_ratings = pd.read_excel(EXCEL_RATINGS)
df_mapping = pd.read_excel(EXCEL_MAPPING)
df_geom = pd.read_csv(CSV_GEOM)
df_dino = pd.read_csv(CSV_DINO)

geom_merge_key = 'face_id' if 'face_id' in df_geom.columns else 'image_name'
df_combined = pd.merge(df_mapping[['face_id', 'Number']], df_geom, left_on='face_id', right_on=geom_merge_key,
                       how='inner')
df = pd.merge(df_ratings, df_combined, left_on='image', right_on='Number', how='inner')
df = pd.merge(df, df_dino, on='image_name', how='inner')

df = df.dropna(subset=ALL_28_FEATURES + ['rating', 'rater', 'image'])
categories = sorted(df['rating'].unique())
df['rating'] = pd.Categorical(df['rating'], categories=categories, ordered=True)
rating_cats_numeric = np.array(categories).astype(float)

# ⭐️ 提取旧尺子 (mean & std)
training_stats = {}
for feat in ALL_28_FEATURES:
    mean_val = df[feat].mean()
    std_val = df[feat].std()
    training_stats[feat] = {'mean': mean_val, 'std': std_val}
    df[feat] = (df[feat] - mean_val) / std_val

df['rater'] = df['rater'].astype(str)
df['image'] = df['image'].astype(str)

# 搭建 Bambi 空壳
formula = f"rating ~ 1 + {' + '.join(ALL_28_FEATURES)} + (1|rater) + (1|image)"
model = bmb.Model(formula, data=df, family="cumulative")
print("✅ 系统全部就绪！准备开始评估...\n")


# =====================================================================
# 三、 特征提取模块
# =====================================================================
def calculate_angle(p1, p2, p3):
    v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1.0, 1.0)))


def extract_geom_14d(img_path):
    """提取 14维可解释几何特征"""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return None

    faces = face_app.get(img)
    if len(faces) == 0: return None

    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    w, h = x2 - x1, y2 - y1
    kps = face.kps

    eye_center_x, eye_center_y = (kps[0][0] + kps[1][0]) / 2, (kps[0][1] + kps[1][1]) / 2
    mouth_center_x, mouth_center_y = (kps[3][0] + kps[4][0]) / 2, (kps[3][1] + kps[4][1]) / 2

    face_hw_ratio = h / w if w > 0 else 0
    eye_dist = np.hypot(kps[0][0] - kps[1][0], kps[0][1] - kps[1][1])
    eye_face_w_ratio = eye_dist / w if w > 0 else 0
    eye_y_ratio = eye_center_y / h if h > 0 else 0

    mouth_w = np.hypot(kps[3][0] - kps[4][0], kps[3][1] - kps[4][1])
    mouth_face_w_ratio = mouth_w / w if w > 0 else 0
    mouth_nose_ratio = np.hypot(mouth_center_x - kps[2][0], mouth_center_y - kps[2][1]) / h if h > 0 else 0

    upper_face_h, middle_face_h, lower_face_h = eye_center_y - y1, kps[2][1] - eye_center_y, y2 - kps[2][1]
    upper_mid_ratio = upper_face_h / middle_face_h if middle_face_h > 0 else 0
    mid_lower_ratio = middle_face_h / lower_face_h if lower_face_h > 0 else 0
    upper_lower_ratio = upper_face_h / lower_face_h if lower_face_h > 0 else 0
    three_courts_balance = 1.0 - (abs(upper_mid_ratio - 1) + abs(mid_lower_ratio - 1) + abs(upper_lower_ratio - 1)) / 3

    eye_sym_score = 1.0 - (abs(kps[0][1] - kps[1][1]) / (h + 1e-6))
    nose_sym_score = 1.0 - (abs(kps[2][0] - eye_center_x) / (w + 1e-6))
    mouth_sym_score = 1.0 - (abs(mouth_center_x - kps[2][0]) / (w + 1e-6))
    total_symmetry = (eye_sym_score + nose_sym_score + mouth_sym_score) / 3

    le_nose_re_angle = calculate_angle(kps[0], kps[2], kps[1])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    edge_density = np.sum(cv2.Canny(gray, 50, 150) > 0) / (gray.shape[0] * gray.shape[1])

    face_brightness, face_contrast, face_clarity = 0, 0, 0
    y1_int, y2_int = max(0, int(y1)), min(img.shape[0], int(y2))
    x1_int, x2_int = max(0, int(x1)), min(img.shape[1], int(x2))

    if y2_int > y1_int and x2_int > x1_int:
        face_img = img[y1_int:y2_int, x1_int:x2_int]
        if face_img.size > 0:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_brightness = face_gray.mean()
            face_contrast = face_gray.std()
            face_clarity = cv2.Laplacian(face_gray, cv2.CV_64F).var()

    return {
        'face_hw_ratio': face_hw_ratio, 'eye_face_w_ratio': eye_face_w_ratio, 'mouth_face_w_ratio': mouth_face_w_ratio,
        'three_courts_balance': three_courts_balance, 'upper_lower_ratio': upper_lower_ratio,
        'eye_y_ratio': eye_y_ratio,
        'total_symmetry': total_symmetry, 'le_nose_re_angle': le_nose_re_angle, 'mouth_nose_ratio': mouth_nose_ratio,
        'face_brightness': face_brightness, 'face_contrast': face_contrast, 'face_clarity': face_clarity,
        'saturation': saturation, 'edge_density': edge_density
    }


def extract_dino_14d(img_path):
    pil_img = Image.open(img_path).convert('RGB')
    inputs = dino_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
        features_np = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    features_14d = pca_model.transform(features_np)[0]
    return {f'PC{i + 1}': val for i, val in enumerate(features_14d)}


# =====================================================================
# 四、 主预测流程
# =====================================================================
def predict_score(image_path):
    filename = os.path.basename(image_path)
    print(f"📸 开始分析照片: {filename}")

    # 1. 提取 14维几何白盒特征
    geom_feats = extract_geom_14d(image_path)
    if geom_feats is None:
        print(f"❌ 照片 {filename} 未检测到人脸，或者检测失败！")
        return

    # 🎯 新增逻辑：保存 14 个可解释特征到图片所在目录
    img_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(filename)[0]
    csv_save_path = os.path.join(img_dir, f"{base_name}_features.csv")

    # 构造 DataFrame 并保存 (使用 utf-8-sig 以便 Excel 正常打开中文路径)
    df_interpretable = pd.DataFrame([geom_feats])
    df_interpretable.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
    print(f"📝 14维可解释特征已保存至: {csv_save_path}")

    # 2. 提取 14维DINOv2黑盒特征
    dino_feats = extract_dino_14d(image_path)

    # 3. 28维特征合体
    raw_28_feats = {**geom_feats, **dino_feats}

    # 4. 标准化 (时空对齐)
    standardized_feats = {}
    for feat in ALL_28_FEATURES:
        raw_val = raw_28_feats[feat]
        mean_val = training_stats[feat]['mean']
        std_val = training_stats[feat]['std']
        standardized_feats[feat] = [(raw_val - mean_val) / std_val]

    # 5. 送入贝叶斯预测
    dummy_df = pd.DataFrame(standardized_feats)
    dummy_df['rater'] = "unknown"
    dummy_df['image'] = filename

    print("🧠 正在调用贝叶斯大模型进行非线性推断...")
    pred = model.predict(trace, data=dummy_df, kind="response_params", include_group_specific=False, inplace=False)

    pred_values = pred.posterior['p'].values if 'p' in pred.posterior.data_vars else pred.posterior[
        'rating_probs'].values
    expected_scores = np.sum(pred_values * rating_cats_numeric, axis=-1)

    final_score = expected_scores.mean()
    lower_hdi = np.percentile(expected_scores, 2.5)
    upper_hdi = np.percentile(expected_scores, 97.5)

    print("\n" + "=" * 45)
    print(" " * 10 + "🏆 终极颜值评估报告 🏆")
    print("=" * 45)
    print(f"📸 被测图片 : {filename}")
    print(f"🌟 客观期望分数 : {final_score:.3f}")
    print(f"📊 95% 置信区间 : [{lower_hdi:.3f} - {upper_hdi:.3f}]")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    if os.path.exists(TEST_IMAGE_PATH):
        predict_score(TEST_IMAGE_PATH)
    else:
        print(f"⚠️ 找不到测试图片：{TEST_IMAGE_PATH}。请检查路径是否正确。")