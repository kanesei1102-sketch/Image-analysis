import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Bio-Image Quantifier Fixed", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Precision Edition")
st.caption("2025年完遂仕様：色逆転バグ修正 & 赤色検出強化版")

# --- 色定義 (HSV: OpenCVスケール H:0-180, S:0-255, V:0-255) ---
# 赤色は0付近と180付近の両方にあるため、特殊処理します
COLOR_RANGES = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)":   {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "青 (DAPI)":  {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])},
    # 赤は関数内で別途定義
}

def get_mask(hsv_img, color_name, sens):
    if color_name == "赤 (RFP)":
        # 赤は色相環の0度付近と180度付近の両方を拾う必要がある
        lower1 = np.array([0, 50, 50])
        upper1 = np.array([10 + sens//2, 255, 255])
        lower2 = np.array([170 - sens//2, 50, 50])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower1, upper1)
        mask2 = cv2.inRange(hsv_img, lower2, upper2)
        return mask1 | mask2
    else:
        conf = COLOR_RANGES[color_name]
        l = np.clip(conf["lower"] - sens, 0, 255)
        u = np.clip(conf["upper"] + sens, 0, 255)
        return cv2.inRange(hsv_img, l, u)

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# --- サイドバー ---
with st.sidebar:
    st.header("Analysis Recipe")
    mode = st.selectbox("解析モード:", [
        "1. 単色面積率 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 汎用共局在解析 (Colocalization)",
        "4. 汎用空間距離解析 (Spatial Distance)"
    ])
    sample_group = st.text_input("グループ名 (X軸):", value="Control")
    st.divider()

    if mode == "1. 単色面積率 (Area)":
        target_a = st.selectbox("解析する色:", ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"])
        sens_a = st.slider("感度", 10, 100, 40)
    
    elif mode == "2. 細胞核カウント (Count)":
        min_size = st.slider("最小細胞サイズ (px)", 10, 500, 50)

    elif mode == "3. 汎用共局在解析 (Colocalization)":
        st.info("2色の重なりを解析")
        target_a = st.selectbox("チャンネルA (基準):", ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"], index=1)
        sens_a = st.slider("A感度", 10, 100, 40)
        target_b = st.selectbox("チャンネルB (対象):", ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"], index=2)
        sens_b = st.slider("B感度", 10, 100, 40)

    elif mode == "4. 空間距離解析 (Spatial Distance)":
        target_a = st.selectbox("起点色(A):", ["赤 (RFP)", "緑 (GFP)", "青 (DAPI)"], index=0)
        sens_a = st.slider("起点A感度", 10, 100, 40)
        target_b = st.selectbox("対象色(B):", ["緑 (GFP)", "青 (DAPI)", "赤 (RFP)"], index=1)
        sens_b = st.slider("対象B感度", 10, 100, 40)

    if st.button("履歴をすべて削除"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メインロジック ---
uploaded_file = st.file_uploader("画像をアップロード...", type=["jpg", "png", "tif"])

if uploaded_file:
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1) # OpenCVはデフォルトでBGR
    
    if img_bgr is not None:
        # 【修正】ここで確実に RGB に変換してから HSV にする
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        val, unit = 0.0, ""
        res_display = img_rgb.copy()

        # 1. 面積率
        if mode == "1. 単色面積率 (Area)":
            mask = get_mask(img_hsv, target_a, sens_a)
            val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
            unit = f"% ({target_a})"
            res_display = mask

        # 2. カウント
        elif mode == "2. 細胞核カウント (Count)":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in cnts if cv2.contourArea(c) > min_size]
            val, unit = len(valid), "cells"
            cv2.drawContours(res_display, valid, -1, (0,255,0), 2)

        # 3. 共局在
        elif mode == "3. 汎用共局在解析 (Colocalization)":
            mask_a = get_mask(img_hsv, target_a, sens_a)
            mask_b = get_mask(img_hsv, target_b, sens_b)
            coloc = cv2.bitwise_and(mask_a, mask_b)
            
            denom = cv2.countNonZero(mask_a)
            val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
            unit = f"% ({target_b} in {target_a})"
            
            # 基準(A)を緑、対象(B)を赤、重なりを黄色で表示
            res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])

        # 4. 空間距離
        elif mode == "4. 空間距離解析 (Spatial Distance)":
            mask_a = get_mask(img_hsv, target_a, sens_a)
            mask_b = get_mask(img_hsv, target_b, sens_b)
            pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
            
            if pts_a and pts_b:
                dists = []
                for pa in pts_a:
                    d = np.min([np.linalg.norm(pa - pb) for pb in pts_b])
                    dists.append(d)
                val = np.mean(dists)
            else:
                val = 0
            unit = "px"
            res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)

        # 表示
        c1, c2 = st.columns(2)
        c1.image(img_rgb, caption="Original")
        c2.image(res_display, caption="Analysis View")
        st.subheader(f"📊 Result: {val:.2f} {unit}")
        
        if st.button("履歴に追加"):
            st.session_state.analysis_history.append({"Group": sample_group, "Value": val, "Unit": unit})
            st.success(f"Added: {val:.2f}")

# --- グラフ ---
st.divider()
if st.session_state.analysis_history:
    df = pd.DataFrame(st.session_state.analysis_history)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="Group", y="Value", ax=ax, palette="muted", alpha=0.6, errorbar="sd", capsize=.1)
    sns.stripplot(data=df, x="Group", y="Value", ax=ax, color=".2", size=8, jitter=True)
    ax.set_ylabel(f"Value ({df['Unit'].iloc[-1]})")
    st.pyplot(fig)
    st.dataframe(df)
