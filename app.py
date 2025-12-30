import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- ページ設定 ---
st.set_page_config(page_title="Bio-Image Quantifier Pro", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Universal Edition")
st.caption("2025年完遂仕様：全色対応・自動解析・統計エンジン")

# --- 色定義 ---
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([130, 255, 255])}
}

# --- ヘルパー関数 ---
def get_mask(hsv_img, color_name, sens):
    conf = COLOR_MAP[color_name]
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

    # パラメータ設定
    if mode == "1. 単色面積率 (Area)":
        target_a = st.selectbox("解析する色:", list(COLOR_MAP.keys()))
        sens_a = st.slider("感度", 10, 100, 40)
    
    elif mode == "2. 細胞核カウント (Count)":
        min_size = st.slider("最小細胞サイズ (px)", 10, 500, 50)

    elif mode == "3. 汎用共局在解析 (Colocalization)":
        target_a = st.selectbox("チャンネルA (基準):", list(COLOR_MAP.keys()), index=1)
        sens_a = st.slider("チャンネルA感度", 10, 100, 40)
        target_b = st.selectbox("チャンネルB (対象):", list(COLOR_MAP.keys()), index=2)
        sens_b = st.slider("チャンネルB感度", 10, 100, 40)

    elif mode == "4. 空間距離解析 (Spatial Distance)":
        target_a = st.selectbox("起点となる色(A):", list(COLOR_MAP.keys()), index=2)
        sens_a = st.slider("起点A感度", 10,10, 100, 40)
        target_b = st.selectbox("対象となる色(B):", list(COLOR_MAP.keys()), index=3)
        sens_b = st.slider("対象B感度", 10, 100, 40)

    if st.button("履歴をすべて削除"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メインロジック ---
uploaded_file = st.file_uploader("画像をアップロード...", type=["jpg", "png", "tif"])

if uploaded_file:
    # 【重要】ファイルポインタを先頭に戻す（これが修正点）
    uploaded_file.seek(0)
    
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("画像の読み込みに失敗しました。ファイル形式を確認してください。")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        val, unit = 0.0, ""
        res_display = img_rgb.copy()

        # --------------------
        # 1. 面積率
        # --------------------
        if mode == "1. 単色面積率 (Area)":
            mask = get_mask(img_hsv, target_a, sens_a)
            val = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100
            unit = f"% ({target_a})"
            res_display = mask

        # --------------------
        # 2. カウント
        # --------------------
        elif mode == "2. 細胞核カウント (Count)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in cnts if cv2.contourArea(c) > min_size]
            val, unit = len(valid), "cells"
            cv2.drawContours(res_display, valid, -1, (0,255,0), 2)

        # --------------------
        # 3. 共局在
        # --------------------
        elif mode == "3. 汎用共局在解析 (Colocalization)":
            mask_a = get_mask(img_hsv, target_a, sens_a)
            mask_b = get_mask(img_hsv, target_b, sens_b)
            coloc = cv2.bitwise_and(mask_a, mask_b)
            val = (cv2.countNonZero(coloc) / cv2.countNonZero(mask_a) * 100) if cv2.countNonZero(mask_a) > 0 else 0
            unit = f"% ({target_b} in {target_a})"
            res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])

        # --------------------
        # 4. 空間距離
        # --------------------
        elif mode == "4. 空間距離解析 (Spatial Distance)":
            mask_a = get_mask(img_hsv, target_a, sens_a)
            mask_b = get_mask(img_hsv, target_b, sens_b)
            pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
            
            if pts_a and pts_b:
                # 距離計算の高速化
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
