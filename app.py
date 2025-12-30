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

st.title("🔬 Bio-Image Quantifier: Universal Multi-Channel Edition")
st.caption("2025年完遂仕様：全色対応の共局在・空間距離・統計解析エンジン")

# --- 色定義の辞書 ---
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([130, 255, 255])}
}

def get_mask(hsv_img, color_name, sens):
    conf = COLOR_MAP[color_name]
    l = np.clip(conf["lower"] - sens, 0, 255)
    u = np.clip(conf["upper"] + sens, 0, 255)
    return cv2.inRange(hsv_img, l, u)

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
        target_a = st.selectbox("解析する色:", list(COLOR_MAP.keys()))
        sens_a = st.slider("感度", 10, 100, 40)
    
    elif mode == "2. 細胞核カウント (Count)":
        min_size = st.slider("最小細胞サイズ (px)", 10, 500, 50)

    elif mode == "3. 汎用共局在解析 (Colocalization)":
        st.info("2つの色の重なりを解析します")
        target_a = st.selectbox("チャンネルA (基準):", list(COLOR_MAP.keys()), index=1)
        sens_a = st.slider("チャンネルA感度", 10, 100, 40)
        target_b = st.selectbox("チャンネルB (対象):", list(COLOR_MAP.keys()), index=2)
        sens_b = st.slider("チャンネルB感度", 10, 100, 40)

    elif mode == "4. 空間距離解析 (Spatial Distance)":
        st.info("チャンネルAからチャンネルBへの最短距離")
        target_a = st.selectbox("起点となる色(A):", list(COLOR_MAP.keys()), index=2)
        sens_a = st.slider("起点A感度", 10, 100, 40)
        target_b = st.selectbox("対象となる色(B):", list(COLOR_MAP.keys()), index=3)
        sens_b = st.slider("対象B感度", 10, 100, 40)

    if st.button("履歴をすべて削除"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メインロジック ---
uploaded_file = st.file_uploader("画像をアップロード...", type=["jpg", "png", "tif"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    val, unit, res_display = 0.0, "", img_rgb.copy()

    # 1. 面積率
    if mode == "1. 単色面積率 (Area)":
        mask = get_mask(img_hsv, target_a, sens_a)
        val = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100
        unit = f"% ({target_a})"
        res_display = mask

    # 2. カウント
    elif mode == "2. 細胞核カウント (Count)":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_size]
        val, unit = len(valid), "cells"
        cv2.drawContours(res_display, valid, -1, (0,255,0), 2)

    # 3. 汎用共局在
    elif mode == "3. 汎用共局在解析 (Colocalization)":
        mask_a = get_mask(img_hsv, target_a, sens_a)
        mask_b = get_mask(img_hsv, target_b, sens_b)
        coloc = cv2.bitwise_and(mask_a, mask_b)
        val = (cv2.countNonZero(coloc) / cv2.countNonZero(mask_a) * 100) if cv2.countNonZero(mask_a) > 0 else 0
        unit = f"% ({target_b} in {target_a})"
        # Aを緑、Bを赤として合成表示（黄色が共局在）
        res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])

    # 4. 空間距離
    elif mode == "4. 空間距離解析 (Spatial Distance)":
        mask_a = get_mask(img_hsv, target_a, sens_a)
        mask_b = get_mask(img_hsv, target_b, sens_b)
        def get_pts(m):
            c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            p = []
            for cnt in c:
                M = cv2.moments(cnt)
                if M["m00"] != 0: p.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
            return p
