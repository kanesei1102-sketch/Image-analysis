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
        sens_a = st.slider("起点A感度", 10,
