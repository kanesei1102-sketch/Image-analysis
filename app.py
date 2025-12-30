import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- ページ設定 ---
st.set_page_config(page_title="Professional Bio-Quantifier Ultimate", layout="wide")

# 解析履歴の保持
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Ultimate Edition")
st.caption("2025年完遂仕様：多重染色・共局在・空間距離・統計解析を完全統合")

# --- サイドバー：解析設定 ---
with st.sidebar:
    st.header("Analysis Recipe")
    
    # 4つのモードを完全搭載
    mode = st.selectbox("解析モードを選択:", [
        "1. 多重染色分離/面積 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 共局在解析 (Colocalization)",
        "4. 空間距離解析 (Spatial Distance)"
    ])
    
    # 共通設定: グループ名
    sample_group = st.text_input("グループ名 (X軸):", placeholder="例: Control, Treatment")
    
    st.divider()
    st.subheader("Parameter Tuning")

    # モード別パラメータ
    if mode == "1. 多重染色分離/面積 (Area)":
        target_color = st.radio("ターゲット色:", ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)"])
        sensitivity = st.slider("色抽出感度", 10, 100, 40)
    
    elif mode == "2. 細胞核カウント (Count)":
        min_size = st.slider("最小細胞サイズ (px)", 10, 500, 50)
        
    elif mode == "3. 共局在解析 (Colocalization)":
        st.info("緑(Green)と赤(Red)の重なりを解析")
        sens_g = st.slider("Green感度", 10, 100, 40)
        sens_r = st.slider("Red感度", 10, 100, 40)

    elif mode == "4. 空間距離解析 (Spatial Distance)":
        st.info("群A(赤)と群B(青/緑)の重心間距離を解析")
        color_a = "赤 (Red)"
        color_b = st.radio("群Bの色:", ["緑 (Green)", "青 (Blue/DAPI)"])
        dist_sens = st.slider("検出感度", 10, 100, 40)

    st.divider()
    if st.button("履歴・グラフをリセット"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メイン：画像解析ロジック ---
uploaded_file = st.file_uploader("画像をアップロード...", type=["jpg", "png", "tif"])

if uploaded_file:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    val = 0.0
    unit = ""
    result_img = img_rgb.copy()

    # ---------------------------------------------------------
    # 1. 多重染色分離 / 面積率
    # ---------------------------------------------------------
    if mode == "1. 多重染色分離/面積 (Area)":
        lower, upper = None, None
        if target_color == "茶色 (DAB)":
            lower = np.array([10, 50, 20])
            upper = np.array([30, 255, 255])
        elif target_color == "緑 (GFP)":
            lower = np.array([35, 50, 50])
            upper = np.array([85, 255, 255])
        else: # 赤
            lower = np.array([0, 50, 50])
            upper = np.array([10, 255, 255])
            # 赤は170-180も含むが簡易版として0-10を使用、必要ならmask結合
        
        # 感度適用
        lower = np.clip(lower - sensitivity, 0, 255)
        upper = np.clip(upper + sensitivity, 0, 255)
        
        mask = cv2.inRange(img_hsv, lower, upper)
        val = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100
        unit = "% (Area)"
        result_img = mask

    # ---------------------------------------------------------
    # 2. 細胞核カウント
    # ---------------------------------------------------------
    elif mode == "2. 細胞核カウント (Count)":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > min_size]
        
        cv2.drawContours(result_img, valid, -1, (0, 255, 0), 2)
        val = len(valid)
        unit = "cells"

    # ---------------------------------------------------------
    # 3. 共局在解析 (Colocalization)
    # ---------------------------------------------------------
    elif mode == "3. 共局在解析 (Colocalization)":
        # Green Mask
        lower_g = np.array([35, 50, 50])
        upper_g = np.array([85, 255, 255])
        mask_g = cv2.inRange(img_hsv, np.clip(lower_g-sens_g,0,255), np.clip(upper_g+sens_g,0,255))
        
        # Red Mask
        lower_r = np.array([0, 50, 50])
        upper_r = np.array([10, 255, 255])
        mask_r = cv2.inRange(img_hsv, np.clip(lower_r-sens_r,0,255), np.clip(upper_r+sens_r,0,255))
        
        # Overlap (AND)
        coloc = cv2.bitwise_and(mask_g, mask_r)
        
        # 共局在率 = (重なり面積 / 緑面積) * 100
        area_g = cv2.countNonZero(mask_g)
        area_coloc = cv2.countNonZero(coloc)
        val = (area_coloc / area_g * 100) if area_g > 0 else 0
        unit = "% (Coloc/Green)"
        
        # 可視化: 緑+赤+黄(重なり)
        result_img = cv2.merge([mask_r, mask_g, np.zeros_like(mask_g)])

    # ---------------------------------------------------------
    # 4. 空間距離解析 (Spatial Distance)
    # ---------------------------------------------------------
    elif mode == "4. 空間距離解析 (Spatial Distance)":
        # Group A (Red)
        mask_a = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        
        # Group B (Green or Blue)
        if color_b == "緑 (Green)":
            mask_b = cv2.inRange(img_hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        else: # Blue (DAPIなど: H 100-130)
            mask_b = cv2.inRange(img_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            
        def get_centroids(m):
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = []
            for c in cnts:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    # 重心計算：ここが長すぎて切れていた可能性があります
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    pts.append(np.array([cx, cy]))
            return pts
