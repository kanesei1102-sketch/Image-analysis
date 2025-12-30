import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- ページ設定 ---
st.set_page_config(page_title="Bio-Image Quantifier", layout="wide")

# 履歴保持用のリスト
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Professional Bio-Image Quantifier")

# --- サイドバー：解析レシピ ---
with st.sidebar:
    st.header("Analysis Recipe")
    mode = st.selectbox("解析モードを選択:", [
        "1. 多重染色分離/面積 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 共局在解析 (Colocalization)",
        "4. 空間距離解析 (Spatial Distance)"
    ])
    sample_group = st.text_input("グループ名 (X軸):", value="Control")
    st.divider()
    
    # モード別パラメータ
    if mode == "1. 多重染色分離/面積 (Area)":
        target_color = st.radio("ターゲット色:", ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)"])
        sensitivity = st.slider("色抽出感度", 10, 100, 40)
    elif mode == "2. 細胞核カウント (Count)":
        min_size = st.slider("最小細胞サイズ (px)", 10, 500, 50)
    elif mode == "3. 共局在解析 (Colocalization)":
        sens_g = st.slider("Green感度", 10, 100, 40)
        sens_r = st.slider("Red感度", 10, 100, 40)
    elif mode == "4. 空間距離解析 (Spatial Distance)":
        color_b_name = st.radio("群Bの色:", ["緑 (Green)", "青 (Blue/DAPI)"])

    if st.button("履歴をすべて削除"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メイン：画像アップロード ---
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "tif"])

if uploaded_file:
    # 画像の読み込み（ここで確実にメモリに載せる）
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 解析用変数
    val = 0.0
    unit = ""
    res_display = None

    # --- 解析実行（ボタンなしで自動実行） ---
    if mode == "1. 多重染色分離/面積 (Area)":
        if target_color == "茶色 (DAB)": lower, upper = np.array([10, 50, 20]), np.array([30, 255, 255])
        elif target_color == "緑 (GFP)": lower, upper = np.array([35, 50, 50]), np.array([85, 255, 255])
        else: lower, upper = np.array([0, 50, 50]), np.array([10, 255, 255])
        
        mask = cv2.inRange(img_hsv, np.clip(lower-sensitivity,0,255), np.clip(upper+sensitivity,0,255))
        val = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100
        unit = "%"
        res_display = mask

    elif mode == "2. 細胞核カウント (Count)":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_size]
        val = len(valid)
        unit = "cells"
        res_display = img_rgb.copy()
        cv2.drawContours(res_display, valid, -1, (0,255,0), 2)

    elif mode == "3. 共局在解析 (Colocalization)":
        m_g = cv2.inRange(img_hsv, np.array([35,50,50]), np.array([85,255,255]))
        m_r = cv2.inRange(img_hsv, np.array([0,50,50]), np.array([10,255,255]))
        coloc = cv2.bitwise_and(m_g, m_r)
        val = (cv2.countNonZero(coloc) / cv2.countNonZero(m_g) * 100) if cv2.countNonZero(m_g) > 0 else 0
        unit = "% (Coloc)"
        res_display = cv2.merge([m_r, m_g, np.zeros_like(m_g)])

    elif mode == "4. 空間距離解析 (Spatial Distance)":
        m_a = cv2.inRange(img_hsv, np.array([0,50,50]), np.array([10,255,255]))
        if color_b_name == "緑 (Green)": m_b = cv2.inRange(img_hsv, np.array([35,50,50]), np.array([85,255,255]))
        else: m_b = cv2.inRange(img_hsv, np.array([100,50,50]), np.array([130,255,255]))
        
        def get_pts(m):
            c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            p = []
            for cnt in c:
                M = cv2.moments(cnt)
                if M["m00"] != 0: p.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
            return p
        pts_a, pts_b = get_pts(m_a), get_pts(m_b)
        val = np.mean([np.min([np.linalg.norm(pa-pb) for pb in pts_b]) for pa in pts_a]) if pts_a and pts_b else 0
        unit = "px"
        res_display = img_rgb

    # --- 結果の表示 ---
    c1, c2 = st.columns(2)
    c1.image(img_rgb, caption="Original Image")
    c2.image(res_display, caption="Analysis View")
    
    st.subheader(f"📊 Result: {val:.2f} {unit}")
    
    if st.button("この値を履歴（グラフ）に追加"):
        st.session_state.analysis_history.append({"Group": sample_group, "Value": val, "Unit": unit})
        st.success(f"Added to history: {val:.2f} {unit}")

# --- 統計グラフ ---
st.divider()
if st.session_state.analysis_history:
    df = pd.DataFrame(st.session_state.analysis_history)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Group", y="Value", ax=ax, alpha=0.6, errorbar="sd", capsize=.1)
    sns.stripplot(data=df, x="Group", y="Value", ax=ax, color=".2", jitter=True)
    ax.set_ylabel(f"Value ({df['Unit'].iloc[-1]})")
    st.pyplot(fig)
    st.dataframe(df)
