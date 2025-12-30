import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- ページ設定 ---
st.set_page_config(page_title="Bio-Image Quantifier Ultimate", layout="wide")

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
        target_color = st.radio("ターゲット色:", ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI/Hoechst)"])
        sensitivity = st.slider("色抽出感度", 10, 100, 40)
    elif mode == "2. 細胞核カウント (Count)":
        min_size = st.slider("最小細胞サイズ (px)", 10, 500, 50)
    elif mode == "3. 共局在解析 (Colocalization)":
        st.info("緑(Green)と赤(Red)の重なりを解析")
        sens_g = st.slider("Green感度", 10, 100, 40)
        sens_r = st.slider("Red感度", 10, 100, 40)
    elif mode == "4. 空間距離解析 (Spatial Distance)":
        color_b_name = st.radio("群B（ターゲット）の色:", ["緑 (Green)", "青 (Blue/DAPI)"])
        st.caption("※群Aは「赤 (Red)」固定で解析します")

    if st.button("履歴をすべて削除"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メイン：画像アップロード ---
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "tif"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    val = 0.0
    unit = ""
    res_display = None

    # --- 解析実行 ---
    if mode == "1. 多重染色分離/面積 (Area)":
        if target_color == "茶色 (DAB)": 
            lower, upper = np.array([10, 50, 20]), np.array([30, 255, 255])
        elif target_color == "緑 (GFP)": 
            lower, upper = np.array([35, 50, 50]), np.array([85, 255, 255])
        elif target_color == "青 (DAPI/Hoechst)": 
            lower, upper = np.array([100, 50, 50]), np.array([130, 255, 255])
        else: # 赤
            lower, upper = np.array([0, 50, 50]), np.array([10, 255, 255])
        
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
        m_a = cv2.inRange(img_hsv, np.array([0,50,50]), np.array([10,255,255])) # Red
        if color_b_name == "緑 (Green)": 
            m_b = cv2.inRange(img_hsv, np.array([35,50,50]), np.array([85,255,255]))
        else: # Blue
            m_b = cv2.inRange(img_hsv, np.array([100,50,50]), np.array([130,255,255]))
        
        def get_pts(m):
            c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            p = []
            for cnt in c:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                    p.append(np.array([cx, cy]))
            return p
            
        pts_a, pts_b = get_pts(m_a), get_pts(m_b)
        if pts_a and pts_b:
            val = np.mean([np.min([np.linalg.norm(pa-pb) for pb in pts_b]) for pa in pts_a])
        else:
            val = 0
        unit = "px"
        res_display = cv2.addWeighted(img_rgb, 0.7, cv2.merge([m_a, m_b, np.zeros_like(m_a)]), 0.3, 0)

    # --- 結果の表示 ---
    c1, c2 = st.columns(2)
    c1.image(img_rgb, caption="Original Image", use_container_width=True)
    c2.image(res_display, caption="Analysis Result (Mask/Detection)", use_container_width=True)
    
    st.subheader(f"📊 Result: {val:.2f} {unit}")
    
    if st.button("この値を履歴（グラフ）に追加"):
        st.session_state.analysis_history.append({"Group": sample_group, "Value": val, "Unit": unit})
        st.success(f"History updated: {sample_group} = {val:.2f} {unit}")

# --- 統計グラフ ---
st.divider()
if st.session_state.analysis_history:
    st.subheader("📈 Statistical Graph (Dot-plot + Bar)")
    df = pd.DataFrame(st.session_state.analysis_history)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("ticks")
    sns.barplot(data=df, x="Group", y="Value", ax=ax, palette="muted", alpha=0.6, errorbar="sd", capsize=.1)
    sns.stripplot(data=df, x="Group", y="Value", ax=ax, color=".2", size=8, jitter=True)
    
    current_unit = df['Unit'].iloc[-1]
    ax.set_ylabel(f"Value ({current_unit})")
    sns.despine()
    st.pyplot(fig)
    
    # 簡易有意差検定
    groups = df["Group"].unique()
    if len(groups) == 2:
        g1 = df[df["Group"] == groups[0]]["Value"]
        g2 = df[df["Group"] == groups[1]]["Value"]
        if len(g1) > 1 and len(g2) > 1:
            _, p = stats.ttest_ind(g1, g2)
            st.write(f"**T-test p-value ({groups[0]} vs {groups[1]}):** `{p:.4f}`")

    st.dataframe(df)
