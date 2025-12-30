import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- ページ設定 ---
st.set_page_config(page_title="Professional Bio-Image Quantifier", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Professional Image Analysis Engine")
st.caption("2025年完遂仕様：解析・蓄積・有意差検定をこれ一台で完結")

# --- サイドバー：解析設定 ---
with st.sidebar:
    st.header("Analysis Parameters")
    mode = st.selectbox("解析モード:", ["陽性面積率 (IHC/DAB)", "細胞核カウント (DAPI)"])
    sample_name = st.text_input("サンプル名:", placeholder="例: Control-01")
    
    if mode == "陽性面積率 (IHC/DAB)":
        threshold_val = st.slider("二値化しきい値", 0, 255, 120)
    else:
        min_size = st.slider("最小細胞サイズ", 10, 1000, 100)

    if st.button("履歴をすべてリセット"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メイン：解析セクション ---
uploaded_file = st.file_uploader("解析する画像をアップロード...", type=["jpg", "png", "tif"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    
    # 解析実行
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_val = 0
    unit = ""
    
    if mode == "陽性面積率 (IHC/DAB)":
        _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
        result_val = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100
        unit = "%"
        display_img = mask
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = [c for c in contours if cv2.contourArea(c) > min_size]
        res_img = img_rgb.copy()
        cv2.drawContours(res_img, valid_cnts, -1, (0, 255, 0), 2)
        result_val = len(valid_cnts)
        unit = "cells"
        display_img = res_img

    with col1:
        st.image(img_rgb, caption="Original", use_container_width=True)
    with col2:
        st.image(display_img, caption="Detection Result", use_container_width=True)

    st.metric(f"Current Result ({mode})", f"{result_val:.2f} {unit}")
    
    if st.button("このデータを履歴に追加してグラフ化"):
        name = sample_name if sample_name else f"Sample_{len(st.session_state.analysis_history)+1}"
        st.session_state.analysis_history.append({"Sample": name, "Value": result_val})
        st.success(f"Added: {name}")

# --- 統計・グラフセクション ---
st.divider()
if st.session_state.analysis_history:
    df = pd.DataFrame(st.session_state.analysis_history)
    
    st.subheader("📊 Statistical Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    
    # 棒グラフ + ドットプロット
    sns.barplot(data=df, x="Sample", y="Value", ax=ax, palette="Blues_d", alpha=0.7)
    sns.stripplot(data=df, x="Sample", y="Value", ax=ax, color=".3", size=8)
    
    ax.set_ylabel(f"Value ({unit})")
    sns.despine()
    
    # 簡易有意差検定 (2群以上ある場合)
    groups = df["Sample"].unique()
    if len(groups) >= 2:
        g1 = df[df["Sample"] == groups[0]]["Value"]
        g2 = df[df["Sample"] == groups[1]]["Value"]
        if len(g1) > 1 and len(g2) > 1:
            _, p = stats.ttest_ind(g1, g2)
            st.write(f"**Statistical Note:** Comparing {groups[0]} and {groups[1]}, p-value = {p:.4f}")

    st.pyplot(fig)
    st.dataframe(df)
