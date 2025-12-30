import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bio-Image Quantifier Pro", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Pro Edition")
st.caption("2025年最終版：一括解析・N数統合・グラフ生成（Teal Blue）")

# 定数・初期設定
DEFAULT_HUE = {
    "Red_Low": (0, 10), "Red_High": (170, 180),
    "Green": (35, 85), "Blue": (95, 145), "Brown": (10, 30)
}
COLORS = ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"]

with st.sidebar:
    st.header("Analysis Recipe")
    with st.expander("🎨 色の定義を微調整 (Calibration)", expanded=False):
        h_red_l = st.slider("赤(低)範囲", 0, 30, DEFAULT_HUE["Red_Low"], key="h_r_l")
        h_red_h = st.slider("赤(高)範囲", 150, 180, DEFAULT_HUE["Red_High"], key="h_r_h")
        h_green = st.slider("緑(GFP)範囲", 20, 100, DEFAULT_HUE["Green"], key="h_g")
        h_blue = st.slider("青(DAPI)範囲", 80, 160, DEFAULT_HUE["Blue"], key="h_b")
        h_brown = st.slider("茶(DAB)範囲", 0, 50, DEFAULT_HUE["Brown"], key="h_br")

    mode = st.selectbox("解析モードを選択:", [
        "1. 単色面積率 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 汎用共局在解析 (Colocalization)",
        "4. 汎用空間距離解析 (Spatial Distance)",
        "5. 割合トレンド解析 (Ratio Analysis)"
    ])
    st.divider()

    # 変数初期化
    target_a, target_b = "青 (DAPI)", "赤 (RFP)"
    sens_a, sens_b = 20, 20
    bright_a, bright_b = 30, 60 
    sens_common, bright_common = 20, 60
    min_size, bright_count = 50, 50
    sample_group = "Control"
    ratio_val = 0
    trend_metric = ""

    if mode.startswith("5."):
        st.markdown("### 🔢 条件設定 (N数追加)")
        trend_metric = st.radio("測定対象:", ["共局在率", "面積率"])
        ratio_val = st.number_input("ソート用数値 (割合):", value=0, step=10)
        ratio_label = st.text_input("条件ラベル (例: 160:40):", value=f"{ratio_val}%")
        sample_group = ratio_label 
        st.divider()
        st.markdown("#### 解析パラメータ")
        if trend_metric == "共局在率":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", COLORS, index=3, key="m5_ta") 
                sens_a = st.slider("A感度", 5, 50, 20, key="m5_sa")
                bright_a = st.slider("A輝度", 0, 255, 30, key="m5_ba")
            with c2:
                target_b = st.selectbox("CH-B (対象):", COLORS, index=2, key="m5_tb") 
                sens_b = st.slider("B感度", 5, 50, 20, key="m5_sb")
                bright_b = st.slider("B輝度", 0, 255, 60, key="m5_bb")
        else:
            target_a = st.selectbox("解析色:", COLORS, index=2, key="m5_ta_area")
            sens_a = st.slider("感度", 5, 50, 20, key="m5_sa_area")
            bright_a = st.slider("輝度", 0, 255, 60, key="m5_ba_area")
    else:
        sample_group = st.text_input("グループ名 (例: Control):", value="Control")
        st.divider()
        if mode.startswith("1."):
            target_a = st.selectbox("解析色:", COLORS, index=2)
            sens_a = st.slider("感度", 5, 50, 20)
            bright_a = st.slider("輝度", 0, 255, 60)
        elif mode.startswith("2."):
            min_size = st.slider("最小サイズ(px)", 10, 500, 50)
            bright_count = st.slider("輝度しきい値", 0, 255, 50)
        elif mode.startswith("3."):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", COLORS, index=3)
                sens_a = st.slider("A感度", 5, 50, 20)
                bright_a = st.slider("A輝度", 0, 255, 30)
            with c2:
                target_b = st.selectbox("CH-B (対象):", COLORS, index=2)
                sens_b = st.slider("B感度", 5, 50, 20)
                bright_b = st.slider("B輝度", 0, 255, 60)
        elif mode.startswith("4."):
            target_a = st.selectbox("起点A:", COLORS, index=3)
            target_b = st.selectbox("対象B:", COLORS, index=2)
            sens_common = st.slider("色感度", 5, 50, 20)
            bright_common = st.slider("輝度", 0, 255, 60)

    st.divider()
    graph_type = st.radio("📊 グラフの種類:", ["箱ひげ図 (Box Plot)", "棒グラフ (Bar Plot)", "バイオリン図 (Violin Plot)", "ドットプロット (Strip Plot)"], index=1)
    st.divider()
    if st.button("履歴・グラフを全消去"):
        st.session_state.analysis_history = []
        st.rerun()

def get_mask_dynamic(hsv_img, color_name, sens, bright_min):
    min_saturation = max(0, 50 - sens)
    h, s, v = cv2.split(hsv_img)
    v_mask = cv2.threshold(v, bright_min, 255, cv2.THRESH_BINARY)[1]
    color_mask = np.zeros_like(v_mask)
    if color_name == "赤 (RFP)":
        l1, h1 = h_red_l; l2, h2 = h_red_h
        color_mask = cv2.inRange(hsv_img, np.array([l1, min_saturation, 0]), np.array([h1, 255, 255])) | \
                     cv2.inRange(hsv_img, np.array([l2, min_saturation, 0]), np.array([h2, 255, 255]))
    elif color_name == "緑 (GFP)":
        l, h = h_green
        color_mask = cv2.inRange(hsv_img, np.array([l, min_saturation, 0]), np.array([h, 255, 255]))
    elif color_name == "青 (DAPI)":
        l, h = h_blue
        color_mask = cv2.inRange(hsv_img, np.array([l, min_saturation, 0]), np.array([h, 255, 255]))
    elif color_name == "茶色 (DAB)":
        l, h = h_brown
        color_mask = cv2.inRange(hsv_img, np.array([l, min_saturation, 0]), np.array([h, 255, 255]))
    return cv2.bitwise_and(color_mask, v_mask)

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0: pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

uploaded_files = st.file_uploader("画像をまとめてアップロード (N数追加)", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} 枚受信。解析中...")
    batch_results = []
    for i, file in enumerate(uploaded_files):
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            val, unit, res_display = 0.0, "", img_rgb.copy()
            
            # モード判定（あいまい検索で確実にヒットさせる）
            is_area = "面積" in mode or (mode.startswith("5.") and "面積" in trend_metric)
            is_count = "カウント" in mode
            is_coloc = "共局在" in mode or (mode.startswith("5.") and "共局在" in trend_metric)
            is_dist = "距離" in mode

            if is_area:
                mask = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                unit = "% Area"
                res_display = mask
            elif is_count:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                _, otsu = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                final = cv2.bitwise_and(th, otsu)
                cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                val, unit = len(valid), "cells"
                cv2.drawContours(res_display, valid, -1, (0,255,0), 2)
            elif is_coloc:
                mask_a = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                mask_b = get_mask_dynamic(img_hsv, target_b, sens_b, bright_b)
                coloc = cv2.bitwise_and(mask_a, mask_b)
                denom = cv2.countNonZero(mask_a)
                val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                unit = "% Coloc"
                res_display = cv2.merge([np.zeros_like(mask_a), mask_a, mask_b]) # 黄色表示
            elif is_dist:
                mask_a = get_mask_dynamic(img_hsv, target_a, sens_common, bright_common)
                mask_b = get_mask_dynamic(img_hsv, target_b, sens_common, bright_common)
                pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                if pts_a and pts_b:
                    val = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                else: val = 0
                unit = "px Dist"
                res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([np.zeros_like(mask_a), mask_a, mask_b]), 0.4, 0)
            
            # 万が一 unit が空なら強制代入
            if unit == "": unit = "(No Unit)"

            batch_results.append({
                "Group": sample_group, "Value": val, "Unit": unit,
                "Is_Trend": mode.startswith("5."), "Ratio_Value": ratio_val if mode.startswith("5.") else 0
            })
            with st.expander(f"📷 Img {i+1}: {val:.2f} {unit}", expanded=True):
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original", use_container_width=True)
                c2.image(res_display, caption="Result", use_container_width=True)

    st.divider()
    if st.button(f"これら {len(batch_results)} 件を「{sample_group}」データとして統合", type="primary"):
        st.session_state.analysis_history.extend(batch_results)
        st.success("✅ データ追加完了")

if st.session_state.analysis_history:
    st.divider()
    st.header("📈 Integrated Report")
    df = pd.DataFrame(st.session_state.analysis_history)
    if df["Is_Trend"].any(): df = df.sort_values(by="Ratio_Value")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("white")
    teal_color = "#006d77" 
    
    if graph_type == "棒グラフ (Bar Plot)":
        sns.barplot(data=df, x="Group", y="Value", ax=ax, color=teal_color, capsize=.1, errorbar="sd", alpha=0.9)
    elif graph_type == "箱ひげ図 (Box Plot)":
        sns.boxplot(data=df, x="Group", y="Value", ax=ax, color=teal_color, width=0.5, fliersize=0)
    elif graph_type == "バイオリン図 (Violin Plot)":
        sns.violinplot(data=df, x="Group", y="Value", ax=ax, color=teal_color, inner="quartile")
    elif graph_type == "ドットプロット (Strip Plot)":
        sns.stripplot(data=df, x="Group", y="Value", ax=ax, size=10, color=teal_color, jitter=True)
        sns.pointplot(data=df, x="Group", y="Value", ax=ax, errorbar=None, color="firebrick", markers="_", scale=1.5, join=False)

    if graph_type in ["棒グラフ (Bar Plot)", "箱ひげ図 (Box Plot)", "バイオリン図 (Violin Plot)"]:
        sns.stripplot(data=df, x="Group", y="Value", ax=ax, color="black", size=6, jitter=True, alpha=0.7)

    y_label = df['Unit'].iloc[0]
    if "%" in y_label: y_label = "Positive rate [%]"
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    sns.despine()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    st.pyplot(fig)
    
    st.markdown("### 📊 統計サマリー")
    summary = df.groupby("Group")["Value"].agg(['count', 'mean', 'std']).reset_index()
    summary.columns = ["Condition", "N", "Mean", "SD"]
    st.dataframe(summary.style.format({"Mean": "{:.2f}", "SD": "{:.2f}"}))
    st.download_button("CSVダウンロード", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
