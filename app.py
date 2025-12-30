import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Bio-Image Quantifier Ultimate", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Ultimate Edition")
st.caption("2025年完遂仕様：数値順ソート対応・統合グラフ機能")

# --- 色定義 ---
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

# --- 関数群 ---
def get_mask(hsv_img, color_name, sens, bright_min):
    min_saturation = 30
    if color_name == "赤 (RFP)":
        lower1 = np.array([0, min_saturation, bright_min])
        upper1 = np.array([10 + sens//2, 255, 255])
        lower2 = np.array([170 - sens//2, min_saturation, bright_min])
        upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    else:
        conf = COLOR_MAP[color_name]
        l = np.clip(conf["lower"] - sens, 0, 255)
        u = np.clip(conf["upper"] + sens, 0, 255)
        l[2] = max(l[2], bright_min)
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
    
    mode = st.selectbox("解析モードを選択:", [
        "1. 単色面積率 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 汎用共局在解析 (Colocalization)",
        "4. 汎用空間距離解析 (Spatial Distance)",
        "5. 割合トレンド解析 (Ratio Analysis) ★"
    ])
    
    st.divider()

    # --- モード5（数値順グラフ） ---
    if mode == "5. 割合トレンド解析 (Ratio Analysis) ★":
        st.markdown("### 🔢 割合・濃度ごとの比較")
        trend_metric = st.radio("測定対象:", ["共局在率 (Colocalization)", "面積率 (Area)"])
        
        # X軸となる数値入力
        ratio_val = st.number_input("条件の数値 (割合/濃度):", value=0, step=10, help="この数値順にグラフが並びます")
        ratio_unit = st.text_input("単位:", value="%", placeholder="%, µM")
        
        sample_group = f"{ratio_val}{ratio_unit}"
        st.info(f"データラベル: **{sample_group}**")
        
        st.divider()
        st.markdown("#### 解析パラメータ")
        if trend_metric == "共局在率 (Colocalization)":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", list(COLOR_MAP.keys()), index=3) 
                sens_a = st.slider("A感度", 5, 50, 20, key="t_s_a")
                bright_a = st.slider("A輝度", 0, 255, 60, key="t_b_a")
            with c2:
                target_b = st.selectbox("CH-B (対象):", list(COLOR_MAP.keys()), index=2) 
                sens_b = st.slider("B感度", 5, 50, 20, key="t_s_b")
                bright_b = st.slider("B輝度", 0, 255, 60, key="t_b_b")
        else: 
            target_a = st.selectbox("解析色:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("感度", 5, 50, 20, key="t_s_a")
            bright_a = st.slider("輝度", 0, 255, 60, key="t_b_a")

    # --- 通常モード ---
    else:
        sample_group = st.text_input("グループ名 (X軸):", value="Control")
        st.divider()
        if mode == "1. 単色面積率 (Area)":
            target_a = st.selectbox("解析色:", list(COLOR_MAP.keys()))
            sens_a = st.slider("感度", 5, 50, 20)
            bright_a = st.slider("輝度", 0, 255, 60)
        elif mode == "2. 細胞核カウント (Count)":
            min_size = st.slider("最小サイズ", 10, 500, 50)
            bright_count = st.slider("輝度", 0, 255, 50)
        elif mode == "3. 汎用共局在解析 (Colocalization)":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("A感度", 5, 50, 20)
                bright_a = st.slider("A輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B (対象):", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("B感度", 5, 50, 20)
                bright_b = st.slider("B輝度", 0, 255, 60)
        elif mode == "4. 汎用空間距離解析 (Spatial Distance)":
            target_a = st.selectbox("起点A:", list(COLOR_MAP.keys()), index=2)
            target_b = st.selectbox("対象B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("色感度", 5, 50, 20)
            bright_common = st.slider("輝度", 0, 255, 60)

    if st.button("履歴・グラフを全消去"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メイン解析 ---
uploaded_file = st.file_uploader("画像をアップロード...", type=["jpg", "png", "tif"])

if uploaded_file:
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        val, unit = 0.0, ""
        res_display = img_rgb.copy()

        # 解析ロジック (共通化)
        if mode.startswith("1.") or (mode.startswith("5.") and trend_metric == "面積率 (Area)"):
            mask = get_mask(img_hsv, target_a, sens_a, bright_a)
            val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
            unit = f"% Area ({target_a})"
            res_display = mask

        elif mode.startswith("2."):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            final = cv2.bitwise_and(th, otsu)
            cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in cnts if cv2.contourArea(c) > min_size]
            val, unit = len(valid), "cells"
            cv2.drawContours(res_display, valid, -1, (0,255,0), 2)

        elif mode.startswith("3.") or (mode.startswith("5.") and trend_metric == "共局在率 (Colocalization)"):
            mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
            mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
            coloc = cv2.bitwise_and(mask_a, mask_b)
            denom = cv2.countNonZero(mask_a)
            val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
            unit = f"% Coloc ({target_b} in {target_a})"
            res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])

        elif mode.startswith("4."):
            mask_a = get_mask(img_hsv, target_a, sens_common, bright_common)
            mask_b = get_mask(img_hsv, target_b, sens_common, bright_common)
            pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
            if pts_a and pts_b:
                val = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
            else: val = 0
            unit = "px Distance"
            res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)

        # 表示
        c1, c2 = st.columns(2)
        c1.image(img_rgb, caption="Original", use_container_width=True)
        c2.image(res_display, caption="Analysis View", use_container_width=True)
        
        # 保存ボタン
        if len(res_display.shape) == 2: save_img = res_display
        else: save_img = cv2.cvtColor(res_display, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", save_img)
        st.download_button("📷 解析画像をダウンロード", buf.tobytes(), "result.png", "image/png")

        st.subheader(f"📊 Result: {val:.2f} {unit}")
        
        if st.button("グラフに追加"):
            entry = {"Group": sample_group, "Value": val, "Unit": unit}
            if mode.startswith("5."):
                entry["Is_Trend"] = True
                entry["Ratio_Value"] = ratio_val
            else:
                entry["Is_Trend"] = False
                entry["Ratio_Value"] = 0
            
            st.session_state.analysis_history.append(entry)
            st.success(f"Added: {sample_group} = {val:.2f}")

# --- グラフ描画 ---
st.divider()
st.header("📈 Analysis Report")

if st.session_state.analysis_history:
    df = pd.DataFrame(st.session_state.analysis_history)
    has_trend = df["Is_Trend"].any()
    
    if has_trend:
        # 数値でソートするが、描画は「棒グラフ」をメインにする
        df_trend = df[df["Is_Trend"] == True].sort_values(by="Ratio_Value")
        
        st.markdown("### 📊 割合比較 (Sorted Bar Plot)")
        # タブの順序を逆にしました：棒グラフが先
        tab1, tab2 = st.tabs(["棒グラフ (Bar)", "散布図 (Scatter)"])
        
        with tab1:
            # 独立した条件として比較する棒グラフ
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=df_trend, x="Group", y="Value", ax=ax, 
                        palette="viridis", capsize=.1, errorbar="sd")
            sns.stripplot(data=df_trend, x="Group", y="Value", ax=ax, 
                          color=".2", size=8, jitter=True)
            ax.set_ylabel(df_trend['Unit'].iloc[0])
            st.pyplot(fig)
            
        with tab2:
            # 相関を見たい場合の散布図（線は引かない）
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df_trend, x="Ratio_Value", y="Value", ax=ax, 
                            color="crimson", s=100)
            ax.set_xlabel("Ratio Value")
            ax.set_ylabel(df_trend['Unit'].iloc[0])
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
    else:
        # 通常モード
        st.markdown("### 📊 通常比較 (Bar Plot)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="Group", y="Value", ax=ax, palette="muted", capsize=.1)
        sns.stripplot(data=df, x="Group", y="Value", ax=ax, color=".2", jitter=True)
        ax.set_ylabel(df['Unit'].iloc[-1])
        st.pyplot(fig)

    st.dataframe(df)
    st.download_button("CSV保存", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
