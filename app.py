import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bio-Image Color Calibrator", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Color Calibrator")
st.caption("2025年最終版：色認識ズレを修正するキャリブレーション機能搭載")

# --- 初期設定値 (HSVの色相 H: 0-180) ---
# 緑が赤に誤認識される場合、赤の範囲を狭めるか、緑の範囲を広げる必要がある
DEFAULT_HUE = {
    "Red_Low": (0, 10),      # 赤の低域
    "Red_High": (170, 180),  # 赤の高域（折り返し）
    "Green": (35, 85),       # 緑
    "Blue": (100, 140),      # 青
    "Brown": (10, 30)        # 茶
}

# --- サイドバー ---
with st.sidebar:
    st.header("Analysis Recipe")
    
    # ★ここが新機能：色の定義をユーザーがいじれるようにする
    with st.expander("🎨 色の定義を微調整 (Calibration)", expanded=False):
        st.write("※「緑が赤に認識される」等のズレがある場合、ここを調整してください。")
        
        h_red_l = st.slider("赤(低)の色相範囲", 0, 30, DEFAULT_HUE["Red_Low"], help="通常 0-10。広げすぎると茶色や黄色を拾います。")
        h_red_h = st.slider("赤(高)の色相範囲", 150, 180, DEFAULT_HUE["Red_High"], help="通常 170-180。")
        h_green = st.slider("緑(GFP)の色相範囲", 20, 100, DEFAULT_HUE["Green"], help="通常 35-85。黄色っぽい緑なら左(25〜)へ広げてください。")
        h_blue = st.slider("青(DAPI)の色相範囲", 80, 160, DEFAULT_HUE["Blue"], help="通常 100-140。")
        h_brown = st.slider("茶(DAB)の色相範囲", 0, 50, DEFAULT_HUE["Brown"], help="通常 10-30。")

    mode = st.selectbox("解析モード:", [
        "1. 単色面積率 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 汎用共局在解析 (Colocalization)",
        "4. 汎用空間距離解析 (Spatial Distance)",
        "5. 割合トレンド解析 (Ratio Analysis)"
    ])
    
    st.divider()

    # --- モード設定 (バッチ対応) ---
    if mode == "5. 割合トレンド解析 (Ratio Analysis)":
        trend_metric = st.radio("測定対象:", ["共局在率 (Colocalization)", "面積率 (Area)"])
        ratio_val = st.number_input("数値条件 (割合/濃度):", value=0, step=10)
        ratio_unit = st.text_input("単位:", value="%", key="unit")
        sample_group = f"{ratio_val}{ratio_unit}"
        
        st.markdown("#### 解析パラメータ")
        # ターゲット選択肢
        colors = ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"]
        
        if trend_metric == "共局在率 (Colocalization)":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", colors, index=3) 
                sens_a = st.slider("A感度(彩度)", 5, 50, 20, key="t_s_a")
                bright_a = st.slider("A輝度", 0, 255, 60, key="t_b_a")
            with c2:
                target_b = st.selectbox("CH-B (対象):", colors, index=2) 
                sens_b = st.slider("B感度(彩度)", 5, 50, 20, key="t_s_b")
                bright_b = st.slider("B輝度", 0, 255, 60, key="t_b_b")
        else: # 面積
            target_a = st.selectbox("解析色:", colors, index=2)
            sens_a = st.slider("感度(彩度)", 5, 50, 20, key="t_s_a")
            bright_a = st.slider("輝度", 0, 255, 60, key="t_b_a")

    else:
        sample_group = st.text_input("グループ名 (X軸):", value="Control")
        colors = ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"]
        
        if mode == "1. 単色面積率 (Area)":
            target_a = st.selectbox("解析色:", colors)
            sens_a = st.slider("感度(彩度)", 5, 50, 20)
            bright_a = st.slider("輝度", 0, 255, 60)
        elif mode == "2. 細胞核カウント (Count)":
            min_size = st.slider("最小サイズ(px)", 10, 500, 50)
            bright_count = st.slider("輝度しきい値", 0, 255, 50)
        elif mode == "3. 汎用共局在解析 (Colocalization)":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", colors, index=3)
                sens_a = st.slider("A感度(彩度)", 5, 50, 20)
                bright_a = st.slider("A輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B (対象):", colors, index=2)
                sens_b = st.slider("B感度(彩度)", 5, 50, 20)
                bright_b = st.slider("B輝度", 0, 255, 60)
        elif mode == "4. 汎用空間距離解析 (Spatial Distance)":
            target_a = st.selectbox("起点A:", colors, index=2)
            target_b = st.selectbox("対象B:", colors, index=3)
            sens_common = st.slider("色感度", 5, 50, 20)
            bright_common = st.slider("輝度", 0, 255, 60)

    if st.button("履歴クリア"):
        st.session_state.analysis_history = []
        st.rerun()

# --- 関数: 動的なマスク生成 ---
def get_mask_dynamic(hsv_img, color_name, sens, bright_min):
    # サイドバーで設定された値を使う
    # sens(感度)はここでは「彩度(Saturation)の許容範囲」として使う
    
    min_saturation = max(0, 50 - sens) # 感度が高い＝彩度が低くても拾う
    
    if color_name == "赤 (RFP)":
        # ユーザー設定値を適用
        l1, h1 = h_red_l
        l2, h2 = h_red_h
        
        lower1 = np.array([l1, min_saturation, bright_min])
        upper1 = np.array([h1, 255, 255])
        lower2 = np.array([l2, min_saturation, bright_min])
        upper2 = np.array([h2, 255, 255])
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    
    elif color_name == "緑 (GFP)":
        l, h = h_green
        lower = np.array([l, min_saturation, bright_min])
        upper = np.array([h, 255, 255])
        return cv2.inRange(hsv_img, lower, upper)
        
    elif color_name == "青 (DAPI)":
        l, h = h_blue
        lower = np.array([l, min_saturation, bright_min])
        upper = np.array([h, 255, 255])
        return cv2.inRange(hsv_img, lower, upper)

    elif color_name == "茶色 (DAB)":
        l, h = h_brown
        lower = np.array([l, min_saturation, bright_min])
        upper = np.array([h, 255, 255])
        return cv2.inRange(hsv_img, lower, upper)
    
    return np.zeros(hsv_img.shape[:2], dtype=np.uint8)

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# --- メイン処理 ---
uploaded_files = st.file_uploader("画像をまとめてアップロード", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} 枚読み込み中...")
    batch_results = []
    
    for i, file in enumerate(uploaded_files):
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            val, unit = 0.0, ""
            res_display = img_rgb.copy()
            
            # --- 解析 (Dynamic Mask使用) ---
            if mode == "1. 単色面積率 (Area)" or (mode.startswith("5.") and trend_metric == "面積率 (Area)"):
                mask = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                unit = f"% Area"
                res_display = mask

            elif mode == "2. 細胞核カウント (Count)":
                # Countモードはグレースケールベースなので色定義は関係なし
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                final = cv2.bitwise_and(th, otsu)
                cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                val, unit = len(valid), "cells"
                cv2.drawContours(res_display, valid, -1, (0,255,0), 2)

            elif mode == "3. 汎用共局在解析 (Colocalization)" or (mode.startswith("5.") and trend_metric == "共局在率 (Colocalization)"):
                mask_a = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                mask_b = get_mask_dynamic(img_hsv, target_b, sens_b, bright_b)
                coloc = cv2.bitwise_and(mask_a, mask_b)
                denom = cv2.countNonZero(mask_a)
                val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                unit = f"% Coloc"
                res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])

            elif mode == "4. 汎用空間距離解析 (Spatial Distance)":
                mask_a = get_mask_dynamic(img_hsv, target_a, sens_common, bright_common)
                mask_b = get_mask_dynamic(img_hsv, target_b, sens_common, bright_common)
                pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                if pts_a and pts_b:
                    val = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                else: val = 0
                unit = "px Dist"
                res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)
            
            entry = {
                "Group": sample_group, "Value": val, "Unit": unit,
                "Is_Trend": mode.startswith("5."), "Ratio_Value": ratio_val if mode.startswith("5.") else 0
            }
            batch_results.append(entry)
            
            # --- 結果表示 (確認用) ---
            header_text = f"📷 Img {i+1}: {file.name} | Result: {val:.2f} {unit}"
            with st.expander(header_text, expanded=True):
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original", use_container_width=True)
                c2.image(res_display, caption="Analysis Result (Check Colors Here)", use_container_width=True)

    st.divider()
    if st.button(f"これら {len(batch_results)} 件をグラフに追加", type="primary"):
        st.session_state.analysis_history.extend(batch_results)
        st.success("✅ 追加しました！")

# --- グラフ ---
if st.session_state.analysis_history:
    st.divider()
    st.header("📈 Analysis Report")
    df = pd.DataFrame(st.session_state.analysis_history)
    has_trend = df["Is_Trend"].any()
    
    if has_trend:
        df_trend = df[df["Is_Trend"] == True].sort_values(by="Ratio_Value")
        tab1, tab2 = st.tabs(["棒グラフ (Bar)", "散布図 (Scatter)"])
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=df_trend, x="Group", y="Value", ax=ax, palette="viridis", capsize=.1)
            sns.stripplot(data=df_trend, x="Group", y="Value", ax=ax, color=".2", jitter=True)
            ax.set_ylabel(df_trend['Unit'].iloc[0])
            st.pyplot(fig)
        with tab2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df_trend, x="Ratio_Value", y="Value", ax=ax, color="crimson", s=100)
            ax.set_xlabel("Ratio Value")
            ax.set_ylabel(df_trend['Unit'].iloc[0])
            st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="Group", y="Value", ax=ax, palette="muted", capsize=.1)
        sns.stripplot(data=df, x="Group", y="Value", ax=ax, color=".2", jitter=True)
        ax.set_ylabel(df['Unit'].iloc[-1])
        st.pyplot(fig)

    st.dataframe(df)
    st.download_button("CSV保存", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
