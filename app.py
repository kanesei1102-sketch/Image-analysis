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
st.caption("2025年最終版：N数蓄積・統合グラフ生成機能搭載")

# --- 定数・初期設定 ---
DEFAULT_HUE = {
    "Red_Low": (0, 10), "Red_High": (170, 180),
    "Green": (35, 85), "Blue": (100, 140), "Brown": (10, 30)
}
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

# --- サイドバー設定 ---
with st.sidebar:
    st.header("Analysis Recipe")
    
    with st.expander("🎨 色の定義を微調整 (Calibration)", expanded=False):
        h_red_l = st.slider("赤(低)範囲", 0, 30, DEFAULT_HUE["Red_Low"], key="h_r_l")
        h_red_h = st.slider("赤(高)範囲", 150, 180, DEFAULT_HUE["Red_High"], key="h_r_h")
        h_green = st.slider("緑(GFP)範囲", 20, 100, DEFAULT_HUE["Green"], key="h_g")
        h_blue = st.slider("青(DAPI)範囲", 80, 160, DEFAULT_HUE["Blue"], key="h_b")
        h_brown = st.slider("茶(DAB)範囲", 0, 50, DEFAULT_HUE["Brown"], key="h_br")

    mode = st.selectbox("解析モード:", [
        "1. 単色面積率 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 汎用共局在解析 (Colocalization)",
        "4. 汎用空間距離解析 (Spatial Distance)",
        "5. 割合トレンド解析 (Ratio Analysis)"
    ])
    
    st.divider()

    # --- 条件設定 (ここが重要) ---
    # N数を増やすためには、同じ名前（ラベル）で保存する必要があります
    if mode == "5. 割合トレンド解析 (Ratio Analysis)":
        st.markdown("### 🔢 条件設定 (N数追加用)")
        trend_metric = st.radio("測定対象:", ["共局在率", "面積率"])
        
        # ソート用数値と表示ラベル
        ratio_val = st.number_input("ソート用数値 (割合):", value=0, step=10, help="グラフのX軸の並び順用")
        ratio_label = st.text_input("条件ラベル (例: 160:40):", value=f"{ratio_val}%")
        
        st.info(f"この解析結果は **「{ratio_label}」** グループに蓄積されます。")
        sample_group = ratio_label 
        
        st.divider()
        colors = ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"]
        if trend_metric == "共局在率":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", colors, index=3) 
                sens_a = st.slider("A感度", 5, 50, 20, key="tsa")
                bright_a = st.slider("A輝度", 0, 255, 60, key="tba")
            with c2:
                target_b = st.selectbox("CH-B (対象):", colors, index=2) 
                sens_b = st.slider("B感度", 5, 50, 20, key="tsb")
                bright_b = st.slider("B輝度", 0, 255, 60, key="tbb")
        else: 
            target_a = st.selectbox("解析色:", colors, index=2)
            sens_a = st.slider("感度", 5, 50, 20, key="tsa")
            bright_a = st.slider("輝度", 0, 255, 60, key="tba")
    else:
        sample_group = st.text_input("グループ名 (例: Control):", value="Control")
        st.info(f"この解析結果は **「{sample_group}」** グループに蓄積されます。")
        st.divider()
        # (他モードの設定省略なし)
        colors = ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"]
        if mode == "1. 単色面積率 (Area)":
            target_a = st.selectbox("解析色:", colors)
            sens_a = st.slider("感度", 5, 50, 20)
            bright_a = st.slider("輝度", 0, 255, 60)
        elif mode == "2. 細胞核カウント (Count)":
            min_size = st.slider("最小サイズ", 10, 500, 50)
            bright_count = st.slider("輝度", 0, 255, 50)
        elif mode == "3. 汎用共局在解析 (Colocalization)":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", colors, index=3)
                sens_a = st.slider("A感度", 5, 50, 20)
                bright_a = st.slider("A輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B (対象):", colors, index=2)
                sens_b = st.slider("B感度", 5, 50, 20)
                bright_b = st.slider("B輝度", 0, 255, 60)
        elif mode == "4. 汎用空間距離解析 (Spatial Distance)":
            target_a = st.selectbox("起点A:", colors, index=2)
            target_b = st.selectbox("対象B:", colors, index=3)
            sens_common = st.slider("色感度", 5, 50, 20)
            bright_common = st.slider("輝度", 0, 255, 60)

    st.divider()
    if st.button("履歴・グラフを全消去"):
        st.session_state.analysis_history = []
        st.rerun()

# --- 関数定義 ---
def get_mask_dynamic(hsv_img, color_name, sens, bright_min):
    min_saturation = max(0, 50 - sens)
    if color_name == "赤 (RFP)":
        l1, h1 = h_red_l; l2, h2 = h_red_h
        return cv2.inRange(hsv_img, np.array([l1, min_saturation, bright_min]), np.array([h1, 255, 255])) | \
               cv2.inRange(hsv_img, np.array([l2, min_saturation, bright_min]), np.array([h2, 255, 255]))
    elif color_name == "緑 (GFP)":
        l, h = h_green
        return cv2.inRange(hsv_img, np.array([l, min_saturation, bright_min]), np.array([h, 255, 255]))
    elif color_name == "青 (DAPI)":
        l, h = h_blue
        return cv2.inRange(hsv_img, np.array([l, min_saturation, bright_min]), np.array([h, 255, 255]))
    elif color_name == "茶色 (DAB)":
        l, h = h_brown
        return cv2.inRange(hsv_img, np.array([l, min_saturation, bright_min]), np.array([h, 255, 255]))
    return np.zeros(hsv_img.shape[:2], dtype=np.uint8)

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0: pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# --- メインエリア ---
uploaded_files = st.file_uploader("画像をまとめてアップロード (N数追加)", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} 枚の画像を受信。解析中...")
    batch_results = []
    
    for i, file in enumerate(uploaded_files):
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            val, unit, res_display = 0.0, "", img_rgb.copy()
            
            # 解析ロジック
            if mode == "1. 単色面積率 (Area)" or (mode.startswith("5.") and trend_metric == "面積率"):
                mask = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                unit = "% Area"
                res_display = mask
            elif mode == "2. 細胞核カウント (Count)":
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                _, otsu = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                final = cv2.bitwise_and(th, otsu)
                cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                val, unit = len(valid), "cells"
                cv2.drawContours(res_display, valid, -1, (0,255,0), 2)
            elif mode == "3. 汎用共局在解析 (Colocalization)" or (mode.startswith("5.") and trend_metric == "共局在率"):
                mask_a = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                mask_b = get_mask_dynamic(img_hsv, target_b, sens_b, bright_b)
                coloc = cv2.bitwise_and(mask_a, mask_b)
                denom = cv2.countNonZero(mask_a)
                val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                unit = "% Coloc"
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
            
            # 結果エントリ
            batch_results.append({
                "Group": sample_group, "Value": val, "Unit": unit,
                "Is_Trend": mode.startswith("5."), "Ratio_Value": ratio_val if mode.startswith("5.") else 0
            })
            
            # 画像確認
            with st.expander(f"📷 Img {i+1}: {val:.2f} {unit}", expanded=True):
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original", use_container_width=True)
                c2.image(res_display, caption="Result", use_container_width=True)

    st.divider()
    if st.button(f"これら {len(batch_results)} 件を「{sample_group}」データとして統合", type="primary"):
        st.session_state.analysis_history.extend(batch_results)
        st.success(f"✅ {sample_group} にデータを追加しました！(現在のN数を確認してください)")

# --- 統合グラフ描画 ---
if st.session_state.analysis_history:
    st.divider()
    st.header("📈 Integrated Report (Mean ± SD)")
    
    df = pd.DataFrame(st.session_state.analysis_history)
    
    # トレンド解析なら数値順にソート
    if df["Is_Trend"].any():
        df = df.sort_values(by="Ratio_Value")
    
    # --- データを集計して表示 ---
    # ここがポイント：seabornが自動で同じGroupのデータをまとめて、平均値バーとエラーバーを出してくれます
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("white")
    teal_color = "#005b8e"
    
    # 1. 平均値の棒グラフ (N>=2なら自動でエラーバーが付く)
    sns.barplot(
        data=df, x="Group", y="Value", ax=ax, 
        color=teal_color, capsize=.1, errorbar="sd", # 標準偏差のエラーバーを表示
        alpha=0.8
    )
    
    # 2. 個々のデータを黒い点で打つ (N数が可視化される)
    sns.stripplot(
        data=df, x="Group", y="Value", ax=ax, 
        color="black", size=6, jitter=True, alpha=0.7
    )

    y_label = df['Unit'].iloc[0]
    if "%" in y_label: y_label = "Positive rate [%]"
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    
    sns.despine()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)

    st.pyplot(fig)

    # 集計データの表示
    st.markdown("### 📊 統計データ")
    summary = df.groupby("Group")["Value"].agg(['count', 'mean', 'std']).reset_index()
    summary.columns = ["Condition", "N", "Mean", "SD"]
    st.dataframe(summary)
    
    with st.expander("全生データを見る"):
        st.dataframe(df)
        st.download_button("CSVダウンロード", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
