import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime  # JST日時取得用

# グラフ描画ライブラリ（altair, matplotlib, seaborn）のインポートを削除

st.set_page_config(page_title="Bio-Image Quantifier Pro (Extraction Only)", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Pro Edition (Extraction)")
st.caption("2025年最終版：解析・データ抽出専用（グラフ機能なし）")

# --- 色定義 ---
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

# --- 関数群 ---
def get_mask(hsv_img, color_name, sens, bright_min):
    if color_name == "赤 (RFP)":
        lower1 = np.array([0, 30, bright_min])
        upper1 = np.array([10 + sens//2, 255, 255])
        lower2 = np.array([170 - sens//2, 30, bright_min])
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
        "5. 割合トレンド解析 (Ratio Analysis)"
    ])
    st.divider()

    if mode == "5. 割合トレンド解析 (Ratio Analysis)":
        st.markdown("### 🔢 条件設定 (Batch)")
        trend_metric = st.radio("測定対象:", ["共局在率 (Colocalization)", "面積率 (Area)"])
        ratio_val = st.number_input("今回の数値条件 (割合/濃度):", value=0, step=10)
        ratio_unit = st.text_input("単位:", value="%", key="unit")
        sample_group = f"{ratio_val}{ratio_unit}"
        st.info(f"ラベル: **{sample_group}**")
        st.divider()
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
    else:
        sample_group = st.text_input("グループ名 (X軸):", value="Control")
        st.divider()
        if mode == "1. 単色面積率 (Area)":
            target_a = st.selectbox("解析色:", list(COLOR_MAP.keys()))
            sens_a = st.slider("感度", 5, 50, 20)
            bright_a = st.slider("輝度", 0, 255, 60)
        elif mode == "2. 細胞核カウント (Count)":
            min_size = st.slider("最小サイズ(px)", 10, 500, 50)
            bright_count = st.slider("輝度しきい値", 0, 255, 50)
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

    if st.button("履歴を全消去"):
        st.session_state.analysis_history = []
        st.rerun()
        # --- サイドバーの最後の方に追加 ---
    with st.sidebar:
        st.divider()
        st.caption("【免責事項 / Disclaimer】")
        st.caption("""
        本ツールは画像解析の補助を目的としています。
        照明条件や設定により結果が変動するため、最終的な解釈および結論については、
        利用者が専門的知見に基づいて判断してください。
    
        This tool is for assistive purposes. Final interpretations should be 
        made by the user based on professional expertise.
        """)

# --- メインエリア ---
uploaded_files = st.file_uploader("画像をまとめてアップロード", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} 枚の画像を解析中...")
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
            
            # --- 解析ロジック ---
            if mode == "1. 単色面積率 (Area)" or (mode.startswith("5.") and trend_metric == "面積率 (Area)"):
                mask = get_mask(img_hsv, target_a, sens_a, bright_a)
                val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                unit = f"% Area"
                res_display = mask
            elif mode == "2. 細胞核カウント (Count)":
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
                mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
                mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                coloc = cv2.bitwise_and(mask_a, mask_b)
                denom = cv2.countNonZero(mask_a)
                val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                unit = f"% Coloc"
                res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
            elif mode == "4. 汎用空間距離解析 (Spatial Distance)":
                mask_a = get_mask(img_hsv, target_a, sens_common, bright_common)
                mask_b = get_mask(img_hsv, target_b, sens_common, bright_common)
                pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                if pts_a and pts_b:
                    val = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                else: val = 0
                unit = "px Dist"
                res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)
            
            # --- 0未満防止 ---
            val = max(0.0, val)

            entry = {
                "Group": sample_group,
                "Value": val,
                "Unit": unit,
                "Is_Trend": mode.startswith("5."),
                "Ratio_Value": ratio_val if mode.startswith("5.") else 0
            }
            batch_results.append(entry)
            
            # --- Expanderのタイトルを固定 ---
            with st.expander(f"📷 Image {i+1}: {file.name}", expanded=True):
                st.markdown(f"### Result: **{val:.2f} {unit}**")
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original", use_container_width=True)
                c2.image(res_display, caption="Analyzed", use_container_width=True)

    if st.button(f"データ {len(batch_results)} 件を追加", type="primary"):
        st.session_state.analysis_history.extend(batch_results)
        st.rerun()

# --- レポート表示セクション (データのみ) ---
if st.session_state.analysis_history:
    st.divider()
    st.header("💾 Data Export")
    
    df = pd.DataFrame(st.session_state.analysis_history)
    df["Value"] = df["Value"].clip(lower=0) 

    # 日本時間(JST)のファイル名を生成
    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    file_name = f"quantified_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"

    st.dataframe(df, use_container_width=True)
    st.download_button("📥 CSVデータを保存", df.to_csv(index=False).encode('utf-8'), file_name, "text/csv")
