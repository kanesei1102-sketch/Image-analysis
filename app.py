import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Bio-Image High-Intensity", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: High-Intensity Edition")
st.caption("2025年完遂仕様：濃い染色・強発光サンプル対応版")

# --- 色定義 ---
# 色相(H)の範囲だけを定義し、明度(V)はスライダーで動的に決める
COLOR_HUE = {
    "茶色 (DAB)": (10, 30),
    "緑 (GFP)":   (35, 85),
    "青 (DAPI)":  (100, 140),
    # 赤は特殊処理
}

def get_mask(hsv_img, color_name, sensitivity, min_brightness):
    # 感度(Sensitivity) -> 色相(H)の広さ
    # 輝度(Brightness) -> 明度(V)の下限 (これ以下は無視)
    
    # 彩度(S)の下限も少し上げて、白っぽいノイズを除く
    min_saturation = 30 
    
    if color_name == "赤 (RFP)":
        # 赤はHが0付近と180付近
        lower1 = np.array([0, min_saturation, min_brightness])
        upper1 = np.array([10 + sensitivity//2, 255, 255])
        lower2 = np.array([170 - sensitivity//2, min_saturation, min_brightness])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower1, upper1)
        mask2 = cv2.inRange(hsv_img, lower2, upper2)
        return mask1 | mask2
    else:
        h_range = COLOR_HUE[color_name]
        # 色相の範囲を感度で調整
        h_min = np.clip(h_range[0] - sensitivity, 0, 180)
        h_max = np.clip(h_range[1] + sensitivity, 0, 180)
        
        lower = np.array([h_min, min_saturation, min_brightness])
        upper = np.array([h_max, 255, 255])
        return cv2.inRange(hsv_img, lower, upper)

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
    
    st.markdown("### 🎚️ 濃さ・明るさの調整")
    st.info("※「濃すぎる」場合は、輝度しきい値を上げてください")

    if mode == "1. 単色面積率 (Area)":
        target_a = st.selectbox("解析色:", ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"])
        sens_a = st.slider("色味の広さ (感度)", 5, 50, 20, help="色相の範囲。値を大きくすると違う色も拾います")
        bright_a = st.slider("輝度しきい値 (足切り)", 0, 255, 50, help="これより暗い画素は無視します。濃い画像なら100以上に上げてみて！")
    
    elif mode == "2. 細胞核カウント (Count)":
        min_size = st.slider("最小細胞サイズ (px)", 10, 500, 50)
        bright_count = st.slider("輝度しきい値", 0, 255, 50)

    elif mode == "3. 汎用共局在解析 (Colocalization)":
        c1, c2 = st.columns(2)
        with c1:
            target_a = st.selectbox("CH-A (基準):", ["青 (DAPI)", "緑 (GFP)", "赤 (RFP)", "茶色 (DAB)"], index=0)
            sens_a = st.slider("A: 色味範囲", 5, 50, 20)
            bright_a = st.slider("A: 輝度しきい値", 0, 255, 60)
        with c2:
            target_b = st.selectbox("CH-B (対象):", ["赤 (RFP)", "緑 (GFP)", "青 (DAPI)", "茶色 (DAB)"], index=0)
            sens_b = st.slider("B: 色味範囲", 5, 50, 20)
            bright_b = st.slider("B: 輝度しきい値", 0, 255, 60)

    elif mode == "4. 空間距離解析 (Spatial Distance)":
        target_a = st.selectbox("起点色(A):", ["赤 (RFP)", "緑 (GFP)", "青 (DAPI)"], index=0)
        target_b = st.selectbox("対象色(B):", ["緑 (GFP)", "青 (DAPI)", "赤 (RFP)"], index=1)
        sens_common = st.slider("共通: 色味範囲", 5, 50, 20)
        bright_common = st.slider("共通: 輝度しきい値", 0, 255, 60)

    if st.button("履歴を削除"):
        st.session_state.analysis_history = []
        st.rerun()

# --- メインロジック ---
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

        # 1. 面積率
        if mode == "1. 単色面積率 (Area)":
            mask = get_mask(img_hsv, target_a, sens_a, bright_a)
            val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
            unit = f"% ({target_a})"
            res_display = mask

        # 2. カウント (グレースケール輝度ベース)
        elif mode == "2. 細胞核カウント (Count)":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # 輝度しきい値以下の暗い場所を0にする
            _, thresh_mask = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
            
            # その上で大津の二値化
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            final_mask = cv2.bitwise_and(thresh_mask, otsu)
            
            cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in cnts if cv2.contourArea(c) > min_size]
            val, unit = len(valid), "cells"
            cv2.drawContours(res_display, valid, -1, (0,255,0), 2)

        # 3. 共局在
        elif mode == "3. 汎用共局在解析 (Colocalization)":
            mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
            mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
            
            coloc = cv2.bitwise_and(mask_a, mask_b)
            denom = cv2.countNonZero(mask_a)
            val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
            unit = f"% ({target_b} in {target_a})"
            
            # A=緑, B=赤 表示
            res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])

        # 4. 空間距離
        elif mode == "4. 空間距離解析 (Spatial Distance)":
            mask_a = get_mask(img_hsv, target_a, sens_common, bright_common)
            mask_b = get_mask(img_hsv, target_b, sens_common, bright_common)
            pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
            if pts_a and pts_b:
                dists = []
                for pa in pts_a:
                    d = np.min([np.linalg.norm(pa - pb) for pb in pts_b])
                    dists.append(d)
                val = np.mean(dists)
            else:
                val = 0
            unit = "px"
            res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)

        # 表示 & ダウンロード
        c1, c2 = st.columns(2)
        c1.image(img_rgb, caption="Original")
        c2.image(res_display, caption="Analysis View (Brightness Filtered)")
        
        # 画像保存
        if len(res_display.shape) == 2: save_img = res_display
        else: save_img = cv2.cvtColor(res_display, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".png", save_img)
        st.download_button("📷 解析画像をダウンロード", buf.tobytes(), "result.png", "image/png")

        st.subheader(f"📊 Result: {val:.2f} {unit}")
        if st.button("履歴に追加"):
            st.session_state.analysis_history.append({"Group": sample_group, "Value": val, "Unit": unit})

# --- グラフ ---
st.divider()
if st.session_state.analysis_history:
    df = pd.DataFrame(st.session_state.analysis_history)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="Group", y="Value", ax=ax, palette="muted", alpha=0.6, errorbar="sd", capsize=.1)
    sns.stripplot(data=df, x="Group", y="Value", ax=ax, color=".2", size=8, jitter=True)
    ax.set_ylabel(f"Value ({df['Unit'].iloc[-1]})")
    st.pyplot(fig)
    st.dataframe(df)
