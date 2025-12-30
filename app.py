import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- ページ設定 ---
st.set_page_config(page_title="Bio-Image Quantifier Pro", layout="wide")

# --- セッション状態の初期化 ---
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Pro Edition")
st.caption("2025年最終版：一括解析・色補正・グラフスタイル選択機能搭載")

# --- 定数定義 ---
# 色ごとの標準的なHSV範囲（初期値）
DEFAULT_HUE = {
    "Red_Low": (0, 10), "Red_High": (170, 180),
    "Green": (35, 85), "Blue": (100, 140), "Brown": (10, 30)
}

# --- サイドバー：設定エリア ---
with st.sidebar:
    st.header("Analysis Recipe")
    
    # 1. 色の定義（キャリブレーション）
    with st.expander("🎨 色の定義を微調整 (Calibration)", expanded=False):
        st.caption("※色が正しく認識されない場合、ここを調整してください")
        h_red_l = st.slider("赤(低)範囲", 0, 30, DEFAULT_HUE["Red_Low"], key="h_r_l")
        h_red_h = st.slider("赤(高)範囲", 150, 180, DEFAULT_HUE["Red_High"], key="h_r_h")
        h_green = st.slider("緑(GFP)範囲", 20, 100, DEFAULT_HUE["Green"], key="h_g")
        h_blue = st.slider("青(DAPI)範囲", 80, 160, DEFAULT_HUE["Blue"], key="h_b")
        h_brown = st.slider("茶(DAB)範囲", 0, 50, DEFAULT_HUE["Brown"], key="h_br")

    # 2. 解析モード選択
    mode = st.selectbox("解析モードを選択:", [
        "1. 単色面積率 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 汎用共局在解析 (Colocalization)",
        "4. 汎用空間距離解析 (Spatial Distance)",
        "5. 割合トレンド解析 (Ratio Analysis)"
    ])
    
    st.divider()

    # 3. モード別詳細設定
    colors = ["茶色 (DAB)", "緑 (GFP)", "赤 (RFP)", "青 (DAPI)"]

    if mode == "5. 割合トレンド解析 (Ratio Analysis)":
        st.markdown("### 🔢 条件設定 (Batch)")
        trend_metric = st.radio("測定対象:", ["共局在率 (Colocalization)", "面積率 (Area)"])
        
        # グラフのX軸ラベル用設定
        ratio_val = st.number_input("ソート用数値 (割合):", value=0, step=10, help="グラフの並び順を決めるための数値")
        ratio_label = st.text_input("表示ラベル (例: 160:40):", value=f"{ratio_val}%")
        sample_group = ratio_label 
        
        st.divider()
        st.markdown("#### 解析パラメータ")
        if trend_metric == "共局在率 (Colocalization)":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", colors, index=3) 
                sens_a = st.slider("A感度", 5, 50, 20, key="t_s_a")
                bright_a = st.slider("A輝度", 0, 255, 60, key="t_b_a")
            with c2:
                target_b = st.selectbox("CH-B (対象):", colors, index=2) 
                sens_b = st.slider("B感度", 5, 50, 20, key="t_s_b")
                bright_b = st.slider("B輝度", 0, 255, 60, key="t_b_b")
        else: # 面積
            target_a = st.selectbox("解析色:", colors, index=2)
            sens_a = st.slider("感度", 5, 50, 20, key="t_s_a")
            bright_a = st.slider("輝度", 0, 255, 60, key="t_b_a")

    else:
        sample_group = st.text_input("グループ名 (X軸):", value="Control")
        st.divider()
        
        if mode == "1. 単色面積率 (Area)":
            target_a = st.selectbox("解析色:", colors)
            sens_a = st.slider("感度", 5, 50, 20)
            bright_a = st.slider("輝度", 0, 255, 60)
        elif mode == "2. 細胞核カウント (Count)":
            min_size = st.slider("最小サイズ(px)", 10, 500, 50)
            bright_count = st.slider("輝度しきい値", 0, 255, 50)
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

    # 4. グラフ設定
    st.divider()
    graph_type = st.radio("📊 グラフの種類:", 
                          ["棒グラフ (Bar Plot)", "箱ひげ図 (Box Plot)", "バイオリン図 (Violin Plot)", "ドットプロット (Strip Plot)"])

    # 5. リセットボタン
    st.divider()
    if st.button("履歴・グラフを全消去"):
        st.session_state.analysis_history = []
        st.rerun()

# --- 関数定義 ---

# 動的マスク生成（サイドバーの設定値を使用）
def get_mask_dynamic(hsv_img, color_name, sens, bright_min):
    # 感度が高い＝彩度(S)が低くても拾う
    min_saturation = max(0, 50 - sens)
    
    # 輝度(V)によるフィルタリング（暗いノイズを除去）
    # inRangeのマスク後にV値でANDを取るよりも、下限値を設定する方が高速
    
    if color_name == "赤 (RFP)":
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

# 重心取得（距離解析用）
def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# --- メインエリア：一括アップロード & 解析 ---
uploaded_files = st.file_uploader("画像をまとめてアップロード (複数選択可)", 
                                  type=["jpg", "png", "tif"], 
                                  accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} 枚の画像を読み込みました。解析結果を確認してください。")
    
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
                mask = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
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
                mask_a = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                mask_b = get_mask_dynamic(img_hsv, target_b, sens_b, bright_b)
                coloc = cv2.bitwise_and(mask_a, mask_b)
                denom = cv2.countNonZero(mask_a)
                val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                unit = f"% Coloc"
                # 共局在表示（黄色＝赤＋緑）
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
            
            # 結果エントリ作成
            entry = {
                "Group": sample_group, 
                "Value": val, 
                "Unit": unit,
                "Is_Trend": mode.startswith("5."), 
                "Ratio_Value": ratio_val if mode.startswith("5.") else 0
            }
            batch_results.append(entry)
            
            # 画像表示 (確認用)
            header_text = f"📷 Img {i+1}: {file.name} | Result: {val:.2f} {unit}"
            with st.expander(header_text, expanded=True):
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original Image", use_container_width=True)
                c2.image(res_display, caption="Analysis Result", use_container_width=True)

    # 全データ追加ボタン
    st.divider()
    if st.button(f"これら {len(batch_results)} 件のデータをグラフに追加", type="primary"):
        st.session_state.analysis_history.extend(batch_results)
        st.success(f"✅ {len(batch_results)} 件のデータを追加しました！")

# --- グラフ描画セクション ---
if st.session_state.analysis_history:
    st.divider()
    st.header("📈 Analysis Report")
    
    df = pd.DataFrame(st.session_state.analysis_history)
    
    # トレンド解析モードのデータが含まれている場合、数値順にソート
    if df["Is_Trend"].any():
        df = df.sort_values(by="Ratio_Value")
    
    # グラフ描画
    fig, ax = plt.subplots(figsize=(8, 5))
    base_color = "steelblue" # 落ち着いた青色
    
    if graph_type == "棒グラフ (Bar Plot)":
        # エラーバーなし（clean bar plot）
        sns.barplot(data=df, x="Group", y="Value", ax=ax, color=base_color, errorbar=None, capsize=.1)
        
    elif graph_type == "箱ひげ図 (Box Plot)":
        sns.boxplot(data=df, x="Group", y="Value", ax=ax, color=base_color, width=0.5)
        sns.stripplot(data=df, x="Group", y="Value", ax=ax, color=".2", jitter=True)
        
    elif graph_type == "バイオリン図 (Violin Plot)":
        sns.violinplot(data=df, x="Group", y="Value", ax=ax, color=base_color, inner="quartile")
        
    elif graph_type == "ドットプロット (Strip Plot)":
        sns.stripplot(data=df, x="Group", y="Value", ax=ax, size=10, color=base_color, jitter=True)
        # 平均値のバーを表示
        sns.pointplot(data=df, x="Group", y="Value", ax=ax, errorbar=None, color="firebrick", markers="_", scale=1.5, join=False)

    ax.set_ylabel(df['Unit'].iloc[0])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 描画
    st.pyplot(fig)

    # データテーブルとダウンロード
    with st.expander("詳細データを見る"):
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("CSV形式でダウンロード", csv, "analysis_data.csv", "text/csv")
