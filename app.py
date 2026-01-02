import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime  # JST日時取得用

# ---------------------------------------------------------
# 0. ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier Pro (Final)", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ---------------------------------------------------------
# 1. 関数定義
# ---------------------------------------------------------
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

def get_mask(hsv_img, color_name, sens, bright_min):
    """通常の抽出用マスク（細胞カウント用）"""
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

def get_tissue_mask(hsv_img, color_name, sens, bright_min):
    """【組織面積計算用】穴埋め処理付きマスク"""
    # 1. 基本的な色抽出
    mask = get_mask(hsv_img, color_name, sens, bright_min)
    
    # 2. モルフォロジー演算（クロージング）で隙間を埋める
    kernel = np.ones((15, 15), np.uint8) 
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. さらに輪郭内部を塗りつぶす（Fill Holes）
    cnts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    # ある程度大きい塊だけを組織とみなす（微小ノイズ除去）
    valid_tissue = [c for c in cnts if cv2.contourArea(c) > 500]
    cv2.drawContours(mask_filled, valid_tissue, -1, 255, thickness=cv2.FILLED)
    
    return mask_filled

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# ---------------------------------------------------------
# 2. サイドバー設定
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 【Notice / ご案内】")
    st.info("""
    This tool is a beta version. If you plan to use results from this tool in your publications or conference presentations, **please contact the developer (Seiji Kaneko) in advance.**

    本ツールは現在開発中のベータ版です。論文掲載や学会発表等に使用される際は、**事前に開発者（金子）まで必ず一報ください。**

    👉 **[Contact & Feedback Form / 連絡窓口](https://forms.gle/xgNscMi3KFfWcuZ1A)**

    We will provide guidance on validation support and proper acknowledgments/co-authorship.
    バリデーションのサポートや、謝辞・共著の記載についてご案内させていただきます。
    """)
    st.divider()

    st.header("Analysis Recipe")
    mode = st.selectbox("解析モードを選択:", [
        "1. 単色面積率 (Area)",
        "2. 細胞核カウント (Count)",
        "3. 汎用共局在解析 (Colocalization)",
        "4. 汎用空間距離解析 (Spatial Distance)",
        "5. 割合トレンド解析 (Ratio Analysis)"
    ])
    st.divider()

    # --- モード別設定 ---
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
            bright_count = st.slider("細胞輝度しきい値", 0, 255, 50)
            
            # --- ★組織エリア正規化設定 ---
            st.divider()
            use_roi_norm = st.checkbox("組織エリア(CK8など)で密度を計算する", value=True)
            if use_roi_norm:
                # 【重要】警告文の追加
                st.markdown("""
                :red[**実際の染色に用いた色をお選びください。その他の色で解析しようとするとノイズが影響を及ぼし、正確な細胞核カウントが行えません。**]
                """)
                
                roi_color = st.selectbox("組織の色 (分母):", list(COLOR_MAP.keys()), index=2) 
                sens_roi = st.slider("組織感度", 5, 50, 20, key="roi_sens")
                bright_roi = st.slider("組織輝度", 0, 255, 40, key="roi_bright")

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

    # --- スケール設定 ---
    st.divider()
    with st.expander("📏 スケール設定 (Calibration)", expanded=True):
        st.caption("1ピクセルあたりの実寸を入力すると、面積(mm²)や密度(cells/mm²)を自動算出します。")
        # ★初期値: 1.5267
        scale_val = st.number_input("1pxの長さ (μm/px)", value=1.5267, step=0.1, format="%.4f", help="0の場合、ピクセル単位のみで計算します")

    if st.button("履歴を全消去"):
        st.session_state.analysis_history = []
        st.rerun()

    st.divider()
    st.caption("【免責事項 / Disclaimer】")
    st.caption("""
    本ツールは画像解析の補助を目的としています。
    照明条件や設定により結果が変動するため、最終的な解釈および結論については、
    利用者が専門的知見に基づいて判断してください。
 
    This tool is for assistive purposes. Final interpretations should be 
    made by the user based on professional expertise.
    """)

# ---------------------------------------------------------
# 3. メインエリア
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition")
st.caption("2025年最終版：解析・データ抽出専用 (Scale: 1.5267 μm/px)")

uploaded_files = st.file_uploader("画像をまとめてアップロード", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} 枚の画像を解析中...")
    batch_results = []
    
    for i, file in enumerate(uploaded_files):
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        
        # =========================================================
        # ★【16-bit対応＆オートスケーリング】
        # =========================================================
        # 1. まずは「そのまま(UNCHANGED)」読み込む
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if img_raw is not None:
            # 2. 16-bit (uint16) または 8-bit超の輝度がある場合
            if img_raw.dtype == np.uint16 or img_raw.max() > 255:
                # オートスケーリング: 上位2%を飽和させて0-255に正規化
                p_min, p_max = np.percentile(img_raw, (0, 98))
                img_8bit = np.clip((img_raw - p_min) * (255.0 / (p_max - p_min + 1e-5)), 0, 255).astype(np.uint8)
                
                # モノクロ16bitならRGBへ変換
                if len(img_8bit.shape) == 2:
                    img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = img_8bit
            else:
                # 通常の画像はカラー読み込み
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # 以降の処理は元のコード通り
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            val, unit = 0.0, ""
            res_display = img_rgb.copy()
            
            # --- 視野面積の計算 ---
            fov_area_mm2 = 0.0
            if scale_val > 0:
                h, w = img_rgb.shape[:2]
                fov_area_mm2 = (h * w) * ((scale_val / 1000) ** 2)

            # --- 解析ロジック ---
            # 1. 面積率 (Area)
            if mode == "1. 単色面積率 (Area)" or (mode.startswith("5.") and trend_metric == "面積率 (Area)"):
                mask = get_mask(img_hsv, target_a, sens_a, bright_a)
                val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                unit = f"% Area"
                res_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                res_display[:, :, 0] = 0
                res_display[:, :, 2] = 0
                
                real_area_str = ""
                if fov_area_mm2 > 0:
                    real_area = fov_area_mm2 * (val / 100)
                    real_area_str = f"{real_area:.4f} mm²"

            # 2. カウント (Count)
            elif mode == "2. 細胞核カウント (Count)":
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                final = cv2.bitwise_and(th, otsu)
                cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                val, unit = len(valid), "cells"
                
                # 細胞描画 (緑)
                cv2.drawContours(res_display, valid, -1, (0,255,0), 2)
                
                density_str = ""
                if scale_val > 0:
                    if 'use_roi_norm' in locals() and use_roi_norm:
                        # 組織マスク生成 (穴埋め版)
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        roi_pixel_count = cv2.countNonZero(mask_roi)
                        real_roi_area_mm2 = roi_pixel_count * ((scale_val / 1000) ** 2)
                        
                        if real_roi_area_mm2 > 0:
                            density = val / real_roi_area_mm2
                            density_str = f"{int(density):,} cells/mm² (ROI)"
                            
                            # 分母エリアを赤枠で描画
                            roi_cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(res_display, roi_cnts, -1, (255,0,0), 3) 
                        else:
                            density_str = "ROI Area is 0"

                    elif fov_area_mm2 > 0:
                        density = val / fov_area_mm2
                        density_str = f"{int(density):,} cells/mm² (FOV)"

            # 3. 共局在 (Coloc)
            elif mode == "3. 汎用共局在解析 (Colocalization)" or (mode.startswith("5.") and trend_metric == "共局在率 (Colocalization)"):
                mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
                mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                coloc = cv2.bitwise_and(mask_a, mask_b)
                denom = cv2.countNonZero(mask_a)
                val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                unit = f"% Coloc"
                res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
            
            # 4. 距離 (Distance)
            elif mode == "4. 汎用空間距離解析 (Spatial Distance)":
                mask_a = get_mask(img_hsv, target_a, sens_common, bright_common)
                mask_b = get_mask(img_hsv, target_b, sens_common, bright_common)
                pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                
                if pts_a and pts_b:
                    # ピクセル距離算出
                    val_px = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                    
                    # スケール換算
                    if scale_val > 0:
                        val = val_px * scale_val
                        unit = "μm Dist"
                    else:
                        val = val_px
                        unit = "px Dist"
                else: 
                    val = 0
                    unit = "Dist"
                
                res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)
            
            val = max(0.0, val)

            # =========================================================
            # ★【修正】ファイル名(Image_Name)をCSVに追加
            # =========================================================
            entry = {
                "Image_Name": file.name,   # <--- ここでファイル名を保存
                "Group": sample_group,
                "Value": val,
                "Unit": unit,
                "Is_Trend": mode.startswith("5."),
                "Ratio_Value": ratio_val if mode.startswith("5.") else 0
            }
            batch_results.append(entry)
            
            with st.expander(f"📷 Image {i+1}: {file.name}", expanded=True):
                st.markdown(f"### Result: **{val:.2f} {unit}**")
                
                if mode == "1. 単色面積率 (Area)" and scale_val > 0:
                    st.metric("実組織面積", real_area_str)
                elif mode == "2. 細胞核カウント (Count)" and scale_val > 0:
                    st.metric("細胞密度", density_str)

                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original (Auto-Scaled)", use_container_width=True)
                c2.image(res_display, caption="Analyzed", use_container_width=True)

    if st.button(f"データ {len(batch_results)} 件を追加", type="primary"):
        st.session_state.analysis_history.extend(batch_results)
        st.rerun()

# ---------------------------------------------------------
# 4. データエクスポート
# ---------------------------------------------------------
if st.session_state.analysis_history:
    st.divider()
    st.header("💾 Data Export")
    df = pd.DataFrame(st.session_state.analysis_history)
    
    # データをCSV保存しやすく整形（列の並び替え）
    # Image_Nameを一番左に持ってくる
    cols = ["Image_Name", "Group", "Value", "Unit", "Is_Trend", "Ratio_Value"]
    # 存在しないカラムがあるとエラーになるので、実際にあるものだけで再構成
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df["Value"] = df["Value"].clip(lower=0) 
    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    file_name = f"quantified_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 CSVデータを保存", df.to_csv(index=False).encode('utf-8'), file_name, "text/csv")
