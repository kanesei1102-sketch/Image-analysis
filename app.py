import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime  # JST日時取得用
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ---------------------------------------------------------
# 0. ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier Pro (Fixed)", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ---------------------------------------------------------
# 1. 関数定義 (画像処理)
# ---------------------------------------------------------
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

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

def get_tissue_mask(hsv_img, color_name, sens, bright_min):
    mask = get_mask(hsv_img, color_name, sens, bright_min)
    kernel = np.ones((15, 15), np.uint8) 
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
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
# 2. バリデーションデータ読込関数 (キャッシュ有効)
# ---------------------------------------------------------
@st.cache_data
def load_validation_data():
    files = {
        'C14': 'quantified_data_20260102_201522.csv',
        'C40': 'quantified_data_20260102_194322.csv',
        'C70': 'quantified_data_20260103_093427.csv',
        'C100': 'quantified_data_20260102_202525.csv'
    }
    data_list = []
    mapping = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}

    for density, filename in files.items():
        try:
            df = pd.read_csv(filename)
            col = 'Image_Name' if 'Image_Name' in df.columns else 'File Name'
            for _, row in df.iterrows():
                fname = str(row[col])
                val = row['Value']
                # チャネル判定
                channel = 'W1' if 'w1' in fname.lower() else 'W2' if 'w2' in fname.lower() else None
                if not channel: continue
                # フォーカスレベル抽出
                f_match = re.search(r'_F(\d+)_', fname)
                if f_match:
                    focus = int(f_match.group(1))
                    accuracy = (val / mapping[density]) * 100
                    data_list.append({
                        'Density': density,
                        'Ground Truth': mapping[density],
                        'Focus': focus,
                        'Channel': channel,
                        'Value': val,
                        'Accuracy': accuracy
                    })
        except FileNotFoundError:
            pass 
    return pd.DataFrame(data_list)

# アプリ起動時にロード
df_val = load_validation_data()

# ---------------------------------------------------------
# 3. メインレイアウト & サイドバー (完全復元)
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition")
st.caption("2026年最新版：解析・データ抽出専用 (Scale: 1.5267 μm/px)")

tab_main, tab_val = st.tabs(["🚀 解析実行 (Image Analysis)", "🏆 性能証明 (Validation Report)"])

# --- サイドバー設定 ---
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

    if mode == "5. 割合トレンド解析 (Ratio Analysis)":
        st.markdown("### 🔢 条件設定 (Batch)")
        trend_metric = st.radio("測定対象:", ["共局在率 (Colocalization)", "面積率 (Area)"])
        ratio_val = st.number_input("条件値:", value=0, step=10)
        ratio_unit = st.text_input("単位:", value="%", key="unit")
        sample_group = f"{ratio_val}{ratio_unit}"
        st.info(f"ラベル: **{sample_group}**")
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
            
            use_roi_norm = st.checkbox("組織エリア(CK8など)で密度を計算する", value=True)
            if use_roi_norm:
                st.markdown("""
                :red[**実際の染色に用いた色をお選びください。その他の色で解析しようとするとノイズが影響を及ぼし、正確な細胞核カウントが行えません。**]
                """)
                roi_color = st.selectbox("組織の色:", list(COLOR_MAP.keys()), index=2) 
                sens_roi = st.slider("組織感度", 5, 50, 20)
                bright_roi = st.slider("組織輝度", 0, 255, 40)

        elif mode == "3. 汎用共局在解析 (Colocalization)":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("A感度", 5, 50, 20)
                bright_a = st.slider("A輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("B感度", 5, 50, 20)
                bright_b = st.slider("B輝度", 0, 255, 60)
        elif mode == "4. 汎用空間距離解析 (Spatial Distance)":
            target_a = st.selectbox("起点A:", list(COLOR_MAP.keys()), index=2)
            target_b = st.selectbox("対象B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("色感度", 5, 50, 20)
            bright_common = st.slider("輝度", 0, 255, 60)

    st.divider()
    with st.expander("📏 スケール設定 (Calibration)", expanded=True):
        st.caption("1ピクセルあたりの実寸を入力すると、面積(mm²)や密度(cells/mm²)を自動算出します。")
        scale_val = st.number_input("1pxの長さ (μm/px)", value=1.5267, format="%.4f")

    if st.button("履歴を全消去"):
        st.session_state.analysis_history = []
        st.rerun()

    st.divider()
    st.caption("【免責事項 / Disclaimer】")
    st.caption("""
    本ツールは画像解析の補助を目的としています。
    照明条件や設定により結果が変動するため、最終的な解釈および結論については、
    利用者が専門的知見に基づいて判断してください。
    """)

# ---------------------------------------------------------
# 4. タブ1: 解析実行 (ロジック完全復元 + 16bit Float演算対応)
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("画像をまとめてアップロード", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"{len(uploaded_files)} 枚の画像を解析中...")
        batch_results = []
        
        for i, file in enumerate(uploaded_files):
            file.seek(0)
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            
            # === [START] 16bit / 32bit Float 内部演算ロジック ===
            # cv2.IMREAD_UNCHANGED (-1) でオリジナルの深度を維持してロード
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            
            img_bgr = None
            if img_raw is not None:
                # 32bit Float に変換して演算精度を確保
                img_float = img_raw.astype(np.float32)

                # Min-Max Normalization (32bit精度で計算)
                # (x - min) / (max - min) * 255.0
                min_val = np.min(img_float)
                max_val = np.max(img_float)
                
                if max_val > min_val:
                    img_norm = (img_float - min_val) / (max_val - min_val) * 255.0
                else:
                    # 真っ黒または単色の場合
                    img_norm = np.clip(img_float, 0, 255)

                # 解析用フォーマット (uint8) へ変換
                # ※ここで初めて8bitに丸めることで、スライダー等の既存機能と互換性を維持
                img_8bit = np.clip(img_norm, 0, 255).astype(np.uint8)
                
                # チャンネル形式を BGR (3ch) に統一
                if len(img_8bit.shape) == 2:  # Grayscale -> BGR
                    img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
                elif img_8bit.shape[2] == 4:  # BGRA (透明度あり) -> BGR
                    img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_BGRA2BGR)
                elif img_8bit.shape[2] == 3:  # BGR
                    img_bgr = img_8bit
            # === [END] 16bit / 32bit Float 内部演算ロジック ===
            
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                
                val, unit = 0.0, ""
                res_display = img_rgb.copy()
                
                fov_area_mm2 = 0.0
                if scale_val > 0:
                    h, w = img_rgb.shape[:2]
                    fov_area_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                # --- 1. Area (実面積計算付き) ---
                if mode == "1. 単色面積率 (Area)" or (mode.startswith("5.") and trend_metric == "面積率 (Area)"):
                    mask = get_mask(img_hsv, target_a, sens_a, bright_a)
                    val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                    unit = f"% Area"
                    res_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    res_display[:, :, 0] = 0; res_display[:, :, 2] = 0
                    real_area_str = ""
                    if fov_area_mm2 > 0:
                        real_area = fov_area_mm2 * (val / 100)
                        real_area_str = f"{real_area:.4f} mm²"

                # --- 2. Count (密度計算付き) ---
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
                    
                    density_str = ""
                    if scale_val > 0:
                        if 'use_roi_norm' in locals() and use_roi_norm:
                            mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                            roi_pixel_count = cv2.countNonZero(mask_roi)
                            real_roi_area_mm2 = roi_pixel_count * ((scale_val / 1000) ** 2)
                            if real_roi_area_mm2 > 0:
                                density = val / real_roi_area_mm2
                                density_str = f"{int(density):,} cells/mm² (ROI)"
                                roi_cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(res_display, roi_cnts, -1, (255,0,0), 3) 
                            else:
                                density_str = "ROI Area is 0"
                        elif fov_area_mm2 > 0:
                            density = val / fov_area_mm2
                            density_str = f"{int(density):,} cells/mm² (FOV)"

                # --- 3. Coloc ---
                elif mode == "3. 汎用共局在解析 (Colocalization)" or (mode.startswith("5.") and trend_metric == "共局在率 (Colocalization)"):
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
                    mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b)
                    denom = cv2.countNonZero(mask_a)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                    unit = f"% Coloc"
                    res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
                
                # --- 4. Distance ---
                elif mode == "4. 汎用空間距離解析 (Spatial Distance)":
                    mask_a = get_mask(img_hsv, target_a, sens_common, bright_common)
                    mask_b = get_mask(img_hsv, target_b, sens_common, bright_common)
                    pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                    if pts_a and pts_b:
                        val_px = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                        if scale_val > 0:
                            val = val_px * scale_val; unit = "μm Dist"
                        else:
                            val = val_px; unit = "px Dist"
                    else: 
                        val = 0; unit = "Dist"
                    res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)
                
                val = max(0.0, val)

                # 結果登録 (ファイル名も確実に)
                entry = {
                    "File Name": file.name,
                    "Group": sample_group,
                    "Value": val,
                    "Unit": unit,
                    "Is_Trend": mode.startswith("5."),
                    "Ratio_Value": ratio_val if mode.startswith("5.") else 0
                }
                batch_results.append(entry)
                
                # 結果表示 (st.metricで綺麗に表示)
                st.divider()
                st.markdown(f"### 📷 Image {i+1}: {file.name}")
                st.markdown(f"### Result: **{val:.2f} {unit}**")
                
                if mode == "1. 単色面積率 (Area)" and scale_val > 0 and 'real_area_str' in locals():
                    st.metric("実組織面積", real_area_str)
                elif mode == "2. 細胞核カウント (Count)" and scale_val > 0 and 'density_str' in locals():
                    st.metric("細胞密度", density_str)

                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original", use_container_width=True)
                c2.image(res_display, caption="Analyzed", use_container_width=True)

        if st.button(f"データ {len(batch_results)} 件を追加", type="primary"):
            st.session_state.analysis_history.extend(batch_results)
            st.rerun()

    if st.session_state.analysis_history:
        st.divider()
        st.header("💾 Data Export")
        df = pd.DataFrame(st.session_state.analysis_history)
        df["Value"] = df["Value"].clip(lower=0) 
        
        # カラム順序の整理
        cols = ["File Name", "Group", "Value", "Unit", "Is_Trend", "Ratio_Value"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        now = datetime.datetime.now() + datetime.timedelta(hours=9)
        file_name = f"quantified_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 CSVデータを保存", df.to_csv(index=False).encode('utf-8'), file_name, "text/csv")

# ---------------------------------------------------------
# 5. タブ2: 性能証明 (Full Version)
# ---------------------------------------------------------
with tab_val:
    st.header("🏆 性能バリデーション・最終報告 (2026 Latest)")
    
    st.markdown("""
    * **検証ソース:** [Broad Bioimage Benchmark Collection (BBBC005)](https://bbbc.broadinstitute.org/BBBC005)
    * **検証総数:** 3,200枚 (C14, C40, C70, C100 × 各800枚)
    * **方法論:** 各密度グループに対して**個別にパラメータを最適化**し、適切なキャリブレーション下での最大性能を実証しました。
    """)

    if not df_val.empty:
        gt_map = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
        df_hq = df_val[(df_val['Focus'] >= 1) & (df_val['Focus'] <= 5)]
        
        # 統計メトリクス
        w1_hq = df_hq[df_hq['Channel'] == 'W1']
        avg_acc = w1_hq['Accuracy'].mean()
        df_lin = w1_hq.groupby('Ground Truth')['Value'].mean().reset_index()
        r2 = np.corrcoef(df_lin['Ground Truth'], df_lin['Value'])[0, 1]**2

        m1, m2, m3 = st.columns(3)
        m1.metric("核カウント平均精度 (W1)", f"{avg_acc:.1f}%", help="Focus 1-5平均")
        m2.metric("統計的線形性 (R²)", f"{r2:.4f}", help="実測値ベース")
        m3.metric("連続処理安定性", "3,200+ 枚")

        st.divider()

        # グラフ1: Linearity
        st.subheader("📈 1. 計数能力と線形性 (Linearity)")
        st.info("💡 **結論:** W1（核）は極めて高い線形性を示し、W2（細胞体）はV字型の乖離を示します。")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='Ideal')
        ax1.scatter(df_lin['Ground Truth'], df_lin['Value'], color='#1f77b4', s=100, label='W1 (Nuclei)', zorder=5)
        # W2も描画
        w2_lin = df_hq[df_hq['Channel'] == 'W2'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax1.scatter(w2_lin['Ground Truth'], w2_lin['Value'], color='#ff7f0e', s=100, marker='D', label='W2 (Cytoplasm)', zorder=5)
        
        z = np.polyfit(df_lin['Ground Truth'], df_lin['Value'], 1)
        ax1.plot(df_lin['Ground Truth'], np.poly1d(z)(df_lin['Ground Truth']), '#1f77b4', alpha=0.5, label='W1 Reg')
        ax1.set_xlabel('Ground Truth'); ax1.set_ylabel('Measured'); ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.divider()

        # グラフ2 & 3
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 2. 密度別精度比較")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            df_bar = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().reset_index()
            df_bar['Density'] = pd.Categorical(df_bar['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.barplot(data=df_bar, x='Density', y='Accuracy', hue='Channel', palette={'W1': '#1f77b4', 'W2': '#ff7f0e'}, ax=ax2)
            ax2.axhline(100, color='red', linestyle='--'); ax2.set_ylabel('Accuracy (%)')
            st.pyplot(fig2)
        
        with c2:
            st.subheader("📉 3. 光学的な堅牢性")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            df_decay = df_val[df_val['Channel'] == 'W1'].copy()
            df_decay['Density'] = pd.Categorical(df_decay['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.lineplot(data=df_decay, x='Focus', y='Accuracy', hue='Density', marker='o', ax=ax3)
            ax3.axhline(100, color='red', linestyle='--'); ax3.set_ylabel('Accuracy (%)')
            st.pyplot(fig3)

        st.divider()

        # 4. 数値テーブル (W1/W2完全版)
        st.subheader("📋 4. バリデーション数値データサマリー")
        summary = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().unstack().reset_index()
        summary['真値'] = summary['Density'].map(gt_map)
        summary['W1実測'] = (summary['W1']/100)*summary['真値']
        summary['W2実測'] = (summary['W2']/100)*summary['真値']
        summary['Density'] = pd.Categorical(summary['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        summary = summary.sort_values('Density')

        st.table(summary[['Density', '真値', 'W1', 'W1実測', 'W2', 'W2実測']].rename(columns={
            'W1': 'W1精度(%)', 'W1実測': 'W1平均(Cells)',
            'W2': 'W2精度(%)', 'W2実測': 'W2平均(Cells)'
        }))
        
        st.info("💡 **結論:** W1(核)は全領域で高精度を維持。W2(細胞体)は密度による変動(過少/過剰)が激しく定量に不適です。")
    else:
        st.error("CSVファイルが読み込めません。リポジトリに配置してください。")
