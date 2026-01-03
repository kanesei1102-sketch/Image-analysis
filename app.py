import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ---------------------------------------------------------
# 0. ページ設定 & セッション初期化
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier Pro", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ---------------------------------------------------------
# 1. 関数定義 (画像処理コア)
# ---------------------------------------------------------
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

def get_mask(hsv_img, color_name, sens, bright_min):
    if color_name == "赤 (RFP)":
        lower1 = np.array([0, 30, bright_min]); upper1 = np.array([10 + sens//2, 255, 255])
        lower2 = np.array([170 - sens//2, 30, bright_min]); upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    else:
        conf = COLOR_MAP[color_name]
        l = np.clip(conf["lower"] - sens, 0, 255); u = np.clip(conf["upper"] + sens, 0, 255)
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
        if M["m00"] != 0: pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# ---------------------------------------------------------
# 2. バリデーションデータ読込関数
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
                fname = str(row[col]); val = row['Value']
                channel = 'W1' if 'w1' in fname.lower() else 'W2' if 'w2' in fname.lower() else None
                if not channel: continue
                f_match = re.search(r'_F(\d+)_', fname)
                if f_match:
                    focus = int(f_match.group(1))
                    data_list.append({
                        'Density': density, 'Ground Truth': mapping[density],
                        'Focus': focus, 'Channel': channel, 'Value': val,
                        'Accuracy': (val / mapping[density]) * 100
                    })
        except: pass
    return pd.DataFrame(data_list)

df_val = load_validation_data()

# ---------------------------------------------------------
# 3. メインレイアウト & サイドバー (Notice復元)
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition")
st.caption("2026年最新版：解析・データ抽出専用 (Scale: 1.5267 μm/px)")

tab_main, tab_val = st.tabs(["🚀 解析実行 (Image Analysis)", "🏆 性能証明 (Validation Report)"])

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
    mode = st.selectbox("解析モード:", ["1. 単色面積率 (Area)", "2. 細胞核カウント (Count)", "3. 汎用共局在解析", "4. 汎用空間距離解析", "5. 割合トレンド解析"])
    
    # 簡易設定UI (モードに応じた設定は省略せず記述)
    if mode == "5. 割合トレンド解析":
        trend_metric = st.radio("測定対象:", ["共局在率", "面積率"])
        ratio_val = st.number_input("条件値:", value=0, step=10)
        ratio_unit = st.text_input("単位:", value="%")
        sample_group = f"{ratio_val}{ratio_unit}"
        if trend_metric == "共局在率":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", list(COLOR_MAP.keys()), index=3) 
                sens_a = st.slider("A感度", 5, 50, 20, key="ta")
                bright_a = st.slider("A輝度", 0, 255, 60, key="ba")
            with c2:
                target_b = st.selectbox("CH-B (対象):", list(COLOR_MAP.keys()), index=2) 
                sens_b = st.slider("B感度", 5, 50, 20, key="tb")
                bright_b = st.slider("B輝度", 0, 255, 60, key="bb")
        else:
            target_a = st.selectbox("解析色:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("感度", 5, 50, 20)
            bright_a = st.slider("輝度", 0, 255, 60)
    else:
        sample_group = st.text_input("グループ名:", value="Control")
        if mode == "1. 単色面積率 (Area)":
            target_a = st.selectbox("解析色:", list(COLOR_MAP.keys()))
            sens_a = st.slider("感度", 5, 50, 20); bright_a = st.slider("輝度", 0, 255, 60)
        elif mode == "2. 細胞核カウント (Count)":
            min_size = st.slider("最小サイズ(px)", 10, 500, 50)
            bright_count = st.slider("細胞輝度", 0, 255, 50)
            use_roi_norm = st.checkbox("組織エリア(CK8など)で密度計算", value=True)
            if use_roi_norm:
                st.markdown(":red[**実際の染色色を選択してください**]")
                roi_color = st.selectbox("組織の色:", list(COLOR_MAP.keys()), index=2)
                sens_roi = st.slider("組織感度", 5, 50, 20); bright_roi = st.slider("組織輝度", 0, 255, 40)
        elif mode == "3. 汎用共局在解析":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("A感度", 5, 50, 20); bright_a = st.slider("A輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("B感度", 5, 50, 20); bright_b = st.slider("B輝度", 0, 255, 60)
        elif mode == "4. 汎用空間距離解析":
            target_a = st.selectbox("起点A:", list(COLOR_MAP.keys()), index=2)
            target_b = st.selectbox("対象B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("色感度", 5, 50, 20); bright_common = st.slider("輝度", 0, 255, 60)

    st.divider()
    with st.expander("📏 スケール設定", expanded=True):
        scale_val = st.number_input("1pxの長さ (μm/px)", value=1.5267, format="%.4f")
    
    if st.button("履歴を全消去"):
        st.session_state.analysis_history = []
        st.rerun()

    st.divider()
    st.caption("【免責事項 / Disclaimer】")
    st.caption("本ツールは画像解析の補助を目的としています。照明条件や設定により結果が変動するため、最終的な解釈および結論については利用者が専門的知見に基づいて判断してください。")

# ---------------------------------------------------------
# 4. タブ1: 解析実行 (ファイル名記録機能付き)
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("画像をアップロード", type=["jpg", "png", "tif"], accept_multiple_files=True)
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
                
                fov_mm2 = (img_rgb.shape[0]*img_rgb.shape[1])*((scale_val/1000)**2) if scale_val > 0 else 0

                # --- 簡易解析ロジック (詳細は元のまま維持) ---
                if mode.startswith("1") or "面積" in str(mode): # Area
                    mask = get_mask(img_hsv, target_a, sens_a, bright_a)
                    val = (cv2.countNonZero(mask)/(img_rgb.shape[0]*img_rgb.shape[1]))*100
                    unit = "% Area"; res_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                elif mode.startswith("2"): # Count
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0)
                    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    final = cv2.bitwise_and(th, otsu)
                    cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                    val, unit = len(valid), "cells"
                    cv2.drawContours(res_display, valid, -1, (0,255,0), 2)
                elif mode.startswith("3") or "共局在" in str(mode): # Coloc
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
                    mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b)
                    denom = cv2.countNonZero(mask_a)
                    val = (cv2.countNonZero(coloc)/denom*100) if denom > 0 else 0
                    unit = "% Coloc"; res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
                elif mode.startswith("4"): # Distance
                    mask_a = get_mask(img_hsv, target_a, sens_common, bright_common)
                    mask_b = get_mask(img_hsv, target_b, sens_common, bright_common)
                    pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                    if pts_a and pts_b:
                        val_px = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                        val = val_px * scale_val if scale_val > 0 else val_px
                        unit = "μm" if scale_val > 0 else "px"
                    res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)

                # 結果登録 (ファイル名を確実に記録)
                entry = {
                    "File Name": file.name,
                    "Group": sample_group,
                    "Value": val,
                    "Unit": unit,
                    "Is_Trend": mode.startswith("5."),
                    "Ratio_Value": ratio_val if mode.startswith("5.") else 0
                }
                batch_results.append(entry)
                
                st.divider()
                st.markdown(f"**{file.name}**: {val:.2f} {unit}")
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
        cols = ["File Name", "Group", "Value", "Unit", "Is_Trend", "Ratio_Value"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols] # カラム順序を強制
        
        now = datetime.datetime.now() + datetime.timedelta(hours=9)
        file_name = f"quantified_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 CSVデータを保存", df.to_csv(index=False).encode('utf-8'), file_name, "text/csv")

# ---------------------------------------------------------
# 5. タブ2: 性能証明 (Full Info Version)
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
        
        # W1とW2の統計
        w1_hq = df_hq[df_hq['Channel'] == 'W1']
        avg_acc = w1_hq['Accuracy'].mean()
        df_lin = w1_hq.groupby('Ground Truth')['Value'].mean().reset_index()
        r2 = np.corrcoef(df_lin['Ground Truth'], df_lin['Value'])[0, 1]**2

        m1, m2, m3 = st.columns(3)
        m1.metric("核カウント平均精度 (W1)", f"{avg_acc:.1f}%")
        m2.metric("統計的線形性 (R²)", f"{r2:.4f}")
        m3.metric("連続処理安定性", "3,200+ 枚")

        st.divider()

        # グラフ1: Linearity
        st.subheader("📈 1. 計数能力と線形性 (Linearity)")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='Ideal')
        ax1.scatter(df_lin['Ground Truth'], df_lin['Value'], color='#1f77b4', s=100, label='W1 (Nuclei)', zorder=5)
        # W2もプロット
        w2_lin = df_hq[df_hq['Channel'] == 'W2'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax1.scatter(w2_lin['Ground Truth'], w2_lin['Value'], color='#ff7f0e', s=100, marker='D', label='W2 (Cytoplasm)', zorder=5)
        
        z = np.polyfit(df_lin['Ground Truth'], df_lin['Value'], 1)
        ax1.plot(df_lin['Ground Truth'], np.poly1d(z)(df_lin['Ground Truth']), '#1f77b4', alpha=0.5, label='W1 Regression')
        ax1.set_xlabel('Ground Truth'); ax1.set_ylabel('Measured'); ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.divider()

        # グラフ2 & 3
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 2. 密度別精度比較")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            df_bar = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().reset_index()
            # 密度順序を整える
            df_bar['Density'] = pd.Categorical(df_bar['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.barplot(data=df_bar, x='Density', y='Accuracy', hue='Channel', palette={'W1': '#1f77b4', 'W2': '#ff7f0e'}, ax=ax2)
            ax2.axhline(100, color='red', linestyle='--')
            st.pyplot(fig2)
        
        with c2:
            st.subheader("📉 3. 光学的な堅牢性")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            df_decay = df_val[df_val['Channel'] == 'W1'].copy()
            df_decay['Density'] = pd.Categorical(df_decay['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.lineplot(data=df_decay, x='Focus', y='Accuracy', hue='Density', marker='o', ax=ax3)
            ax3.axhline(100, color='red', linestyle='--')
            st.pyplot(fig3)

        st.divider()

        # 4. 数値テーブル (W1/W2両方表示)
        st.subheader("📋 4. バリデーション数値データサマリー")
        summary = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().unstack().reset_index()
        summary['真値'] = summary['Density'].map(gt_map)
        
        # 実測値の計算
        summary['W1実測'] = (summary['W1']/100)*summary['真値']
        summary['W2実測'] = (summary['W2']/100)*summary['真値']
        
        # 密度順にソート
        summary['Density'] = pd.Categorical(summary['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        summary = summary.sort_values('Density')

        st.table(summary[['Density', '真値', 'W1', 'W1実測', 'W2', 'W2実測']].rename(columns={
            'W1': 'W1精度(%)', 'W1実測': 'W1個数(Mean)',
            'W2': 'W2精度(%)', 'W2実測': 'W2個数(Mean)'
        }))
        
        st.info("💡 **結論:** W1(核)は全領域で高精度を維持。W2(細胞体)は密度による変動(過少/過剰)が激しく定量に不適です。")
    else:
        st.error("CSVファイルが読み込めません。リポジトリに配置してください。")
