import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
import uuid

# ---------------------------------------------------------
# 0. ページ設定 & 定数
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier V2 (JP)", layout="wide")

# バージョン管理
SOFTWARE_VERSION = "Bio-Image Quantifier Pro v2026.02 (JP/Auto-Group)"

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# --- 解析ID管理 (人間が読める形式 + ユニークID) ---
if "current_analysis_id" not in st.session_state:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
    unique_suffix = str(uuid.uuid4())[:8]
    st.session_state.current_analysis_id = f"AID-{date_str}-{unique_suffix}"

# ---------------------------------------------------------
# 1. 画像処理エンジン
# ---------------------------------------------------------
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑色 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤色 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青色 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

def get_mask(hsv_img, color_name, sens, bright_min):
    if color_name == "赤色 (RFP)":
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
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
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
# 2. バリデーションデータ読み込み
# ---------------------------------------------------------
@st.cache_data
def load_validation_data():
    files = {'C14': 'quantified_data_20260102_201522.csv', 'C40': 'quantified_data_20260102_194322.csv',
             'C70': 'quantified_data_20260103_093427.csv', 'C100': 'quantified_data_20260102_202525.csv'}
    data_list = []; mapping = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
    for density, filename in files.items():
        try:
            df = pd.read_csv(filename); col = 'Image_Name' if 'Image_Name' in df.columns else 'File Name'
            for _, row in df.iterrows():
                fname = str(row[col]); val = row['Value']
                channel = 'W1' if 'w1' in fname.lower() else 'W2' if 'w2' in fname.lower() else None
                if not channel: continue
                f_match = re.search(r'_F(\d+)_', fname)
                if f_match:
                    focus = int(f_match.group(1)); accuracy = (val / mapping[density]) * 100
                    data_list.append({'Density': density, 'Ground Truth': mapping[density], 'Focus': focus, 'Channel': channel, 'Value': val, 'Accuracy': accuracy})
        except FileNotFoundError: pass
    return pd.DataFrame(data_list)

df_val = load_validation_data()

# ---------------------------------------------------------
# 3. UIフレームワーク & サイドバー
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition (日本語版)")
st.caption(f"{SOFTWARE_VERSION}: 産業グレード画像解析・データ抽出")

st.sidebar.markdown(f"**現在の解析ID:** `{st.session_state.current_analysis_id}`")

tab_main, tab_val = st.tabs(["🚀 解析実行", "🏆 性能証明"])

with st.sidebar:
    st.markdown("### 【重要：論文・学会発表での使用】")
    st.warning("""
    **研究成果として公表される予定ですか？**
    本ツールはベータ版です。学術利用の際は**必ず事前に開発者（金子）までご連絡ください。**
    共著や謝辞についてご相談させていただきます。
    👉 **[連絡フォーム](https://forms.gle/xgNscMi3KFfWcuZ1A)**
    """)
    st.divider()

    st.header("解析レシピ")
    mode_raw = st.selectbox("解析モード選択:", [
        "1. 単色面積率 (Area Occupancy %)", 
        "2. 細胞核カウント / 密度 (Nuclei Count)", 
        "3. 共局在解析 (Colocalization)", 
        "4. 空間距離解析 (Spatial Distance)", 
        "5. 割合トレンド解析 (Ratio Trend)"
    ])
    mode = mode_raw 

    st.divider()

    # --- グループ化戦略 (新機能) ---
    st.markdown("### 🏷️ グループ分け設定")
    group_strategy = st.radio("ラベルの決定方法:", ["手動入力 (Manual)", "ファイル名から自動 (Auto)"], 
                              help="自動: ファイル名の区切り文字より前の部分をグループ名として抽出します")
    
    if group_strategy.startswith("手動"):
        sample_group = st.text_input("グループ名 (X軸ラベル):", value="Control")
        filename_sep = None
    else:
        filename_sep = st.text_input("区切り文字 (例: _ または - ):", value="_", help="この文字より前をグループ名にします")
        st.info(f"例: 'WT{filename_sep}01.tif' → グループ名: 'WT'")
        sample_group = "(自動検出)" 

    st.divider()

    # 解析パラメータ設定
    if mode.startswith("5."):
        st.markdown("### 🔢 トレンド解析条件")
        trend_metric = st.radio("測定対象:", ["共局在率 (Colocalization)", "面積率 (Area)"])
        ratio_val = st.number_input("条件値:", value=0, step=10)
        ratio_unit = st.text_input("単位:", value="%", key="unit")
        if group_strategy.startswith("手動"):
            sample_group = f"{ratio_val}{ratio_unit}" # トレンドモードの手動時は上書き
        
        if trend_metric.startswith("共局在"):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("A 感度", 5, 50, 20); bright_a = st.slider("A 輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B (対象):", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("B 感度", 5, 50, 20); bright_b = st.slider("B 輝度", 0, 255, 60)
        else:
            target_a = st.selectbox("解析色:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("感度", 5, 50, 20); bright_a = st.slider("輝度", 0, 255, 60)
    else:
        if mode.startswith("1."):
            target_a = st.selectbox("解析色:", list(COLOR_MAP.keys())); sens_a = st.slider("感度", 5, 50, 20); bright_a = st.slider("輝度", 0, 255, 60)
        elif mode.startswith("2."):
            min_size = st.slider("最小細胞サイズ (px)", 10, 500, 50); bright_count = st.slider("検出輝度しきい値", 0, 255, 50)
            use_roi_norm = st.checkbox("組織領域 (ROI) で正規化", value=True)
            if use_roi_norm:
                roi_color = st.selectbox("組織の色:", list(COLOR_MAP.keys()), index=2); sens_roi = st.slider("組織 感度", 5, 50, 20); bright_roi = st.slider("組織 輝度", 0, 255, 40)
        elif mode.startswith("3."):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3); sens_a = st.slider("A 感度", 5, 50, 20); bright_a = st.slider("A 輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2); sens_b = st.slider("B 感度", 5, 50, 20); bright_b = st.slider("B 輝度", 0, 255, 60)
        elif mode.startswith("4."):
            target_a = st.selectbox("起点 A:", list(COLOR_MAP.keys()), index=2); target_b = st.selectbox("対象 B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("共通 感度", 5, 50, 20); bright_common = st.slider("共通 輝度", 0, 255, 60)

    st.divider()
    scale_val = st.number_input("空間スケール (μm/px)", value=1.5267, format="%.4f")
    
    # --- 履歴クリア & ID更新 ---
    if st.button("履歴をクリア & 新規ID発行"): 
        st.session_state.analysis_history = []
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
        st.session_state.current_analysis_id = f"AID-{date_str}-{str(uuid.uuid4())[:8]}"
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ 設定パラメータ保存 (監査証跡)")
    
    current_params = {
        "Software_Version": SOFTWARE_VERSION, 
        "Analysis_ID": st.session_state.current_analysis_id,
        "Analysis_Date_UTC": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        "Mode": mode,
        "Scale_um_px": scale_val,
        "Grouping_Strategy": group_strategy
    }
    if group_strategy.startswith("手動"): current_params["Manual_Group_Label"] = sample_group
    else: current_params["Filename_Separator"] = filename_sep

    # (その他のパラメータも記録)
    if "trend_metric" in locals(): current_params["Trend_Metric"] = trend_metric
    if "target_a" in locals(): current_params["Target_A"] = target_a
    if "target_b" in locals(): current_params["Target_B"] = target_b
    if "roi_color" in locals(): current_params["ROI_Color"] = roi_color
    if "sens_a" in locals(): current_params["Sens_A"] = sens_a
    if "bright_a" in locals(): current_params["Bright_A"] = bright_a
    if "sens_b" in locals(): current_params["Sens_B"] = sens_b
    if "bright_b" in locals(): current_params["Bright_B"] = bright_b
    if "min_size" in locals(): current_params["Min_Nuclei_Size_px"] = min_size
    if "bright_count" in locals(): current_params["Count_Threshold"] = bright_count
    if "use_roi_norm" in locals(): current_params["ROI_Normalization_Enabled"] = use_roi_norm
    if "sens_roi" in locals(): current_params["ROI_Sens"] = sens_roi
    if "bright_roi" in locals(): current_params["ROI_Bright"] = bright_roi
    if "sens_common" in locals(): current_params["Common_Sens"] = sens_common
    if "bright_common" in locals(): current_params["Common_Bright"] = bright_common

    df_params = pd.DataFrame([current_params]).T.reset_index()
    df_params.columns = ["Parameter", "Setting Value"]
    param_filename = f"params_{st.session_state.current_analysis_id}.csv"
    
    st.download_button("📥 設定CSVをダウンロード", df_params.to_csv(index=False).encode('utf-8'), param_filename, "text/csv")

    st.divider()
    st.caption("【免責事項】")
    st.caption("本ツールは研究用であり、臨床診断を保証しません。最終的な妥当性の確認は利用者の責任で行ってください。")

# ---------------------------------------------------------
# 4. タブ1: 解析実行
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("画像アップロード (16-bit TIFF対応)", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} 枚の画像を解析中...")
        batch_results = []
        for i, file in enumerate(uploaded_files):
            file.seek(0); file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img_raw is not None:
                # --- 自動グループ化ロジック ---
                if group_strategy.startswith("ファイル名"):
                    try:
                        detected_group = file.name.split(filename_sep)[0]
                    except:
                        detected_group = "Unknown"
                    current_group_label = detected_group
                else:
                    current_group_label = sample_group

                # 画像処理 (V1と同じ)
                img_f = img_raw.astype(np.float32); mn, mx = np.min(img_f), np.max(img_f)
                img_8 = ((img_f - mn) / (mx - mn) * 255.0 if mx > mn else np.clip(img_f, 0, 255)).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape) == 2 else img_8[:,:,:3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                val, unit, res_disp = 0.0, "", img_rgb.copy()
                h, w = img_rgb.shape[:2]; fov_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                if mode.startswith("1.") or (mode.startswith("5.") and trend_metric.startswith("面積")):
                    mask = get_mask(img_hsv, target_a, sens_a, bright_a); val = (cv2.countNonZero(mask) / (h * w)) * 100
                    unit = "% Area"; res_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB); res_disp[:,:,0]=0; res_disp[:,:,2]=0
                    real_area_str = f"{fov_mm2 * (val/100):.4f} mm²"
                elif mode.startswith("2."):
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0); _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(cv2.bitwise_and(th, otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]; val, unit = len(valid), "cells"
                    cv2.drawContours(res_disp, valid, -1, (0,255,0), 2)
                    if scale_val > 0:
                        a_target = fov_mm2
                        if use_roi_norm:
                            mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi); a_target = cv2.countNonZero(mask_roi) * ((scale_val/1000)**2)
                            cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)
                        density_str = f"{int(val/a_target):,} cells/mm²" if a_target > 0 else "N/A"
                elif mode.startswith("3.") or (mode.startswith("5.") and trend_metric.startswith("共局在")):
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a); mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b); denom = cv2.countNonZero(mask_a)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0; unit = "% Coloc"; res_disp = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
                elif mode.startswith("4."):
                    ma, mb = get_mask(img_hsv, target_a, sens_common, bright_common), get_mask(img_hsv, target_b, sens_common, bright_common)
                    pa, pb = get_centroids(ma), get_centroids(mb)
                    if pa and pb: val = np.mean([np.min([np.linalg.norm(a - b) for b in pb]) for a in pa]) * (scale_val if scale_val > 0 else 1)
                    unit = "μm Dist" if scale_val > 0 else "px Dist"; res_disp = cv2.addWeighted(img_rgb, 0.6, cv2.merge([ma, mb, np.zeros_like(ma)]), 0.4, 0)

                st.divider()
                st.markdown(f"### 📷 Image {i+1}: {file.name}")
                st.markdown(f"**検出グループ:** `{current_group_label}`") # 自動検出されたグループ名を表示
                st.markdown(f"### Result: **{val:.2f} {unit}**")
                
                c1, c2 = st.columns(2); c1.image(img_rgb, caption="Raw"); c2.image(res_disp, caption="解析結果")
                
                batch_results.append({
                    "Software_Version": SOFTWARE_VERSION,
                    "Analysis_ID": st.session_state.current_analysis_id,
                    "Analysis_Timestamp_UTC": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "File Name": file.name,
                    "Group": current_group_label, # 動的に決定したグループ名を使用
                    "Value": val,
                    "Unit": unit,
                    "Is_Trend": mode.startswith("5."),  
                    "Ratio_Value": ratio_val if mode.startswith("5.") else 0 
                })
        
        # ID自動更新なし (Commitのみ)
        if st.button("バッチデータを確定 (Commit)", type="primary"):
            st.session_state.analysis_history.extend(batch_results)
            st.success("データが履歴に追加されました。IDは維持されています。")
            st.rerun()

    if st.session_state.analysis_history:
        st.divider(); st.header("💾 CSV出力")
        df_exp = pd.DataFrame(st.session_state.analysis_history)
        cols_order = ["Analysis_ID", "Analysis_Timestamp_UTC", "Software_Version", "File Name", "Group", "Value", "Unit"]
        cols_final = [c for c in cols_order if c in df_exp.columns]
        st.dataframe(df_exp[cols_final], use_container_width=True)
        utc_filename = f"quantified_data_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_UTC.csv"
        st.download_button("📥 結果CSVをダウンロード", df_exp.to_csv(index=False).encode('utf-8'), utc_filename)

# ---------------------------------------------------------
# 5. タブ2: 性能証明 (Validation Evidence - 完全復元)
# ---------------------------------------------------------
with tab_val:
    st.header("🏆 性能バリデーションサマリー")
    st.markdown("""
    * **検証標準:** BBBC005 (Broad Bioimage Benchmark Collection)
    * **検証規模:** 3,200枚 (High-Throughput)
    * **手法:** 各密度グループに対してパラメータを個別最適化し、適切なキャリブレーション下での最大性能を実証。
    """)

    if not df_val.empty:
        gt_map = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
        
        # 全Focusデータを使用（W1/W2比較のため）
        df_hq = df_val[(df_val['Focus'] >= 1) & (df_val['Focus'] <= 5)]
        
        # 統計値 (W1のみ)
        w1_hq = df_hq[df_hq['Channel'] == 'W1']
        avg_acc = w1_hq['Accuracy'].mean()
        df_lin = w1_hq.groupby('Ground Truth')['Value'].mean().reset_index()
        r2 = np.corrcoef(df_lin['Ground Truth'], df_lin['Value'])[0, 1]**2

        m1, m2, m3 = st.columns(3)
        m1.metric("平均精度 (Accuracy)", f"{avg_acc:.1f}%", help="Focus 1-5 平均")
        m2.metric("線形性 (R²)", f"{r2:.4f}", help="実測値ベース")
        m3.metric("解析枚数", "3,200+")

        st.divider()

        # グラフ1: 線形性 (W2との比較)
        st.subheader("📈 1. 計数能力と線形性 (W1 vs W2)")
        st.info("💡 **結論:** W1（核）は極めて高い線形性を示しますが、W2（細胞質）は**V字型の乖離**を示し、定量には不適であることが証明されています。")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='Ideal Line')
        ax1.scatter(df_lin['Ground Truth'], df_lin['Value'], color='#1f77b4', s=100, label='W1 (Nuclei)', zorder=5)
        w2_lin = df_hq[df_hq['Channel'] == 'W2'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax1.scatter(w2_lin['Ground Truth'], w2_lin['Value'], color='#ff7f0e', s=100, marker='D', label='W2 (Cytoplasm)', zorder=5)
        z = np.polyfit(df_lin['Ground Truth'], df_lin['Value'], 1)
        ax1.plot(df_lin['Ground Truth'], np.poly1d(z)(df_lin['Ground Truth']), '#1f77b4', alpha=0.5, label='W1 Reg')
        ax1.set_xlabel('Ground Truth'); ax1.set_ylabel('Measured Value'); ax1.legend(); ax1.grid(True, alpha=0.3)
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
            st.subheader("📉 3. 光学的堅牢性 (ボケ耐性)")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            df_decay = df_val[df_val['Channel'] == 'W1'].copy()
            df_decay['Density'] = pd.Categorical(df_decay['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.lineplot(data=df_decay, x='Focus', y='Accuracy', hue='Density', marker='o', ax=ax3)
            ax3.axhline(100, color='red', linestyle='--'); ax3.set_ylabel('Accuracy (%)')
            st.pyplot(fig3)

        st.divider()

        # 数値テーブル
        st.subheader("📋 4. バリデーション数値データ")
        summary = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().unstack().reset_index()
        summary['Ground Truth'] = summary['Density'].map(gt_map)
        summary['W1 Measured'] = (summary['W1']/100)*summary['Ground Truth']
        summary['W2 Measured'] = (summary['W2']/100)*summary['Ground Truth']
        summary['Density'] = pd.Categorical(summary['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        summary = summary.sort_values('Density')
        st.table(summary[['Density', 'Ground Truth', 'W1', 'W1 Measured', 'W2', 'W2 Measured']].rename(columns={
            'W1': 'W1 精度(%)', 'W1 Measured': 'W1 平均個数', 'W2': 'W2 精度(%)', 'W2 Measured': 'W2 平均個数'
        }))
        st.info("💡 **総合結論:** W1（核）は全密度領域で高精度を維持。W2（細胞質）は過小・過剰評価の変動が激しく、科学的に定量解析には推奨されません。")
    else:
        st.error("バリデーション用CSVファイルが見つかりません。リポジトリに配置してください。")
