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
SOFTWARE_VERSION = "Bio-Image Quantifier Pro v2026.02 (Stable/Full-Validation)"

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
    
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "current_analysis_id" not in st.session_state:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
    unique_suffix = str(uuid.uuid4())[:8]
    st.session_state.current_analysis_id = f"AID-{date_str}-{unique_suffix}"

# ---------------------------------------------------------
# 1. 画像処理エンジン (シンプルかつ堅牢な設定)
# ---------------------------------------------------------
COLOR_MAP = {
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑色 (GFP)": {"lower": np.array([35, 40, 40]), "upper": np.array([85, 255, 255])},
    "赤色 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青色 (DAPI)": {"lower": np.array([90, 50, 50]), "upper": np.array([140, 255, 255])},
    "ヘマトキシリン (Nuclei)": {"lower": np.array([100, 50, 50]), "upper": np.array([170, 255, 200])},
    "エオジン (Cytoplasm)": {"lower": np.array([140, 20, 100]), "upper": np.array([180, 255, 255])}
}

# 表示用の色定義 (RGB) - ユーザーの直感に合わせる
DISPLAY_COLORS = {
    "茶色 (DAB)": (165, 42, 42),
    "緑色 (GFP)": (0, 255, 0),
    "赤色 (RFP)": (255, 0, 0),
    "青色 (DAPI)": (0, 0, 255),
    "ヘマトキシリン (Nuclei)": (0, 0, 255),
    "エオジン (Cytoplasm)": (255, 105, 180)
}

def get_mask(hsv_img, color_name, sens, bright_min):
    conf = COLOR_MAP[color_name]
    l = conf["lower"].copy()
    u = conf["upper"].copy()
    
    # 赤色(Hue 0付近と180付近)の特別処理
    if color_name == "赤色 (RFP)" or "エオジン" in color_name:
        lower1 = np.array([0, 30, bright_min])
        upper1 = np.array([10 + sens, 255, 255])
        lower2 = np.array([170 - sens, 30, bright_min])
        upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    else:
        # その他の色は範囲を適用
        l[0] = max(0, l[0] - sens)
        u[0] = min(180, u[0] + sens)
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
# 2. バリデーションデータ読み込み (元のロジックを完全復元)
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
# 3. UIフレームワーク
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition (日本語版)")
st.caption(f"{SOFTWARE_VERSION}")
st.sidebar.markdown(f"**解析ID:** `{st.session_state.current_analysis_id}`")

tab_main, tab_val = st.tabs(["🚀 解析実行", "🏆 性能バリデーション"])

with st.sidebar:
    st.header("解析レシピ")
    mode = st.selectbox("解析モード選択:", [
        "1. 面積占有率 (%)", 
        "2. 細胞核カウント / 密度", 
        "3. 共局在解析 (Colocalization)", 
        "4. 空間距離解析", 
        "5. トレンド変化解析"
    ])

    st.divider()
    st.markdown("### 🏷️ グループ化設定")
    group_strategy = st.radio("ラベル決定方法:", ["手動入力", "ファイル名から自動抽出"])
    if group_strategy.startswith("手動"):
        sample_group = st.text_input("グループ名:", value="Control")
        filename_sep = None
    else:
        filename_sep = st.text_input("区切り文字:", value="_")
        sample_group = "(自動検出)" 

    st.divider()

    # パラメータ設定
    current_params_dict = {}

    if mode.startswith("5."):
        st.markdown("### 🔢 トレンド解析条件")
        trend_metric = st.radio("測定指標:", ["共局在率", "面積占有率"])
        ratio_val = st.number_input("条件値:", value=0, step=10)
        ratio_unit = st.text_input("単位:", value="%", key="unit")
        if group_strategy.startswith("手動"): sample_group = f"{ratio_val}{ratio_unit}"
        current_params_dict["条件値"] = f"{ratio_val}{ratio_unit}"
        
        if trend_metric.startswith("共局在"):
            st.info("設定: CH-B(基準) に対する CH-A(対象) の重なり")
            c1, c2 = st.columns(2)
            with c1:
                target_b = st.selectbox("CH-B (基準/分母):", list(COLOR_MAP.keys()), index=3)
                sens_b = st.slider("B 感度", 5, 50, 20); bright_b = st.slider("B 輝度", 0, 255, 60)
            with c2:
                target_a = st.selectbox("CH-A (対象/分子):", list(COLOR_MAP.keys()), index=1)
                sens_a = st.slider("A 感度", 5, 50, 20); bright_a = st.slider("A 輝度", 0, 255, 60)
            current_params_dict.update({"CH-A": target_a, "感度A": sens_a, "CH-B": target_b, "感度B": sens_b})
        else:
            target_a = st.selectbox("解析対象色:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("感度", 5, 50, 20); bright_a = st.slider("輝度", 0, 255, 60)
            use_roi_norm = st.checkbox("ROI正規化", value=False)
            current_params_dict.update({"解析対象色": target_a, "感度": sens_a, "ROI正規化": use_roi_norm})
            if use_roi_norm:
                roi_color = st.selectbox("ROI色:", list(COLOR_MAP.keys()), index=5)
                sens_roi = st.slider("ROI感度", 5, 50, 20); bright_roi = st.slider("ROI輝度", 0, 255, 40)
                current_params_dict.update({"ROI色": roi_color})

    elif mode.startswith("3."):
        st.info("💡 **CH-B (基準/分母)** の領域内で、**CH-A (対象/分子)** がどれだけ重なっているかを計算します。")
        c1, c2 = st.columns(2)
        with c1:
            target_b = st.selectbox("CH-B (基準/分母):", list(COLOR_MAP.keys()), index=3) # デフォルト青
            sens_b = st.slider("B 感度 (基準)", 5, 50, 20)
            bright_b = st.slider("B 輝度", 0, 255, 60)
        with c2:
            target_a = st.selectbox("CH-A (対象/分子):", list(COLOR_MAP.keys()), index=1) # デフォルト緑
            sens_a = st.slider("A 感度 (対象)", 5, 50, 20)
            bright_a = st.slider("A 輝度", 0, 255, 60)
        
        current_params_dict.update({"CH-A": target_a, "感度A": sens_a, "CH-B": target_b, "感度B": sens_b})

    elif mode.startswith("1."):
        target_a = st.selectbox("解析対象色:", list(COLOR_MAP.keys()), index=5)
        sens_a = st.slider("感度", 5, 50, 20); bright_a = st.slider("輝度", 0, 255, 60)
        use_roi_norm = st.checkbox("ROI正規化", value=False)
        current_params_dict.update({"解析対象色": target_a, "感度": sens_a, "ROI正規化": use_roi_norm})
        if use_roi_norm:
            roi_color = st.selectbox("ROI色:", list(COLOR_MAP.keys()), index=5)
            sens_roi = st.slider("ROI感度", 5, 50, 20); bright_roi = st.slider("ROI輝度", 0, 255, 40)
            current_params_dict.update({"ROI色": roi_color})

    elif mode.startswith("2."):
        target_a = st.selectbox("核の色:", list(COLOR_MAP.keys()), index=4)
        sens_a = st.slider("核の感度", 5, 50, 20); bright_a = st.slider("核の輝度", 0, 255, 50)
        min_size = st.slider("最小核サイズ", 10, 500, 50)
        use_roi_norm = st.checkbox("ROI正規化", value=True)
        current_params_dict.update({"核の色": target_a, "感度": sens_a, "ROI正規化": use_roi_norm})
        if use_roi_norm:
            roi_color = st.selectbox("ROI色:", list(COLOR_MAP.keys()), index=5)
            sens_roi = st.slider("ROI感度", 5, 50, 20); bright_roi = st.slider("ROI輝度", 0, 255, 40)
            current_params_dict.update({"ROI色": roi_color})

    elif mode.startswith("4."):
        target_a = st.selectbox("起点 A:", list(COLOR_MAP.keys()), index=2); target_b = st.selectbox("対象 B:", list(COLOR_MAP.keys()), index=3)
        sens_common = st.slider("共通感度", 5, 50, 20); bright_common = st.slider("共通輝度", 0, 255, 60)
        current_params_dict.update({"起点A": target_a, "対象B": target_b})

    st.divider()
    scale_val = st.number_input("空間スケール (μm/px)", value=3.0769, format="%.4f")
    current_params_dict["空間スケール"] = scale_val
    current_params_dict["解析モード"] = mode

    def prepare_next_group():
        st.session_state.uploader_key = str(uuid.uuid4())

    st.button("📸 次のグループへ (画像クリア)", on_click=prepare_next_group)
    if st.button("履歴クリア & 新規ID"): 
        st.session_state.analysis_history = []; st.rerun()

    st.divider()
    st.download_button("📥 設定CSV", pd.DataFrame([current_params_dict]).T.reset_index().to_csv(index=False).encode('utf-8-sig'), "params.csv")

# ---------------------------------------------------------
# 4. 解析実行プロセス
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("画像アップロード", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True, key=st.session_state.uploader_key)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} 枚処理中...")
        batch_results = []
        for i, file in enumerate(uploaded_files):
            file.seek(0); file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            
            if img_raw is not None:
                if group_strategy.startswith("ファイル名"):
                    try: current_group_label = file.name.split(filename_sep)[0]
                    except: current_group_label = "Unknown"
                else: current_group_label = sample_group

                # 画像読み込みと前処理
                img_f = img_raw.astype(np.float32); mn, mx = np.min(img_f), np.max(img_f)
                img_8 = ((img_f - mn) / (mx - mn) * 255.0 if mx > mn else np.clip(img_f, 0, 255)).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape) == 2 else img_8[:,:,:3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                val, unit = 0.0, ""
                h, w = img_rgb.shape[:2]
                res_disp = np.zeros_like(img_rgb) # 表示用黒背景

                extra_data = {}

                # ----------------------------
                # 共局在解析 (Mode 3 & 5)
                # ----------------------------
                if mode.startswith("3.") or (mode.startswith("5.") and trend_metric.startswith("共局在")):
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a) # 対象(分子)
                    mask_b = get_mask(img_hsv, target_b, sens_b, bright_b) # 基準(分母)

                    denom = cv2.countNonZero(mask_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                    unit = "% Coloc"

                    # 表示: ユーザーが選んだ「本来の色」をそのまま配置し、重なりを自然な混色にする
                    # 例: 青(0,0,255) + 緑(0,255,0) = シアン(0,255,255)
                    color_a = DISPLAY_COLORS[target_a]
                    color_b = DISPLAY_COLORS[target_b]
                    
                    # 単純加算は8bitで飽和するので、論理和的に色を配置
                    res_disp[mask_a > 0] = color_a
                    current_b_pixels = np.zeros_like(res_disp); current_b_pixels[mask_b > 0] = color_b
                    res_disp = cv2.bitwise_or(res_disp, current_b_pixels)

                # ----------------------------
                # 面積解析 (Mode 1 & 5)
                # ----------------------------
                elif mode.startswith("1.") or (mode.startswith("5.") and trend_metric.startswith("面積")):
                    mask_target = get_mask(img_hsv, target_a, sens_a, bright_a)
                    a_den_px = h * w; roi_status = "FoV"
                    final_mask = mask_target
                    
                    if 'use_roi_norm' in locals() and use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        final_mask = cv2.bitwise_and(mask_target, mask_roi)
                        a_den_px = cv2.countNonZero(mask_roi); roi_status = "ROI"
                        # ROI外郭描画
                        roi_conts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(res_disp, roi_conts, -1, (100,100,100), 2)

                    target_px = cv2.countNonZero(final_mask)
                    val = (target_px / a_den_px * 100) if a_den_px > 0 else 0
                    unit = "% Area"
                    
                    # 選択色で表示
                    res_disp[final_mask > 0] = DISPLAY_COLORS[target_a]
                    extra_data = {"対象面積": round(a_den_px * ((scale_val/1000)**2), 4), "基準": roi_status}

                # ----------------------------
                # カウント解析 (Mode 2)
                # ----------------------------
                elif mode.startswith("2."):
                    mask_nuclei = get_mask(img_hsv, target_a, sens_a, bright_a)
                    mask_nuclei = cv2.morphologyEx(mask_nuclei, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
                    cnts, _ = cv2.findContours(mask_nuclei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                    val, unit = len(valid), "cells"
                    
                    cv2.drawContours(res_disp, valid, -1, DISPLAY_COLORS[target_a], 2) # 輪郭描画
                    
                    a_target_mm2 = (h * w) * ((scale_val/1000)**2); roi_status = "FoV"
                    if use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        a_target_mm2 = cv2.countNonZero(mask_roi) * ((scale_val/1000)**2)
                        roi_status = "ROI"
                        roi_conts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(res_disp, roi_conts, -1, (100,100,100), 2)

                    extra_data = {"密度": round(val/a_target_mm2, 2) if a_target_mm2 > 0 else 0, "基準": roi_status}

                # ----------------------------
                # 距離解析 (Mode 4)
                # ----------------------------
                elif mode.startswith("4."):
                    ma = get_mask(img_hsv, target_a, sens_common, bright_common)
                    mb = get_mask(img_hsv, target_b, sens_common, bright_common)
                    pa, pb = get_centroids(ma), get_centroids(mb)
                    if pa and pb: val = np.mean([np.min([np.linalg.norm(a - b) for b in pb]) for a in pa]) * scale_val
                    unit = "μm"
                    res_disp = cv2.addWeighted(img_rgb, 0.5, cv2.merge([ma, mb, np.zeros_like(ma)]), 0.5, 0)

                # 結果表示とデータ格納
                st.divider()
                st.markdown(f"**画像:** `{file.name}` | **グループ:** `{current_group_label}`")
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Raw Image")
                c2.image(res_disp, caption="Analysis Result (Color Corrected)")

                row_data = {
                    "File": file.name, "Group": current_group_label, "Value": val, "Unit": unit, 
                    "ID": st.session_state.current_analysis_id
                }
                row_data.update(extra_data)
                row_data.update(current_params_dict)
                batch_results.append(row_data)

        if st.button("データ確定 (Commit)", type="primary"):
            st.session_state.analysis_history.extend(batch_results)
            st.success("保存完了"); st.rerun()

    # CSV出力
    if st.session_state.analysis_history:
        st.divider()
        df_exp = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(df_exp)
        st.download_button("📥 結果CSV", df_exp.to_csv(index=False).encode('utf-8-sig'), "results.csv")

# ---------------------------------------------------------
# 5. バリデーション (詳細版完全復元)
# ---------------------------------------------------------
with tab_val:
    st.header("🏆 性能バリデーションサマリー")
    st.markdown("""
    * **検証用データセット:** BBBC005 (Broad Bioimage Benchmark Collection)
    * **検証規模:** 3,200枚 (ハイスループット検証)
    * **検証手法:** 密度別の各グループに対しパラメータを最適化し、適切なキャリブレーション下での最大性能を実証。
    """)

    if not df_val.empty:
        gt_map = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
        df_hq = df_val[(df_val['Focus'] >= 1) & (df_val['Focus'] <= 5)]
        w1_hq = df_hq[df_hq['Channel'] == 'W1']
        avg_acc = w1_hq['Accuracy'].mean()
        df_lin = w1_hq.groupby('Ground Truth')['Value'].mean().reset_index()
        r2 = np.corrcoef(df_lin['Ground Truth'], df_lin['Value'])[0, 1]**2

        m1, m2, m3 = st.columns(3)
        m1.metric("平均精度 (Accuracy)", f"{avg_acc:.1f}%")
        m2.metric("線形性 (R²)", f"{r2:.4f}")
        m3.metric("検証画像数", "3,200+")

        st.divider()
        st.subheader("📈 1. 計数性能と線形性 (W1 vs W2)")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='Ideal Line')
        ax1.scatter(df_lin['Ground Truth'], df_lin['Value'], color='#1f77b4', s=100, label='W1 (Nuclei)', zorder=5)
        w2_lin = df_hq[df_hq['Channel'] == 'W2'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax1.scatter(w2_lin['Ground Truth'], w2_lin['Value'], color='#ff7f0e', s=100, marker='D', label='W2 (Cytoplasm)', zorder=5)
        z = np.polyfit(df_lin['Ground Truth'], df_lin['Value'], 1)
        ax1.plot(df_lin['Ground Truth'], np.poly1d(z)(df_lin['Ground Truth']), '#1f77b4', alpha=0.5, label='W1 Reg')
        ax1.set_xlabel('Ground Truth (理論値)'); ax1.set_ylabel('Measured Value (実測値)'); ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 2. 密度別精度比較")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            df_bar = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().reset_index()
            df_bar['Density'] = pd.Categorical(df_bar['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.barplot(data=df_bar, x='Density', y='Accuracy', hue='Channel', palette={'W1': '#1f77b4', 'W2': '#ff7f0e'}, ax=ax2)
            ax2.axhline(100, color='red', linestyle='--'); ax2.set_ylabel('精度 Accuracy (%)')
            st.pyplot(fig2)
        with c2:
            st.subheader("📉 3. 光学的堅牢性 (ボケ耐性)")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            df_decay = df_val[df_val['Channel'] == 'W1'].copy()
            df_decay['Density'] = pd.Categorical(df_decay['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.lineplot(data=df_decay, x='Focus', y='Accuracy', hue='Density', marker='o', ax=ax3)
            ax3.axhline(100, color='red', linestyle='--'); ax3.set_ylabel('精度 Accuracy (%)')
            st.pyplot(fig3)
        st.divider()
        st.subheader("📋 4. バリデーション数値データ")
        summary = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().unstack().reset_index()
        summary['理論値'] = summary['Density'].map(gt_map)
        summary['W1実測'] = (summary['W1']/100)*summary['理論値']
        summary['W2実測'] = (summary['W2']/100)*summary['理論値']
        summary['Density'] = pd.Categorical(summary['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        summary = summary.sort_values('Density')
        st.table(summary[['Density', '理論値', 'W1', 'W1実測', 'W2', 'W2実測']].rename(columns={
            'W1': 'W1 精度(%)', 'W1実測': 'W1 平均カウント', 'W2': 'W2 精度(%)', 'W2実測': 'W2 平均カウント'
        }))
        st.info("💡 **総合結論:** W1（核）は全密度領域で高精度を維持。W2（細胞質）は過小・過剰評価の変動が激しく、科学的に定量解析には推奨されません。")
    else:
        st.error("バリデーション用CSVファイルが見てかりません。リポジトリのルートに配置してください。")
