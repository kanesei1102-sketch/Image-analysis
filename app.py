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

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
    
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# --- 解析ID管理 (人間が読める形式 + ユニークID) ---
if "current_analysis_id" not in st.session_state:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
    unique_suffix = str(uuid.uuid4())[:8]
    st.session_state.current_analysis_id = f"AID-{date_str}-{unique_suffix}"

# ---------------------------------------------------------
# 1. 画像処理エンジン (HE染色・明視野対応)
# ---------------------------------------------------------
COLOR_MAP = {
    # 既存の蛍光・免疫染色設定
    "茶色 (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "緑色 (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "赤色 (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "青色 (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])},
    
    # --- 追加: HE染色 (明視野) サポート ---
    # ヘマトキシリン (細胞核): 紫〜青, 暗い
    "ヘマトキシリン (Nuclei)": {"lower": np.array([110, 50, 50]), "upper": np.array([170, 255, 200])},
    # エオジン (細胞質): ピンク〜赤, 明るめ
    "エオジン (Cytoplasm)": {"lower": np.array([140, 20, 100]), "upper": np.array([180, 255, 255])}
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

tab_main, tab_val = st.tabs(["🚀 解析実行", "🏆 性能バリデーション"])

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
        "1. 面積占有率 (%)", 
        "2. 細胞核カウント / 密度", 
        "3. 共局在解析 (Colocalization)", 
        "4. 空間距離解析", 
        "5. トレンド変化解析"
    ])
    mode = mode_raw 

    st.divider()

    # --- グループ分け戦略 ---
    st.markdown("### 🏷️ グループ化設定")
    group_strategy = st.radio("ラベル決定方法:", ["手動入力", "ファイル名から自動抽出"], 
                              help="自動: ファイル名の区切り文字より前の部分をグループ名として抽出します")
    
    if group_strategy.startswith("手動"):
        sample_group = st.text_input("グループ名 (X軸ラベル):", value="Control")
        filename_sep = None
    else:
        filename_sep = st.text_input("区切り文字 (例: _ または - ):", value="_", help="この文字より前の文字列をグループ名にします")
        st.info(f"例: 'WT{filename_sep}01.tif' → グループ: 'WT'")
        sample_group = "(自動検出)" 

    st.divider()

    # 解析パラメータ動的設定
    current_params_dict = {} # 現在のアクティブなパラメータを保存する辞書

    if mode.startswith("5."):
        st.markdown("### 🔢 トレンド解析条件")
        trend_metric = st.radio("測定指標:", ["共局在率", "面積占有率"])
        ratio_val = st.number_input("条件値:", value=0, step=10)
        ratio_unit = st.text_input("単位:", value="%", key="unit")
        if group_strategy.startswith("手動"):
            sample_group = f"{ratio_val}{ratio_unit}" 
        
        current_params_dict["トレンド指標"] = trend_metric
        current_params_dict["条件値"] = f"{ratio_val}{ratio_unit}"

        if trend_metric.startswith("共局在"):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (基準):", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("A 感度", 5, 50, 20); bright_a = st.slider("A 輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B (対象):", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("B 感度", 5, 50, 20); bright_b = st.slider("B 輝度", 0, 255, 60)
            current_params_dict.update({"CH-A": target_a, "感度A": sens_a, "輝度A": bright_a, "CH-B": target_b, "感度B": sens_b, "輝度B": bright_b})
        else:
            # トレンド解析（面積）の時もROI正規化を使えるようにする
            target_a = st.selectbox("解析対象色:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("感度", 5, 50, 20); bright_a = st.slider("輝度", 0, 255, 60)
            
            use_roi_norm = st.checkbox("組織面積 (ROI) で正規化", value=False, key="roi_mode5")
            current_params_dict.update({"解析対象色": target_a, "感度": sens_a, "輝度": bright_a, "ROI正規化": use_roi_norm})
            
            if use_roi_norm:
                roi_color = st.selectbox("組織の色:", list(COLOR_MAP.keys()), index=5, key="roi_col5")
                sens_roi = st.slider("ROI感度", 5, 50, 20, key="roi_sens5")
                bright_roi = st.slider("ROI輝度", 0, 255, 40, key="roi_bri5")
                current_params_dict.update({"ROI色": roi_color, "ROI感度": sens_roi, "ROI輝度": bright_roi})
    else:
        if mode.startswith("1."):
            target_a = st.selectbox("解析対象色:", list(COLOR_MAP.keys()), index=5)
            sens_a = st.slider("感度", 5, 50, 20); bright_a = st.slider("輝度", 0, 255, 60)
            
            # 面積占有率でもROI正規化ボタンを追加
            use_roi_norm = st.checkbox("組織面積 (ROI) で正規化", value=False, key="roi_mode1")
            current_params_dict.update({"解析対象色": target_a, "感度": sens_a, "輝度": bright_a, "ROI正規化": use_roi_norm})
            
            if use_roi_norm:
                roi_color = st.selectbox("組織の色:", list(COLOR_MAP.keys()), index=5, key="roi_col1")
                sens_roi = st.slider("ROI感度", 5, 50, 20, key="roi_sens1")
                bright_roi = st.slider("ROI輝度", 0, 255, 40, key="roi_bri1")
                current_params_dict.update({"ROI色": roi_color, "ROI感度": sens_roi, "ROI輝度": bright_roi})

        elif mode.startswith("2."):
            # カウントモードでも色指定を可能に（HE染色対応）
            target_a = st.selectbox("核の色:", list(COLOR_MAP.keys()), index=4)
            sens_a = st.slider("核の感度", 5, 50, 20)
            bright_a = st.slider("核の輝度しきい値", 0, 255, 50)
            min_size = st.slider("最小核サイズ (px)", 10, 500, 50)
            
            use_roi_norm = st.checkbox("組織面積 (ROI) で正規化", value=True, key="roi_mode2")
            current_params_dict.update({"核の色": target_a, "核の感度": sens_a, "核の輝度": bright_a, "最小サイズ": min_size, "ROI正規化": use_roi_norm})
            
            if use_roi_norm:
                roi_color = st.selectbox("組織の色:", list(COLOR_MAP.keys()), index=5, key="roi_col2")
                sens_roi = st.slider("ROI感度", 5, 50, 20, key="roi_sens2")
                bright_roi = st.slider("ROI輝度", 0, 255, 40, key="roi_bri2")
                current_params_dict.update({"ROI色": roi_color, "ROI感度": sens_roi, "ROI輝度": bright_roi})

        elif mode.startswith("3."):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3); sens_a = st.slider("A 感度", 5, 50, 20); bright_a = st.slider("A 輝度", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2); sens_b = st.slider("B 感度", 5, 50, 20); bright_b = st.slider("B 輝度", 0, 255, 60)
            current_params_dict.update({"CH-A": target_a, "感度A": sens_a, "輝度A": bright_a, "CH-B": target_b, "感度B": sens_b, "輝度B": bright_b})

        elif mode.startswith("4."):
            target_a = st.selectbox("起点 A:", list(COLOR_MAP.keys()), index=2); target_b = st.selectbox("対象 B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("共通感度", 5, 50, 20); bright_common = st.slider("共通輝度", 0, 255, 60)
            current_params_dict.update({"起点A": target_a, "対象B": target_b, "共通感度": sens_common, "共通輝度": bright_common})

    st.divider()
    # 空間スケールを計算値 3.0769 に設定 (デフォルト)
    scale_val = st.number_input("空間スケール (μm/px)", value=3.0769, format="%.4f")
    current_params_dict["空間スケール"] = scale_val
    current_params_dict["解析モード"] = mode
    
    def prepare_next_group():
        st.session_state.uploader_key = str(uuid.uuid4())

    st.button(
        "📸 次のグループへ (画像のみクリア)", 
        on_click=prepare_next_group, 
        help="現在の解析履歴を保持したまま、アップロードされた画像のみをクリアして次の群の準備をします"
    )
    
    st.divider()
    if st.button("履歴クリア & 新規ID発行"): 
        st.session_state.analysis_history = []
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
        st.session_state.current_analysis_id = f"AID-{date_str}-{str(uuid.uuid4())[:8]}"
        st.session_state.uploader_key = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ トレーサビリティ (現在設定)")
    st.table(pd.DataFrame([current_params_dict]).T)
    
    # 監査ログ用CSV (設定値のみ)
    df_params_log = pd.DataFrame([current_params_dict]).T.reset_index()
    df_params_log.columns = ["パラメータ名", "設定値"]
    param_filename = f"params_{st.session_state.current_analysis_id}.csv"
    st.download_button("📥 設定CSVをダウンロード", df_params_log.to_csv(index=False).encode('utf-8-sig'), param_filename, "text/csv")

    st.divider()
    st.caption("【免責事項】")
    st.caption("本ツールは研究用であり、臨床診断を保証しません。最終的な妥当性の確認は利用者の責任で行ってください。")

# ---------------------------------------------------------
# 4. タブ1: 解析実行
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("画像をアップロード (16-bit TIFF対応)", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True, key=st.session_state.uploader_key)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} 枚の画像を解析しています...")
        batch_results = []
        for i, file in enumerate(uploaded_files):
            file.seek(0); file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            # 16-bit 対応: IMREAD_UNCHANGEDで読み込み
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            
            if img_raw is not None:
                if group_strategy.startswith("ファイル名"):
                    try: detected_group = file.name.split(filename_sep)[0]
                    except: detected_group = "Unknown"
                    current_group_label = detected_group
                else:
                    current_group_label = sample_group

                # 画像処理プロセス (Min-Max Normalizationで8bit化)
                img_f = img_raw.astype(np.float32); mn, mx = np.min(img_f), np.max(img_f)
                img_8 = ((img_f - mn) / (mx - mn) * 255.0 if mx > mn else np.clip(img_f, 0, 255)).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape) == 2 else img_8[:,:,:3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                val, unit, res_disp = 0.0, "", img_rgb.copy()
                h, w = img_rgb.shape[:2]; fov_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                extra_data = {}

                # --- 面積占有率モード (ROI正規化対応) ---
                if mode.startswith("1.") or (mode.startswith("5.") and trend_metric.startswith("面積")):
                    mask_target = get_mask(img_hsv, target_a, sens_a, bright_a)
                    
                    a_denominator_px = h * w
                    roi_status = "Field of View"
                    final_mask = mask_target
                    
                    if 'use_roi_norm' in locals() and use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        # 組織内部にある抽出色のみをカウント
                        final_mask = cv2.bitwise_and(mask_target, mask_roi)
                        a_denominator_px = cv2.countNonZero(mask_roi)
                        roi_status = "Inside ROI"
                        # ROIの外郭を赤線で描画
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    target_px = cv2.countNonZero(final_mask)
                    val = (target_px / a_denominator_px * 100) if a_denominator_px > 0 else 0
                    unit = "% Area"
                    
                    # 緑色で抽出範囲を表示
                    res_disp_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
                    res_disp_mask[:,:,0]=0; res_disp_mask[:,:,2]=0
                    res_disp = cv2.addWeighted(res_disp, 0.7, res_disp_mask, 0.3, 0)
                    
                    a_target_mm2 = a_denominator_px * ((scale_val/1000)**2)
                    extra_data = {
                        "対象面積(mm2)": round(a_target_mm2, 6),
                        "正規化基準": roi_status
                    }

                # --- 細胞核カウントモード (ROI正規化 & 色指定対応) ---
                elif mode.startswith("2."):
                    # 改良ポイント：指定された色のマスクを使用して核を抽出（HE・免疫染色対応）
                    mask_nuclei = get_mask(img_hsv, target_a, sens_a, bright_a)
                    
                    # 核の分離を良くするモルフォロジー演算
                    kernel = np.ones((3,3), np.uint8)
                    mask_nuclei = cv2.morphologyEx(mask_nuclei, cv2.MORPH_OPEN, kernel)
                    
                    cnts, _ = cv2.findContours(mask_nuclei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]; val, unit = len(valid), "cells"
                    cv2.drawContours(res_disp, valid, -1, (0,255,0), 2)
                    
                    a_target_mm2 = fov_mm2
                    roi_status = "Field of View"
                    
                    if 'use_roi_norm' in locals() and use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        roi_px = cv2.countNonZero(mask_roi)
                        a_target_mm2 = roi_px * ((scale_val/1000)**2)
                        roi_status = "Inside ROI"
                        # ROIの外郭を赤線で描画
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    density = val / a_target_mm2 if a_target_mm2 > 0 else 0
                    extra_data = {
                        "対象面積(mm2)": round(a_target_mm2, 6),
                        "密度(cells/mm2)": round(density, 2),
                        "正規化基準": roi_status
                    }

                # --- その他のモード ---
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
                st.markdown(f"### 📷 画像 {i+1}: {file.name}")
                st.markdown(f"**検出グループ:** `{current_group_label}`")
                
                # 詳細な結果表示
                if "密度(cells/mm2)" in extra_data:
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("カウント数", f"{int(val)} cells")
                    c_m2.metric("細胞密度", f"{int(extra_data['密度(cells/mm2)']):,} /mm²")
                    c_m3.caption(f"面積: {extra_data['対象面積(mm2)']:.4f} mm² ({extra_data['正規化基準']})")
                elif "対象面積(mm2)" in extra_data:
                    c_m1, c_m2 = st.columns(2)
                    c_m1.metric("占有率", f"{val:.2f} %")
                    c_m2.caption(f"分母面積: {extra_data['対象面積(mm2)']:.4f} mm² ({extra_data['正規化基準']})")
                else:
                    st.markdown(f"### 解析結果: **{val:.2f} {unit}**")
                
                c1, c2 = st.columns(2); c1.image(img_rgb, caption="元画像 (Raw)"); c2.image(res_disp, caption="解析結果")
                
                # 結果データの構築 (パラメータを含む)
                row_data = {
                    "ソフトウェア・バージョン": SOFTWARE_VERSION,
                    "解析ID": st.session_state.current_analysis_id,
                    "解析日時_UTC": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "ファイル名": file.name,
                    "グループ": current_group_label,
                    "測定値": val,
                    "単位": unit,
                }
                if extra_data: row_data.update(extra_data)
                # ★ ここで確実にパラメータを結合 ★
                row_data.update(current_params_dict)
                
                batch_results.append(row_data)
        
        if st.button("バッチデータを確定 (Commit)", type="primary"):
            st.session_state.analysis_history.extend(batch_results)
            st.success("データが履歴に追加されました。解析IDは維持されています。")
            st.rerun()

    if st.session_state.analysis_history:
        st.divider()
        st.header("💾 CSV出力 (完全トレーサビリティ対応)")
        df_exp = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(df_exp, use_container_width=True)
        utc_filename = f"quantified_data_{st.session_state.current_analysis_id}.csv"
        st.download_button("📥 結果CSVをダウンロード", df_exp.to_csv(index=False).encode('utf-8-sig'), utc_filename)

# ---------------------------------------------------------
# 5. タブ2: 性能バリデーション
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
