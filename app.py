# File: activity_cliff_app/app.py
import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from cliff_detector import detect_activity_cliffs
from utils import compute_pIC50, smiles_diff_to_images # Updated import
from prompts import generate_prompt, call_llm, generate_fewshot_prompt, generate_rag_prompt, translate_text # Import translate_text
from structure_diff import detect_diff_type
from rdkit import Chem # Import Chem for SMILES validation
from datetime import date # Import date for report generation date

st.set_page_config(page_title="Activity Cliff Analyzer", layout="wide")

# Use @st.cache_resource with a cleanup function (ttl is not strictly needed but good practice)
@st.cache_resource(ttl=3600)
def get_qdrant_client():
    """Initializes and caches the Qdrant client."""
    return QdrantClient(path="/Users/hb/Downloads/activity_cliff_app-2/rag/qdrant")

st.title("🔬 Activity Cliff 자동 탐지 & 해석")

# Clear Cache Button
if st.sidebar.button("캐시 지우기"): # Clear Cache button
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("캐시가 지워졌습니다!")

uploaded_file = st.file_uploader("CSV 업로드 (SMILES, IC50 포함)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터 앞부분:", df.head(2))

    # Validate SMILES entries
    invalid_smiles = []
    if 'SMILES' in df.columns:
        for idx, smiles in df['SMILES'].items():
            if Chem.MolFromSmiles(str(smiles)) is None:
                invalid_smiles.append((idx, smiles))
    else:
        st.error("업로드된 CSV 파일에 'SMILES' 컬럼이 없습니다. 'SMILES' 컬럼이 포함된 파일을 업로드해주세요.")
        st.stop()

    if invalid_smiles:
        st.error("다음 SMILES 항목이 유효하지 않습니다. 파일을 수정하여 다시 업로드해주세요:")
        for idx, smiles in invalid_smiles:
            st.write(f"- 행 {idx}: {smiles}")
        st.stop() # Stop execution if invalid SMILES are found

    if 'pIC50' not in df.columns:
        df['pIC50'] = compute_pIC50(df['IC50'])

    st.sidebar.header("🔧 탐지 설정")
    sim_thres = st.sidebar.slider("구조 유사도 임계값", 0.7, 1.0, 0.85, 0.01)
    act_thres = st.sidebar.slider("pIC50 차이 임계값", 0.5, 3.0, 1.0, 0.1)

    # Replaced checkboxes with radio button for prompt generation strategy
    prompt_strategy = st.sidebar.radio(
        "LLM 해석 요청 방식 선택",
        ("기본", "Few-shot 예시 기반", "RAG 기반 도메인 해석"),
        index=0 # Default to Basic
    )

    # LLM Output Language Selection
    llm_output_lang = st.sidebar.radio(
        "LLM 분석 결과 언어 선택",
        ("한국어", "English"),
        index=0 # Default to Korean
    )

    # Get the cached Qdrant client
    qdrant_client = get_qdrant_client()

    with st.spinner("Activity Cliff 탐지 중..."):
        results = detect_activity_cliffs(df, sim_thres, act_thres)

    st.success(f"{len(results)}개의 Activity Cliff 쌍이 탐지되었습니다.")

    # Download Results Button
    if not results.empty:
        csv_data = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="결과 다운로드 (CSV)",
            data=csv_data,
            file_name="activity_cliffs.csv",
            mime="text/csv",
        )

    for i, row in results.iterrows():
        col1, col2 = st.columns(2)

        # Generate images and highlighted SMILES text
        img1, img2 = smiles_diff_to_images(row['mol1_smiles'], row['mol2_smiles'])

        with col1:
            st.image(img1, caption=f"A: {row['mol1_activity']:.2f}")
            st.markdown(f"SMILES: `{row['mol1_smiles']}`") # Reverted to plain SMILES
        with col2:
            st.image(img2, caption=f"B: {row['mol2_activity']:.2f}")
            st.markdown(f"SMILES: `{row['mol2_smiles']}`") # Reverted to plain SMILES

        st.markdown(f"- 유사도: `{row['sim']:.2f}` / pIC50 차이: `{row['activity_diff']:.2f}`")

        # Added a button to trigger prompt generation and LLM call
        if st.button(f"LLM 해석 생성 - 쌍 {i}", key=f"llm_gen_btn_{i}"):
            system_prompt = ""
            user_prompt = ""
            if prompt_strategy == "RAG 기반 도메인 해석":
                system_prompt, user_prompt = generate_rag_prompt(row, qdrant_client)
            elif prompt_strategy == "Few-shot 예시 기반":
                system_prompt, user_prompt = generate_fewshot_prompt(row)
            else:
                system_prompt, user_prompt = generate_prompt(row)

            # Call LLM with English prompts
            llm_output_raw = call_llm(system_prompt, user_prompt)
            
            # Translate if Korean is selected, otherwise use English
            if llm_output_lang == "한국어":
                llm_output_final = translate_text(llm_output_raw, target_language="Korean")
            else:
                llm_output_final = llm_output_raw

            # Fix line formatting for Markdown display
            llm_output_formatted = llm_output_final.replace("\n", "\n\n")

            # Construct the full report based on the example format
            # Display the two molecules and their info side by side in a single block
            st.markdown("""
< 자동 생성 SAR 요약 리포트 >

**분석 대상:** Janus kinase (JAK) 저해제 후보 화합물
                        
**리포트 생성일:** {report_date}

**핵심 분석 1: 주요 활성 변화 요인 (Key Activity Cliff)**

**요약:** 분자의 특정 3차원 구조가 활성에 {activity_ratio:.0f}배 차이를 유발함.
""".format(
                report_date=date.today().strftime("%Y-%m-%d"),
                activity_ratio=max(row['mol1_activity'], row['mol2_activity']) / min(row['mol1_activity'], row['mol2_activity']),
            ))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
**화합물 ID:** Molecule {row['mol1_idx']} (활성: {row['mol1_activity']:.2f} nM)

**구조:**
""")
                st.image(img1, caption=f"Molecule {row['mol1_idx']}")
            with col_b:
                st.markdown(f"""
**화합물 ID:** Molecule {row['mol2_idx']} (활성: {row['mol2_activity']:.2f} nM)

**구조:**
""")
                st.image(img2, caption=f"Molecule {row['mol2_idx']}")


            st.markdown(f"""
**자동화된 해석 및 가설:**
{llm_output_formatted}
""")

        st.divider() # Add a separator after each pair