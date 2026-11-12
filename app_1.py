"""고등학교 인공지능 수학 2차 발표: 분자 유사도 탐색 프로그램.
실행: python -m streamlit run app.py
주의: 구조 유사도 검색일 뿐, 약효나 안전성을 예측하지 않습니다.
"""
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, MACCSkeys, rdFingerprintGenerator

# 시연용 소규모 분자 목록: 각 이름에 대응하는 구조를 SMILES로 저장
MOLECULES = {
    "아스피린": "CC(=O)Oc1ccccc1C(=O)O",
    "살리실산": "O=C(O)c1ccccc1O",
    "메틸 살리실레이트": "COC(=O)c1ccccc1O",
    "벤조산": "O=C(O)c1ccccc1",
    "아세트아미노펜": "CC(=O)Nc1ccc(O)cc1",
    "이부프로펜": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "나프록센": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "카페인": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "페놀": "Oc1ccccc1",
    "에탄올": "CCO",
    "다이메틸 에터": "COC",
    "아세트산": "CC(=O)O",
    "에틸 아세테이트": "CCOC(=O)C",
}

@st.cache_resource
def fingerprint_generators():
    return {
        "Morgan 지문 (2048비트, 반경 2)": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048),
    }

def fingerprint(mol, method):
    if method.startswith("MACCS"):
        return MACCSkeys.GenMACCSKeys(mol)
    return fingerprint_generators()["Morgan 지문 (2048비트, 반경 2)"].GetFingerprint(mol)

def jaccard_details(fp_a, fp_b):
    # '1'인 위치의 집합을 직접 만들어 교집합/합집합을 계산한다.
    a = set(fp_a.GetOnBits())
    b = set(fp_b.GetOnBits())
    shared = len(a & b)
    union = len(a | b)
    score = shared / union if union else 1.0
    # RDKit 내부 계산을 이용해 수학식의 결과를 검증한다.
    library_score = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    assert abs(score - library_score) < 1e-10
    return shared, union, score

st.set_page_config(page_title="분자 유사도 탐색기", page_icon="🧪", layout="wide")
st.title("🧪 분자 구조 유사도 탐색기")
st.caption("인공지능 수학 · 분자 지문(이진 벡터) · 자카드/타니모토 유사도")
st.info("이 프로그램은 입력한 화합물과 **구조가 유사한 분자**를 작은 예제 목록에서 찾습니다. 약효·독성·치료 효과를 예측하는 AI는 아닙니다.")

left, right = st.columns([2, 1])
with left:
    mode = st.radio("입력 방법", ["예제 물질 선택", "SMILES 직접 입력"], horizontal=True)
    if mode == "예제 물질 선택":
        chosen = st.selectbox("기준 분자", list(MOLECULES), index=0)
        query_smiles = MOLECULES[chosen]
        st.code(query_smiles, language="text")
    else:
        query_smiles = st.text_input("SMILES 구조식", value=MOLECULES["아스피린"], help="예: 에탄올 CCO, 아스피린 CC(=O)Oc1ccccc1C(=O)O")
    method = st.selectbox("분자 지문 종류", ["Morgan 지문 (2048비트, 반경 2)", "MACCS Keys (167비트)"])
    top_n = st.slider("표시할 결과 수", min_value=3, max_value=len(MOLECULES), value=6)
    include_self = st.checkbox("기준 분자와 동일한 물질도 표시", value=False)

query_mol = Chem.MolFromSmiles(query_smiles) if query_smiles.strip() else None
if query_mol is None:
    st.error("올바른 SMILES가 아닙니다. 구조식을 다시 확인해 주세요.")
    st.stop()

with right:
    st.subheader("입력된 구조")
    st.image(Draw.MolToImage(query_mol, size=(340, 220)))
    st.write(f"분자식: **{Chem.rdMolDescriptors.CalcMolFormula(query_mol)}**")
    st.write(f"분자량: **{Descriptors.MolWt(query_mol):.2f} g/mol**")

query_fp = fingerprint(query_mol, method)
results = []
for name, smiles in MOLECULES.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    if not include_self and Chem.MolToSmiles(query_mol) == Chem.MolToSmiles(mol):
        continue
    candidate_fp = fingerprint(mol, method)
    intersection, union, similarity = jaccard_details(query_fp, candidate_fp)
    results.append({"물질명": name, "SMILES": smiles, "자카드 유사도": similarity,
                    "공통 1비트 수": intersection, "합집합 1비트 수": union, "mol": mol})
results.sort(key=lambda row: (-row["자카드 유사도"], row["물질명"]))
results = results[:top_n]

st.subheader("유사도 검색 결과")
if not results:
    st.warning("표시할 결과가 없습니다.")
    st.stop()

df = pd.DataFrame([{k: row[k] for k in ["물질명", "자카드 유사도", "공통 1비트 수", "합집합 1비트 수", "SMILES"]} for row in results])
st.dataframe(df, hide_index=True, use_container_width=True,
             column_config={"자카드 유사도": st.column_config.NumberColumn(format="%.4f")})
st.bar_chart(df.set_index("물질명")["자카드 유사도"])

st.subheader("상위 후보의 화학 구조")
st.image(Draw.MolsToGridImage([r["mol"] for r in results[:6]],
        legends=[f'{r["물질명"]}: {r["자카드 유사도"]:.3f}' for r in results[:6]],
        molsPerRow=3, subImgSize=(250, 190)))

st.subheader("계산 과정을 확인해 보기")
selected = st.selectbox("비교할 물질", [r["물질명"] for r in results])
r = next(row for row in results if row["물질명"] == selected)
st.latex(r"J(A,B)=\frac{|A\cap B|}{|A\cup B|}")
st.write(f"기준 분자와 **{selected}**의 공통 1비트 수: **{r['공통 1비트 수']}**")
st.write(f"두 분자 중 적어도 하나에서 1인 비트 수: **{r['합집합 1비트 수']}**")
st.success(f"자카드 유사도 = {r['공통 1비트 수']} / {r['합집합 1비트 수']} = **{r['자카드 유사도']:.4f}**")

with st.expander("분자 지문(이진 벡터)의 앞부분 보기"):
    candidate_fp = fingerprint(r["mol"], method)
    st.write("기준 분자:", query_fp.ToBitString()[:120] + " …")
    st.write("비교 분자:", candidate_fp.ToBitString()[:120] + " …")
    st.caption("여기에 보이는 120비트는 일부만 표시한 것입니다. 실제 계산에는 전체 비트를 사용합니다.")

st.caption("자료: RDKit 공식 문서. 표시된 검색 대상은 앱에 저장된 예제 화합물이며, PubChem 전체 데이터베이스를 검색하는 것은 아닙니다.")
