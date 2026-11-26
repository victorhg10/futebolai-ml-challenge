from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "futebolai_model.joblib"

st.set_page_config(page_title="FutebolAI", page_icon="⚽")
st.title("⚽ FutebolAI")
st.subheader("Previsão do resultado de uma partida")
st.write("Informe os times e as odds pré-jogo para estimar o resultado mais provável. A previsão é experimental e não garante o placar.")

if not MODEL_PATH.exists():
    st.warning("O modelo ainda não foi treinado. Execute primeiro: py train.py")
    st.stop()

model = joblib.load(MODEL_PATH)
encoder = model.named_steps["preprocessamento"].named_transformers_["times"].named_steps["onehot"]
teams = sorted(set(encoder.categories_[0]) | set(encoder.categories_[1]))
home = st.selectbox("Time mandante", teams)
away_options = [t for t in teams if t != home]
away = st.selectbox("Time visitante", away_options)
col1, col2, col3 = st.columns(3)
with col1:
    odd_home = st.number_input("Odd vitória mandante", min_value=1.01, max_value=50.0, value=2.00, step=0.05)
with col2:
    odd_draw = st.number_input("Odd empate", min_value=1.01, max_value=50.0, value=3.40, step=0.05)
with col3:
    odd_away = st.number_input("Odd vitória visitante", min_value=1.01, max_value=50.0, value=3.50, step=0.05)

if st.button("Analisar partida"):
    sample = pd.DataFrame([{
        "time_mandante": home,
        "time_visitante": away,
        "odd_vitoria_mandante": odd_home,
        "odd_empate": odd_draw,
        "odd_vitoria_visitante": odd_away
    }])
    pred = model.predict(sample)[0]
    labels = {"H": "Vitória do mandante", "D": "Empate", "A": "Vitória do visitante"}
    st.success(f"Resultado estimado: **{labels.get(pred, pred)}**")
    if hasattr(model, "predict_proba"):
        st.write("Probabilidades estimadas pelo modelo:")
        st.bar_chart(pd.DataFrame({"Probabilidade": model.predict_proba(sample)[0]}, index=[labels.get(c, c) for c in model.classes_]))
