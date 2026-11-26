"""Teste de fumaça do app Streamlit usando o harness oficial (AppTest).

Execução:
    .venv\\Scripts\\python tests\\smoke_test_app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
MODEL = ROOT / "models" / "futebolai_model.joblib"
BACKUP = MODEL.with_suffix(".joblib.bak")

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL") + " - " + name + ((" | " + detail) if detail else ""))


def new():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def exceptions(at):
    return [str(e.value) for e in at.exception]


def run():
    # 1. Carga inicial
    at = new()
    check("1. carga inicial sem excecao", len(at.exception) == 0, f"exceptions={exceptions(at)}")
    check("1b. titulo FutebolAI", any("FutebolAI" in t.value for t in at.title),
          f"titles={[t.value for t in at.title]}")
    check("1c. dois selectboxes", len(at.selectbox) == 2, f"n={len(at.selectbox)}")
    check("1d. tres number_inputs", len(at.number_input) == 3, f"n={len(at.number_input)}")
    check("1e. um botao", len(at.button) == 1, f"n={len(at.button)}")

    # 2. Visitante exclui mandante
    home = at.selectbox[0].value
    away_opts = list(at.selectbox[1].options)
    check("2. visitante exclui mandante", home not in away_opts, f"home={home}, n_opcoes={len(away_opts)}")

    # 3. Previsão com valores padrão
    at.button[0].click().run()
    check("3. previsao padrao", len(at.exception) == 0 and len(at.success) >= 1,
          f"success={[s.value for s in at.success]}, exceptions={exceptions(at)}")
    charts = at.get("vega_lite_chart")
    spec = charts[0].proto.spec if charts else ""
    check("3b. grafico de probabilidades", len(charts) >= 1 and "Probabilidade" in spec,
          f"charts={len(charts)}")

    # 4. Favorito mandante
    at = new()
    at.selectbox[0].select("Man City")
    at.selectbox[1].select("Southampton")
    at.number_input[0].set_value(1.30)
    at.number_input[1].set_value(5.50)
    at.number_input[2].set_value(9.00)
    at.button[0].click().run()
    msg = at.success[0].value if at.success else ""
    check("4. favorito mandante", "Vitória do mandante" in msg, msg)

    # 5. Favorito visitante
    at = new()
    at.selectbox[0].select("Southampton")
    at.selectbox[1].select("Man City")
    at.number_input[0].set_value(9.00)
    at.number_input[1].set_value(5.50)
    at.number_input[2].set_value(1.30)
    at.button[0].click().run()
    msg = at.success[0].value if at.success else ""
    check("5. favorito visitante", "Vitória do visitante" in msg, msg)

    # 6. Jogo equilibrado
    at = new()
    at.selectbox[0].select("Arsenal")
    at.selectbox[1].select("Chelsea")
    at.number_input[0].set_value(2.50)
    at.number_input[1].set_value(3.40)
    at.number_input[2].set_value(2.80)
    at.button[0].click().run()
    msg = at.success[0].value if at.success else ""
    check("6. equilibrado preve resultado valido",
          len(at.exception) == 0 and any(x in msg for x in ("Vitória do mandante", "Empate", "Vitória do visitante")),
          msg)

    # 7. Odds nos limites
    at = new()
    at.number_input[0].set_value(1.01)
    at.number_input[1].set_value(50.0)
    at.number_input[2].set_value(50.0)
    at.button[0].click().run()
    check("7. odds nos limites", len(at.exception) == 0 and len(at.success) >= 1,
          f"success={[s.value for s in at.success]}, exceptions={exceptions(at)}")

    # 8. Modelo ausente
    try:
        if BACKUP.exists():
            BACKUP.unlink()
        MODEL.rename(BACKUP)
        at = new()
        warn = [w.value for w in at.warning]
        check("8. aviso de modelo ausente",
              len(at.exception) == 0 and any("modelo ainda não foi treinado" in w for w in warn),
              f"warnings={warn}, exceptions={exceptions(at)}")
    finally:
        if BACKUP.exists():
            BACKUP.rename(MODEL)
    check("8b. modelo restaurado", MODEL.exists(), str(MODEL))

    # 9. Time fora do treino não quebra o pipeline
    model = joblib.load(MODEL)
    sample = pd.DataFrame([{
        "time_mandante": "Time Inexistente",
        "time_visitante": "Outro Time",
        "odd_vitoria_mandante": 2.1,
        "odd_empate": 3.3,
        "odd_vitoria_visitante": 3.6,
    }])
    try:
        pred = model.predict(sample)[0]
        check("9. time fora do treino nao quebra", pred in ("H", "D", "A"), f"pred={pred}")
    except Exception as exc:  # noqa: BLE001
        check("9. time fora do treino nao quebra", False, repr(exc))

    # 10. Probabilidades somam 1
    try:
        proba = model.predict_proba(sample)[0]
        check("10. probabilidades somam 1", abs(float(proba.sum()) - 1.0) < 1e-6, f"soma={proba.sum():.6f}")
    except Exception as exc:  # noqa: BLE001
        check("10. probabilidades somam 1", False, repr(exc))

    failed = [name for name, ok in results if not ok]
    print(f"\nRESUMO: {len(results) - len(failed)}/{len(results)} passaram")
    if failed:
        print("FALHAS:", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    if not MODEL.exists():
        raise SystemExit("Modelo não encontrado. Execute train.py antes do teste.")
    run()
