from pathlib import Path
import json
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
for folder in (DATA, REPORTS, FIGURES, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

SEASONS = ["2021", "2122", "2223", "2324", "2425"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

def load_data():
    frames = []
    for season in SEASONS:
        path = DATA / f"premier_league_{season}.csv"
        if not path.exists():
            url = BASE_URL.format(season)
            print(f"Baixando temporada {season}...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                print(f"Não foi possível baixar {season}: {exc}")
                continue
        try:
            frame = pd.read_csv(path, encoding="latin1").copy()
            frame["Temporada"] = season
            frames.append(frame)
        except Exception as exc:
            print(f"Arquivo da temporada {season} não pôde ser lido: {exc}")
    if not frames:
        raise RuntimeError("Nenhuma temporada foi baixada. Verifique sua conexão com a internet.")
    return pd.concat(frames, ignore_index=True)

def main():
    df = load_data()
    required = ["HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colunas necessárias não encontradas: {missing}")

    df = df[required + (["Date"] if "Date" in df.columns else [])].copy()
    for col in ["B365H", "B365D", "B365A"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).drop_duplicates()
    df = df[(df[["B365H", "B365D", "B365A"]] > 1).all(axis=1)]
    df = df.rename(columns={
        "HomeTeam": "time_mandante",
        "AwayTeam": "time_visitante",
        "B365H": "odd_vitoria_mandante",
        "B365D": "odd_empate",
        "B365A": "odd_vitoria_visitante",
        "FTR": "resultado"
    })

    print("\n=== ANÁLISE EXPLORATÓRIA ===")
    print(f"Partidas analisadas: {len(df)}")
    print("\nDistribuição dos resultados (H=casa, D=empate, A=fora):")
    print(df["resultado"].value_counts().to_string())
    print("\nOdd média por resultado:")
    print(df.groupby("resultado")[["odd_vitoria_mandante", "odd_empate", "odd_vitoria_visitante"]].mean().round(2).to_string())
    print("\nTimes mandantes com mais partidas:")
    print(df["time_mandante"].value_counts().head(5).to_string())

    df["resultado"].value_counts().reindex(["H", "D", "A"]).plot(kind="bar", figsize=(8, 5))
    plt.title("Distribuição dos resultados das partidas")
    plt.xlabel("Resultado (H=casa, D=empate, A=fora)")
    plt.ylabel("Número de partidas")
    plt.tight_layout()
    plt.savefig(FIGURES / "distribuicao_resultados.png")
    plt.close()

    df.groupby("resultado")[["odd_vitoria_mandante", "odd_empate", "odd_vitoria_visitante"]].mean().plot(kind="bar", figsize=(9, 5))
    plt.title("Odds médias de acordo com o resultado")
    plt.xlabel("Resultado (H=casa, D=empate, A=fora)")
    plt.ylabel("Odd média")
    plt.tight_layout()
    plt.savefig(FIGURES / "odds_medias.png")
    plt.close()

    features = ["time_mandante", "time_visitante", "odd_vitoria_mandante", "odd_empate", "odd_vitoria_visitante"]
    X = df[features]
    y = df["resultado"]
    categorical = ["time_mandante", "time_visitante"]
    numeric = ["odd_vitoria_mandante", "odd_empate", "odd_vitoria_visitante"]
    preprocessor = ColumnTransformer([
        ("times", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("odds", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "Regressão Logística": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    }
    results, trained = {}, {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessamento", preprocessor), ("modelo", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "acuracia": round(float(accuracy_score(y_test, pred)), 4),
            "precisao_macro": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
            "revocacao_macro": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4)
        }
        trained[name] = pipe
        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, zero_division=0))

    best_name = max(results, key=lambda name: results[name]["f1_macro"])
    best_model = trained[best_name]
    joblib.dump(best_model, MODELS / "futebolai_model.joblib")
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"Matriz de confusão — {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURES / "matriz_confusao.png")
    plt.close()

    summary = {
        "melhor_modelo": best_name,
        "resultados": results,
        "partidas_analisadas": int(len(df)),
        "partidas_no_teste": int(len(y_test)),
        "classes": {"H": "Vitória do mandante", "D": "Empate", "A": "Vitória do visitante"},
        "nota": "A previsão usa odds pré-jogo e nomes dos times; não garante o resultado real."
    }
    with open(REPORTS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== RESULTADOS FINAIS ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nModelo salvo em models/futebolai_model.joblib")

if __name__ == "__main__":
    main()
