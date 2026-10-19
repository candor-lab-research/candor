import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 300)

KCOLS = ["attn.strength", "attn.peak_layer", "attn.edge_strength", "attn.reach",
         "mlp.strength", "mlp.peak_layer", "mlp.edge_strength", "mlp.reach"]


def candor():
    df = pd.read_csv(HERE / "candor_search_rude.csv")
    df["phase"] = df.trial.apply(lambda t: "random" if t <= 60 else "aimed")
    df["ref"] = df.refusals_judge
    return df


def heretic():
    df = pd.read_csv(HERE / "heretic_search_rude.csv")
    df["phase"] = df.trial.apply(lambda t: "random" if t <= 60 else "aimed")
    df["ref"] = df.refusals_keywords
    df["kl"] = df.kl_heretic_scale
    return df
