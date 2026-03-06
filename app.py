# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from pygam import LinearGAM, s, te, l

# =========================
# CONFIG
# =========================
DATA_FILE = "Historico25-26.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

GAM_PATH = os.path.join(MODEL_DIR, "gam_cfr.joblib")
CLS_PATH = os.path.join(MODEL_DIR, "cls_mpa.joblib")
PRE_PATH = os.path.join(MODEL_DIR, "preproc.json")

PRESSURE_GRID_FROM = 8000
PRESSURE_GRID_BY = 250
CAUDAL_POINTS = 25
CFR_TOL_REL = 0.02

# =========================
# DATA
# =========================
def read_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el CSV: {path}. Ponelo junto a app.py")

    df = pd.read_csv(path, decimal=",")
    df.columns = [c.strip() for c in df.columns]

    colmap = {
        "CAUDAL PROMEDIO(BPM)": "CaudalPromedio_BPM",
        "CAUDAL PROMEDIO (BPM)": "CaudalPromedio_BPM",
        "CAUDAL": "CaudalPromedio_BPM",
        "PRESION PROMEDIO (PSI)": "PresionPromedio_PSI",
        "PRESION PROMEDIO(PSI)": "PresionPromedio_PSI",
        "PRESION": "PresionPromedio_PSI",
        "CFR": "CFR",
        "CLUSTER": "Cluster",
        "CLUSTERS": "Cluster",
        "DISPAROS": "Disparos",
        "DISPARO": "Disparos",
        "TAPON (MTS)": "Tapon_m",
        "TAPON(MTS)": "Tapon_m",
        "TAPON": "Tapon_m",
        "TAPÓN (MTS)": "Tapon_m",
        "ISIP POST": "ISIP_post",
        "ISIP POST ": "ISIP_post",
        "ISIP_POST": "ISIP_post",
        "ISIP": "ISIP_post",
        "ISIP POSTERIOR": "ISIP_post",
        "BLOQUE": "Bloque",
        "CIA": "CIA",
        "COMPAÑIA": "CIA",
        "COMPANIA": "CIA",
        "PAD": "PAD",
        "MPA I": "MPA_raw",
        "MPA": "MPA_raw",
        "MPA_I": "MPA_raw",
    }

    for k, v in colmap.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    required = [
        "CaudalPromedio_BPM",
        "PresionPromedio_PSI",
        "CFR",
        "Cluster",
        "Disparos",
        "Tapon_m",
        "ISIP_post",
        "Bloque",
        "CIA",
        "PAD",
        "MPA_raw",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas: {missing}\nDetectadas: {list(df.columns)}"
        )

    for c in ["Bloque", "CIA", "PAD"]:
        df[c] = df[c].astype(str).str.replace("\u00A0", "", regex=False).str.strip()
        df.loc[df[c].eq("") | df[c].isna(), c] = "DESCONOCIDO"

    mpa0 = df["MPA_raw"].astype(str).str.replace("\u00A0", "", regex=False).str.strip()
    has_mpa = ~(mpa0.isna() | mpa0.eq("") | mpa0.eq("nan"))
    df["MPA_flag"] = np.where(has_mpa, "MPA", "NoMPA")
    df["MPA"] = np.where(has_mpa, mpa0, "SIN_MPA")

    num_cols = [
        "CaudalPromedio_BPM",
        "PresionPromedio_PSI",
        "CFR",
        "Cluster",
        "Disparos",
        "Tapon_m",
        "ISIP_post",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[
        np.isfinite(df["CFR"])
        & np.isfinite(df["CaudalPromedio_BPM"])
        & np.isfinite(df["PresionPromedio_PSI"])
    ].copy()

    if len(df) == 0:
        raise ValueError("Dataset vacío tras filtrar CFR/Caudal/Presión.")

    return df


def clean_mpa_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.upper().str.strip()
    x = x.replace({
        "": np.nan,
        "SIN_MPA": np.nan,
        "NONE": np.nan,
        "NA": np.nan,
        "NAN": np.nan,
    })
    x = x.str.replace(r"\s+", " ", regex=True)
    x = x.replace({
        "CLEAN  SWEEP": "CLEAN SWEEP",
        "DIRTY  SWEEP": "DIRTY SWEEP",
    })
    return x


def top_mpa_segment(df: pd.DataFrame, bloque: str, cia: str, top: int = 3) -> pd.DataFrame:
    sub = df[(df["Bloque"] == bloque) & (df["CIA"] == cia)]
    v = clean_mpa_series(sub["MPA"]).dropna()

    if v.empty:
        return pd.DataFrame(columns=["MPA", "n", "pct"])

    tab = v.value_counts().head(top)
    out = pd.DataFrame({"MPA": tab.index, "n": tab.values})
    out["pct"] = (100 * out["n"] / v.shape[0]).round(1)
    return out


# =========================
# MODELS
# =========================
def train_models(df: pd.DataFrame):
    num_features = [
        "CaudalPromedio_BPM",
        "PresionPromedio_PSI",
        "ISIP_post",
        "Tapon_m",
        "Cluster",
        "Disparos",
    ]
    cat_features = ["Bloque", "CIA"]

    # -------- GAM CFR --------
    df_gam = df.dropna(subset=num_features + cat_features + ["CFR"]).copy()
    if df_gam.empty:
        raise ValueError("No hay filas válidas para entrenar el GAM.")

    X_num = df_gam[num_features].copy()
    X_cat = pd.get_dummies(df_gam[cat_features], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if X.empty:
        raise ValueError("El dataset del GAM quedó vacío tras limpiar NaN/Inf.")

    y_cfr = df_gam.loc[X.index, "CFR"].values
    gam_cols = list(X.columns)

    # te(Q,P) + s(ISIP) + s(Tapon) + s(Cluster) + s(Disparos) + l(dummies Bloque/CIA)
    terms = te(0, 1) + s(2) + s(3) + s(4) + s(5)
    n_cat = X_cat.shape[1]
    for i in range(6, 6 + n_cat):
        terms = terms + l(i)

    gam = LinearGAM(terms).gridsearch(X.values, y_cfr)

    # -------- Clasificador MPA --------
    df_cls = df.dropna(subset=[
        "CaudalPromedio_BPM",
        "PresionPromedio_PSI",
        "Cluster",
        "Disparos",
        "Tapon_m",
        "ISIP_post",
        "Bloque",
        "CIA",
        "MPA_flag",
    ]).copy()

    if df_cls.empty:
        raise ValueError("No hay filas válidas para entrenar el clasificador MPA.")

    y_mpa = (df_cls["MPA_flag"] == "MPA").astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", [
                "CaudalPromedio_BPM",
                "PresionPromedio_PSI",
                "Cluster",
                "Disparos",
                "Tapon_m",
                "ISIP_post",
            ]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    cls = Pipeline(
        steps=[
            ("pre", pre),
            ("rf", RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )
    cls.fit(df_cls[
        [
            "CaudalPromedio_BPM",
            "PresionPromedio_PSI",
            "Cluster",
            "Disparos",
            "Tapon_m",
            "ISIP_post",
            "Bloque",
            "CIA",
        ]
    ], y_mpa)

    meta = {
        "data_file": DATA_FILE,
        "trained_at": pd.Timestamp.now().isoformat(),
        "levels": {
            "Bloque": sorted(df_gam["Bloque"].dropna().unique().tolist()),
            "CIA": sorted(df_gam["CIA"].dropna().unique().tolist()),
        },
        "gam_cols": gam_cols,
        "mpa_rate_global": float(np.round(y_mpa.mean() * 100, 1)),
    }

    return gam, cls, meta


def save_models(gam, cls, meta):
    joblib.dump(gam, GAM_PATH)
    joblib.dump(cls, CLS_PATH)
    with open(PRE_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_models():
    if not (os.path.exists(GAM_PATH) and os.path.exists(CLS_PATH) and os.path.exists(PRE_PATH)):
        return None

    gam = joblib.load(GAM_PATH)
    cls = joblib.load(CLS_PATH)
    with open(PRE_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return gam, cls, meta


# =========================
# OPTIMIZATION
# =========================
def segment_caudal(df, bloque, cia):
    s = df[(df["Bloque"] == bloque) & (df["CIA"] == cia)]["CaudalPromedio_BPM"].dropna().values
    s = s[np.isfinite(s)]
    if len(s) == 0:
        s = df[df["Bloque"] == bloque]["CaudalPromedio_BPM"].dropna().values
        s = s[np.isfinite(s)]
    return s


def caudal_limite(df, bloque, cia, method="p95"):
    s = segment_caudal(df, bloque, cia)
    if len(s) == 0:
        return np.nan
    return float(np.max(s) if method == "max" else np.quantile(s, 0.95))


def block_stats(df, bloque, cia):
    sub = df[(df["Bloque"] == bloque) & (df["CIA"] == cia)]
    disp_med = np.nanmedian(sub["Disparos"].values)
    isip_med = np.nanmedian(sub["ISIP_post"].values)

    if not np.isfinite(disp_med):
        disp_med = np.nanmedian(df[df["Bloque"] == bloque]["Disparos"].values)
    if not np.isfinite(isip_med):
        isip_med = np.nanmedian(df[df["Bloque"] == bloque]["ISIP_post"].values)

    if not np.isfinite(disp_med):
        disp_med = np.nanmedian(df["Disparos"].values)
    if not np.isfinite(isip_med):
        isip_med = np.nanmedian(df["ISIP_post"].values)

    return {
        "Disparos_med": float(disp_med),
        "ISIP_med": float(isip_med),
    }


def gam_predict_cfr(gam, meta, newdf: pd.DataFrame) -> np.ndarray:
    X_num = newdf[[
        "CaudalPromedio_BPM",
        "PresionPromedio_PSI",
        "ISIP_post",
        "Tapon_m",
        "Cluster",
        "Disparos",
    ]].copy()

    X_cat = pd.get_dummies(newdf[["Bloque", "CIA"]], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)

    for c in meta["gam_cols"]:
        if c not in X.columns:
            X[c] = 0.0

    X = X[meta["gam_cols"]]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return gam.predict(X.values)


def consulta_optima(
    gam, cls, meta, df, bloque, cia, tapon_m, cluster,
    psi_max=11000, isip_opt=np.nan, caudal_method="p95"
):
    s_seg = segment_caudal(df, bloque, cia)
    if len(s_seg) == 0:
        raise ValueError("No hay caudales para ese Bloque/CIA (ni para el bloque).")

    if not np.isfinite(tapon_m):
        t_seg = df[(df["Bloque"] == bloque) & (df["CIA"] == cia)]["Tapon_m"].values
        t_seg = t_seg[np.isfinite(t_seg)]
        tapon_m = float(np.median(t_seg) if len(t_seg) else np.nanmedian(df["Tapon_m"].values))

    if not np.isfinite(tapon_m):
        raise ValueError("Tapón inválido (NA).")
    if not np.isfinite(cluster):
        raise ValueError("Cluster inválido.")

    stats = block_stats(df, bloque, cia)

    caudal_hi = caudal_limite(df, bloque, cia, caudal_method)
    caudal_lo = float(np.quantile(s_seg, 0.05))
    if not np.isfinite(caudal_lo):
        caudal_lo = float(np.min(s_seg))
    if not np.isfinite(caudal_hi):
        raise ValueError("No se pudo calcular el caudal máximo del segmento.")
    if caudal_lo > caudal_hi:
        caudal_lo, caudal_hi = caudal_hi, caudal_lo

    presiones = np.arange(PRESSURE_GRID_FROM, min(psi_max, 11000) + 1, PRESSURE_GRID_BY)
    if len(presiones) == 0:
        raise ValueError("Grilla de presión vacía.")

    caudales = np.linspace(caudal_lo, caudal_hi, CAUDAL_POINTS)

    isip_val = float(isip_opt if np.isfinite(isip_opt) else stats["ISIP_med"])
    if not np.isfinite(isip_val):
        isip_val = float(np.nanmedian(df["ISIP_post"].values))

    gridP, gridQ = np.meshgrid(presiones, caudales, indexing="xy")

    newdata = pd.DataFrame({
        "CaudalPromedio_BPM": gridQ.ravel(),
        "PresionPromedio_PSI": gridP.ravel(),
        "Cluster": float(cluster),
        "Disparos": stats["Disparos_med"],
        "Tapon_m": float(tapon_m),
        "ISIP_post": isip_val,
        "Bloque": bloque,
        "CIA": cia,
    })

    cfr_pred = gam_predict_cfr(gam, meta, newdata)

    prob_mpa = cls.predict_proba(newdata[[
        "CaudalPromedio_BPM",
        "PresionPromedio_PSI",
        "Cluster",
        "Disparos",
        "Tapon_m",
        "ISIP_post",
        "Bloque",
        "CIA",
    ]])[:, 1]

    res = pd.DataFrame({
        "Presion_sugerida_PSI": newdata["PresionPromedio_PSI"].values,
        "Caudal_BPM": newdata["CaudalPromedio_BPM"].values,
        "CFR_pred": cfr_pred,
        "Prob_MPA": prob_mpa,
    })

    cfr_min = np.nanmin(res["CFR_pred"].values)
    cand = res[res["CFR_pred"] <= cfr_min * (1 + CFR_TOL_REL)].copy()

    if cand.empty:
        raise ValueError("No se encontraron candidatos válidos dentro del rango de CFR.")

    q_obj = float(np.nanquantile(cand["Caudal_BPM"].values, 0.75))
    cand["dist_q"] = np.abs(cand["Caudal_BPM"].values - q_obj)
    best = cand.sort_values(["dist_q", "Presion_sugerida_PSI"]).iloc[0]

    return {
        "bloque": bloque,
        "cia": cia,
        "cfr_min": float(best["CFR_pred"]),
        "presion": float(best["Presion_sugerida_PSI"]),
        "caudal": float(best["Caudal_BPM"]),
        "prob_mpa": float(best["Prob_MPA"]),
        "isip_usado": isip_val,
        "disparos_usados": stats["Disparos_med"],
        "metodo_caudal": caudal_method,
        "caudal_max": float(caudal_hi),
        "cfr_min_global": float(cfr_min),
    }


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Optimización CFR + MPA", layout="wide")
st.title("Optimización CFR (GAM) + Prob. MPA")

@st.cache_data
def get_df():
    return read_data(DATA_FILE)

df = get_df()

loaded = load_models()
if loaded is None:
    st.info("No hay modelos guardados. Entrenando por primera vez…")
    gam, cls, meta = train_models(df)
    save_models(gam, cls, meta)
else:
    gam, cls, meta = loaded

with st.sidebar:
    st.header("Parámetros")

    bloque = st.selectbox("Bloque", sorted(df["Bloque"].unique()))
    cia = st.selectbox("CIA", sorted(df["CIA"].unique()))

    sub = df[(df["Bloque"] == bloque) & (df["CIA"] == cia)]
    t_med = float(np.nanmedian(sub["Tapon_m"].values)) if len(sub) else float(np.nanmedian(df["Tapon_m"].values))

    tapon = st.number_input("Tapón (m)", value=float(t_med) if np.isfinite(t_med) else 0.0, step=1.0)
    cluster = st.number_input("Cluster (#)", value=10, min_value=1, step=1)
    isip = st.number_input("ISIP (PSI opcional)", value=float("nan"), step=10.0)
    caudal_method = st.selectbox("Caudal Máximo", ["p95", "max"], index=0)
    psi_max = st.number_input("Límite de presión (PSI)", value=11000, min_value=0, step=100)

    calc = st.button("Calcular", type="primary")
    retrain = st.button("Reentrenar y guardar modelos")

if retrain:
    st.warning("Reentrenando…")
    gam, cls, meta = train_models(df)
    save_models(gam, cls, meta)
    st.success("Listo. Modelos actualizados.")

if calc:
    try:
        out = consulta_optima(
            gam, cls, meta, df,
            bloque, cia, tapon, cluster,
            psi_max=psi_max,
            isip_opt=isip,
            caudal_method=caudal_method,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CFR mínima predicha", f"{out['cfr_min']:.3f}")
        c2.metric("Presión sugerida (PSI)", f"{out['presion']:.0f}")
        c3.metric("Caudal recomendado (BPM)", f"{out['caudal']:.1f}")
        c4.metric("Prob. MPA", f"{out['prob_mpa'] * 100:.0f}%")

        st.subheader("Top MPA histórico (Bloque + CIA)")
        st.dataframe(top_mpa_segment(df, bloque, cia, top=3), use_container_width=True)

        with st.expander("Detalle del resultado"):
            st.json(out)

    except Exception as e:
        st.error(f"Error en cálculo: {e}")
else:
    st.info("Completá los parámetros y presioná Calcular.")
