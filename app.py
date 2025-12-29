import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

# =============================
# APP CONFIG
# =============================
st.set_page_config(
    page_title="Project Goldilocks",
    page_icon="🪐",
    layout="wide"
)

# =============================
# LOAD MODEL & SCALER (CACHED)
# =============================
@st.cache_resource
def load_goldilocks_model():
    model = load_model("project_goldilocks_model.keras")
    scaler = joblib.load("project_goldilocks_scaler.pkl")
    return model, scaler

model, scaler = load_goldilocks_model()

# =============================
# HELPER FUNCTIONS
# =============================
def interpret_habitability(score):
    if score < 0.2:
        return "Very Low"
    elif score < 0.4:
        return "Low"
    elif score < 0.6:
        return "Moderate"
    elif score < 0.8:
        return "High"
    else:
        return "Very High"

atmosphere_labels = [
    "No Atmosphere",
    "Thin Atmosphere",
    "Thick Terrestrial Atmosphere",
    "Gas-Dominated Atmosphere"
]

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🪐 Project Goldilocks")
st.sidebar.caption("Exploring Exoplanet Habitability with AI")

page = st.sidebar.radio(
    "Navigate",
    ["📘 About Exoplanets", "🔮 Planet Analyzer"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built using NASA Exoplanet Archive data")

# =============================
# ABOUT PAGE
# =============================
if page == "📘 About Exoplanets":

    st.markdown("""
    # 🪐 Project Goldilocks  
    ### Exploring Exoplanet Habitability with AI  

    *An interactive deep learning system that studies which distant worlds
    might be capable of supporting life as we know it.*
    """)

    st.markdown("---")

    st.header("🌌 What Are Exoplanets?")
    st.markdown("""
    **Exoplanets** are planets that orbit stars outside our solar system.
    Thousands have been discovered, ranging from scorching gas giants
    to small rocky worlds unlike anything we see locally.
    """)

    st.header("🔭 How Do We Discover Exoplanets?")
    st.markdown("""
    Most exoplanets are detected indirectly:

    - **Transit method** – observing dips in a star’s brightness  
    - **Radial velocity** – detecting stellar wobble  

    These methods favor large, close-in planets, creating observational bias.
    """)

    st.header("🌍 How Common Are Potentially Habitable Planets?")
    st.markdown("""
    Earth-like planets are expected to be rare.
    Most known exoplanets are too hot, too cold, or gas-dominated.
    """)

    st.markdown("---")
    st.header("📊 Habitability Score Distribution")
    st.image("graph_1.png", caption="Distribution of Exoplanet Habitability Scores", width=680)

    st.markdown("---")
    st.header("🌡️ Habitability vs Planetary Temperature")
    st.image("graph_2.png", caption="Habitability Score vs Equilibrium Temperature", width=680)

    st.markdown("---")
    st.header("🌫️ Atmosphere Type Distribution")
    st.image("graph_3.png", caption="Distribution of Predicted Exoplanet Atmosphere Types", width=680)

    st.markdown("---")
    st.header("🧠 How Does Project Goldilocks Work?")
    st.markdown("""
    The system uses a **multi-output neural network**:

    - One output predicts **habitability likelihood**
    - The other predicts **atmospheric regime**

    These predictions are **likelihoods**, not confirmations.
    """)

# =============================
# PLANET ANALYZER PAGE
# =============================
elif page == "🔮 Planet Analyzer":

    st.markdown("""
    # 🔮 Planet Analyzer
    ### Explore how planetary conditions affect habitability and atmosphere
    """)

    st.markdown("---")

    # ---------- EARTH PRESET ----------
    if st.button("🌍 Use Earth-like Preset"):
        st.session_state.pl_rade = 1.0
        st.session_state.pl_masse = 1.0
        st.session_state.pl_orbsmax = 1.0
        st.session_state.pl_eqt = 288
        st.session_state.st_teff = 5778
        st.session_state.st_rad = 1.0

    # ---------- INPUTS ----------
    st.subheader("🧪 Planet & Star Inputs")

    pl_rade = st.slider(
        "Planet Radius (Earth radii)", 0.5, 20.0,
        st.session_state.get("pl_rade", 1.0), 0.1, key="pl_rade"
    )

    pl_masse = st.slider(
        "Planet Mass (Earth masses)", 0.1, 1000.0,
        st.session_state.get("pl_masse", 1.0), 0.1, key="pl_masse"
    )

    pl_orbsmax = st.slider(
        "Orbital Distance (AU)", 0.01, 10.0,
        st.session_state.get("pl_orbsmax", 1.0), 0.01, key="pl_orbsmax"
    )

    pl_eqt = st.slider(
        "Equilibrium Temperature (K)", 100, 2000,
        st.session_state.get("pl_eqt", 288), 10, key="pl_eqt"
    )

    st.markdown("**Host Star Properties**")

    st_teff = st.slider(
        "Star Temperature (K)", 3000, 8000,
        st.session_state.get("st_teff", 5778), 50, key="st_teff"
    )

    st_rad = st.slider(
        "Star Radius (Solar radii)", 0.1, 10.0,
        st.session_state.get("st_rad", 1.0), 0.1, key="st_rad"
    )

    st.markdown("---")

    # ---------- MODEL PREDICTION ----------
    X_input = np.array([[pl_rade, pl_masse, pl_orbsmax, pl_eqt, st_teff, st_rad]])
    X_scaled = scaler.transform(X_input)

    habitability_pred, atmosphere_pred = model.predict(X_scaled, verbose=0)

    habitability_score = float(habitability_pred[0][0])
    atmosphere_probs = atmosphere_pred[0]
    atmosphere_class = atmosphere_labels[np.argmax(atmosphere_probs)]

    # ---------- RESULTS ----------
    st.subheader("🧠 Prediction Results")

    st.markdown("**Habitability Likelihood**")
    st.progress(habitability_score)
    st.caption(
        f"Predicted habitability score: **{habitability_score:.2f} "
        f"({interpret_habitability(habitability_score)})**"
    )

    st.markdown("---")

    st.markdown("### 🌫️ Atmospheric Regime")
    st.markdown(f"**Predicted atmosphere:** 🟢 **{atmosphere_class}**")

    # ---------- ATMOSPHERE BAR CHART ----------
    st.markdown("**Atmosphere Probability Distribution**")

    prob_df = pd.DataFrame({
        "Atmosphere Type": atmosphere_labels,
        "Probability": atmosphere_probs
    }).set_index("Atmosphere Type")

    st.bar_chart(prob_df)

    for label, prob in zip(atmosphere_labels, atmosphere_probs):
        st.write(f"{label}: {prob*100:.1f}%")

    st.markdown("---")

    # ---------- HELPER INFO ----------
    st.subheader("ℹ️ Input Guide")

    with st.expander("🌍 Planet Radius"):
        st.write("Earth = 1.0 | Larger planets retain thicker atmospheres")

    with st.expander("⚖️ Planet Mass"):
        st.write("Higher mass = stronger gravity = better gas retention")

    with st.expander("🌡️ Equilibrium Temperature"):
        st.write("Earth ≈ 288 K | Extreme temperatures reduce habitability")

    with st.expander("☀️ Star Properties"):
        st.write("Sun ≈ 5778 K and 1.0 solar radius")

    st.caption("🔍 Predictions are probabilistic, not confirmations.")
