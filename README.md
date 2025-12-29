# 🪐 Project Goldilocks  
### Exploring Exoplanet Habitability with AI

***

## 🌌 Overview

**Project Goldilocks** is an interactive deep learning application that analyzes the physical properties of exoplanets and their host stars to estimate:

- 🌍 **Habitability likelihood**
- 🌫️ **Atmospheric regime**

The project combines astrophysical intuition with a multi-output neural network and is deployed as a user-friendly **Streamlit web application**.

***

## 🔭 Motivation

Thousands of exoplanets have been discovered, yet only a small fraction may be capable of supporting life as we know it.  
Direct measurements of habitability and atmospheres are rare, so scientists rely on **inference from observable properties**.

This project mirrors that reasoning process using AI.

***

## 🧠 How It Works

The system uses a **multi-output deep learning model** trained on data from the **NASA Exoplanet Archive**.

### Inputs
- Planet radius (Earth radii)
- Planet mass (Earth masses)
- Orbital distance (AU)
- Equilibrium temperature (K)
- Host star temperature (K)
- Host star radius (Solar radii)

### Outputs
- **Habitability score** (0–1 likelihood)
- **Atmospheric regime classification**:
  - No atmosphere
  - Thin atmosphere
  - Thick terrestrial atmosphere
  - Gas-dominated atmosphere

Predictions represent **likelihoods**, not confirmations.

***

## 📊 Visual Exploration

The app includes educational visualizations that explain the data and model behavior:

- Distribution of habitability scores
- Habitability vs planetary temperature
- Atmosphere type distribution across known exoplanets

These help users understand *why* certain predictions are made.

***

## 🔮 Planet Analyzer

Users can interactively:
- Adjust planetary and stellar parameters
- Apply an **Earth-like preset**
- View real-time predictions
- Explore atmosphere probabilities via bar charts
- Read human-friendly explanations of results

***

## ⚙️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **scikit-learn**
- **Streamlit**
- **NumPy / Pandas**

***

## 🚀 Deployment

The application is designed for deployment on **Streamlit Cloud** and runs entirely on a pre-trained model without requiring raw training data.

***

## ⚠️ Disclaimer

This project is intended for **educational and exploratory purposes**.  
It does not claim to identify truly habitable planets or confirmed atmospheres.

***

## ✨ Author

Built with curiosity, care, and a love for space 🌌

