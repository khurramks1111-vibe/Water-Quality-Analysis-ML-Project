import streamlit as st
import joblib
import numpy as np

# 1. Page Configuration (Title aur Layout)
st.set_page_config(
    page_title="Water Quality AI Predictor",
    page_icon="💧",
    layout="centered"
)

# --- 2. Model aur Scaler Load Karein ---
@st.cache_resource  # Isse model bar-bar load nahi hoga, app super fast chalegi
def load_ml_assets():
    try:
        model = joblib.load('water_potability_rf_model.pkl')
        scaler = joblib.load('water_potability_scaler.pkl')
        return model, scaler, "🟢 AI Model Connected Successfully"
    except Exception as e:
        return None, None, f"🔴 Error loading model: {str(e)}"

model, scaler, status_message = load_ml_assets()

# --- 3. UI Header ---
st.title("💧 Water Quality Analysis AI")
st.write("Enter the chemical and biological concentrations below to predict water safety.")
st.caption(status_message)

st.markdown("---")

# --- 4. Inputs Layout (2 Columns Grid) ---
# Hum page ko 2 columns mein divide kar rahe hain taaki ui khoobsurat lage
col1, col2 = st.columns(2)

# Features list default values ke sath (mg/L)
features_info = [
    ("Aluminium", "aluminium", 1.65), ("Ammonia", "ammonia", 9.08),
    ("Arsenic", "arsenic", 0.04), ("Barium", "barium", 2.85),
    ("Cadmium", "cadmium", 0.007), ("Chloramine", "chloramine", 0.35),
    ("Chromium", "chromium", 0.83), ("Copper", "copper", 0.17),
    ("Flouride", "flouride", 0.05), ("Bacteria", "bacteria", 0.20),
    ("Viruses", "viruses", 0.00), ("Lead", "lead", 0.054),
    ("Nitrates", "nitrates", 16.08), ("Nitrites", "nitrites", 1.13),
    ("Mercury", "mercury", 0.007), ("Perchlorate", "perchlorate", 37.75),
    ("Radium", "radium", 6.78), ("Selenium", "selenium", 0.08),
    ("Silver", "silver", 0.34), ("Uranium", "uranium", 0.02)
]

input_values = []

# Inputs ko dynamically donon columns mein distribute karna
for index, (display_name, key_name, default_val) in enumerate(features_info):
    # Float step adjustment automatic decimal settings ke liye
    step_val = 0.001 if default_val < 0.1 else 0.01
    
    if index % 2 == 0:
        with col1:
            val = st.number_input(f"{display_name} (mg/L)", value=float(default_val), step=step_val, key=key_name)
            input_values.append(val)
    else:
        with col2:
            val = st.number_input(f"{display_name} (mg/L)", value=float(default_val), step=step_val, key=key_name)
            input_values.append(val)

st.markdown("---")

# --- 5. Prediction Logic & Button ---
if st.button("🚀 ANALYZE WATER SAMPLE", use_container_width=True):
    if model is None or scaler is None:
        st.error("Model files not loaded properly. Check terminal errors.")
    else:
        with st.spinner("AI is analyzing chemical compounds..."):
            # Reshape aur Scale
            data_array = np.array(input_values).reshape(1, -1)
            data_scaled = scaler.transform(data_array)
            
            # Probability aur 0.58 Custom Strict Threshold Rule
            safe_probability = float(model.predict_proba(data_scaled)[:, 1][0])
            is_safe = 1 if safe_probability >= 0.58 else 0
            
            confidence_percentage = safe_probability * 100
            
            # --- 6. Results Display ---
            st.subheader("Analysis Result:")
            if is_safe == 1:
                st.success(f"🟢 **WATER IS SAFE TO DRINK!**")
                st.info(f"**Confidence Score:** {confidence_percentage:.2f}% (Matches Clean Water Patterns)")
            else:
                unsafe_percentage = 100 - confidence_percentage
                st.error(f"🔴 **WATER IS CONTAMINATED / UNSAFE!**")
                st.warning(f"**Risk Factor:** {unsafe_percentage:.2f}% (High levels of restricted/toxic elements detected)")