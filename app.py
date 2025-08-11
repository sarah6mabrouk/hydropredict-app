import streamlit as st
import base64
import os
import joblib as jbl
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from streamlit.components.v1 import html

# Google verification file
if st.query_params.get("verify") == "google6b04cdb89a6ecbd5.html":
    st.write("google-site-verification: google6b04cdb89a6ecbd5.html")
    st.stop()

# == Set Streamlit page config ==
st.set_page_config(page_title="HydroPredict", layout="wide")

# ----------------------------
# Theme Setup via CSS
# ----------------------------
st.markdown("""
    <style>
        :root {
            --primary-color: #7678ff;
            --secondary-color: #8f91ff;
            --text-color: #222222;
            --background-color: #ffffff;
            --card-background: #ffffff;
            --font-family: 'Segoe UI', 'Roboto', sans-serif;
        }

        body {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: var(--font-family);
        }

        .main .block-container {
            padding-top: 2rem;
        }

        .title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }

        .subtitle {
            font-size: 1.25rem;
            margin-bottom: 2rem;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 3rem;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }

        .card {
            background-color: var(--card-background);
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 7px 12px rgba(0, 0, 0, 0.3);
            margin-bottom: 1.5rem;
        }

        .center {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .hydro-button button {
            background-color: var(--primary-color) !important;
            color: white !important;
            border: none;
            padding: 0.5rem 1.5rem;
            font-size: 1rem;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
        }

        .hydro-button button:hover {
            background-color: #0056b3 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.button-link {
    display: inline-block;
    margin-top: 1.5rem;
    margin-right: 1rem;
    padding: 0.75rem 1.5rem;
    background-color: #0068c9;
    
    color: white !important;
    text-decoration: none !important;
    border-radius: 5px;
    font-size: 16px;
    position: relative;
    z-index: 1;
    transition: background-color 0.3s ease;
}

.button-link:hover {
    background-color: #acd7ff;
    color: black !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load image and encode to base64 ===
@st.cache_data
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        return encoded


# Get base64 image (cached across reruns)
def load_base64_image(image_path, label="Image"):
    if os.path.exists(image_path):
        return get_base64_image(image_path)
    else:
        st.warning(f"{label} not found at: {image_path}")
        return ""



# ---------------------------------
# sidebar
#----------------------------------
# Initialize toggle state
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = True

# Inject fixed toggle button with top margin
st.markdown(f"""
<style>
.toggle-button {{
    position: fixed;
    top: 5rem;
    left: 20px;
    background-color: #7678ff;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    z-index: 10000;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: background-color 0.3s ease;
}}
.toggle-button:hover {{
    background-color: #5a5edc;
}}
</style>

<div class="toggle-button" onclick="document.dispatchEvent(new CustomEvent('toggleSidebar'))">
    ☰ {"Hide Menu" if st.session_state.show_sidebar else "Show Menu"}
</div>

<script>
document.addEventListener('toggleSidebar', function() {{
    fetch('/_stcore/toggle_sidebar', {{method: 'POST'}})
}});
</script>
""", unsafe_allow_html=True)

# Fallback Streamlit button (optional for compatibility)
if st.button("☰ Hide Menu" if st.session_state.show_sidebar else "☰ Show Menu"):
    st.session_state.show_sidebar = not st.session_state.show_sidebar

# Fixed sidebar menu
if st.session_state.show_sidebar:
    st.markdown("""
    <style>
    .sidebar-menu {
        position: fixed;
        top: 80px;
        left: 20px;
        width: 220px;
        background-color: #f5f7ff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', sans-serif;
        z-index: 9999;
    }

    .sidebar-menu a {
        display: block;
        margin: 0.5rem 0;
        color: #333;
        text-decoration: none;
        font-size: 16px;
        font-weight: 500;
        transition: color 0.3s ease;
    }

    .sidebar-menu a:hover {
        color: #7678ff;
    }

    .main {
        margin-left: 260px;
    }
    </style>

    <div class="sidebar-menu">
        <a href="#home">🏠 Home</a>
        <a href="#what">📘 What is Predictive Maintenance</a>
        <a href="#why">🔍 Why Predictive Maintenance</a>
        <a href="#process">⚙️ HydroPredict App – Process</a>
        <a href="#model">🤖 ML Model</a>
        <a href="#dashboard">📊 Dashboard</a>
        <a href="#contact">📬 Contact Me</a>
    </div>
    """, unsafe_allow_html=True)

# Content container to avoid overlap
st.markdown("<div style='margin-left:260px'>", unsafe_allow_html=True)


# ---------------------------------
# Header - Home
#----------------------------------

# Path to the header image
image_header = "assets/header_pic.png" 
image_section_1 = "assets/predictive maintenance.png"

base64_image_header = load_base64_image(image_header, label = "Header Image")
base64_image_section1 = load_base64_image(image_section_1, label="Section 1 Image")


# === Hero Section with Background Image ===
st.markdown('<div id="home"></div>', unsafe_allow_html=True)
if base64_image_header:
    st.markdown(
        f"""
        <style>
        .hero-wrapper {{
            position: relative;
            min-height: 100vh;
            width: 100vw;
            padding: 3rem;
            margin: 0;
            text-align: center;
            color: #1e1e1e;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        .hero-background {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('data:image/jpg;base64,{base64_image_header}');
            background-size: cover;
            background-position: center;
            opacity: 0.3;
            z-index: 0;
        }}

        .hero-content {{
            position: relative;
            z-index: 1;
            max-width: 900px;
        }}

        .button-link {{
            display: inline-block;
            margin: 1rem 0.5rem;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #ACD7FF, #ACAEFF);
            color: #1e1e1e;
            text-decoration: none;
            border-radius: 15px;
            
        }}

        .button-link:hover {{
            background-color: #ACAEFF;
        }}
        </style>

        <div class="hero-wrapper">
            <div class="hero-background"></div>
            <div class="hero-content">
                <h1>HydroPredict - Predict Before It Fails</h1>
                <p>Built for industrial engineers, plant operators & maintenance teams</p>
                <p style="font-size: 18px;">
                    HydroPredict is your intelligent solution for predictive maintenance in hydraulic systems.
                    By leveraging advanced analytics, it helps you anticipate issues before they disrupt your operations.
                </p>
                <a href="#what" class="button-link">Learn more</a>
                <a href="https://github.com/sarah6mabrouk/Hydraulic-System-Failure-Prediction-Using-XGBoost-andPCA" target="_blank" class="button-link">GitHub</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.write("Could not load image background. Check image path and file format.")

# -----------------------------
#  Page content
# ------------------------------

# == 1. What is Predictive maintenance

# Anchor for scrolling
st.markdown('<div id="what"></div>', unsafe_allow_html=True)

# Definition of PdM
st.markdown("""
<div style="margin-top: 5rem;">
""", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## <span style='color:#7678ff'>What is Predictive Maintenance?</span>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 18px;">
        Predictive maintenance (PdM) is a proactive strategy that uses historical and real-time data—such as vibration, temperature, and acoustic signals—to monitor the health of equipment and anticipate failures before they happen.<br>
        By identifying anomalies early, it helps reduce unplanned downtime, optimize maintenance schedules, and extend asset lifespan.<br>
        Unlike preventive maintenance, which relies on fixed schedules, PdM responds to actual equipment conditions, making it smarter and more cost-effective.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### <span style='color:blue'>This 2-minute video explains it simply --> </span>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #acd7ff, #ACAEFF);
        border-radius: 10px;
        padding: 10px;
        margin-top: 20px;
    ">
        <div style="
            background: white;
            border-radius: 8px;
            overflow: hidden;
        ">
            <iframe width="100%" height="400"
                src="https://www.youtube.com/embed/f8SisiVFFx4"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
            </iframe>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Close the outer container
st.markdown("</div>", unsafe_allow_html=True)

# == 2. Why predictive maintenance?==
st.markdown('<div id="why"></div>', unsafe_allow_html=True)
st.markdown(f"""
    <div id="predictive-maintenance" style=
        "display: flex; 
        min-height: 400px;
        align-items: center;
        justify-content: space-between;
        margin-top: 5rem;
        gap: 1rem;
        flex-wrap: wrap;">
        <!-- Left Column: image -->
        <div style="flex: 1; min-width: 300px; text-align: left;">
            <img src="data:image/png;base64,{base64_image_section1}" 
            alt="Predictive Maintenance" 
            style=
            "width: 100%;
            max-width: 30rem; 
            border-radius: 15px; 
            ">
        </div>
        <!-- Right Column: text -->
        <div style="flex: 1; min-width: 300px;">
            <h2 style="margin-bottom: 1rem; color:#7678ff;"> Why Predictive Maintenance Matters</h2>
            <p style="font-size: 20px; line-height: 1.6;">
                Predictive Maintenance helps you stay ahead of costly breakdowns by identifying issues before they become failures.
                It reduces downtime, improves safety, and extends the life of the hydraulic systems.
            </p>
            <ul style="font-size: 18px; line-height: 1.6;">
                <li><strong>Cost Efficiency:</strong> Lower maintenance costs by fixing problems early.</li>
                <li><strong>Operational Safety:</strong> Prevent unexpected failures and ensure a safer environment.</li>
            </ul>
        </div>    

    </div>
""", unsafe_allow_html= True)



# == Process behind HydroPredict Predictive Maintenance ==
## Encoding the pictures to base64 (because we're using html inline in streamlit)
base64_HydEn = load_base64_image("assets/hydraulic-energy.png", label="HydEn Image")
base64_step = load_base64_image("assets/step.png", label="Step Image")
base64_insights = load_base64_image("assets/insight.png", label = "Inight Image")

st.markdown('<div id="process"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="margin-top: 5rem; margin-bottom: 5rem;">
    <h2 style="text-align: left; margin-bottom: 2rem; color:#7678ff;"> ⚙️ Discover the streamlined process behind HydroPredict's predictive maintenance technology.
    </h2>
    <div style="
            display:flex;
            flex-wrap:wrap;
            gap: 2rem;
            justify-content: space-between;">
        <!-- First div: hydraulic sys° insights -->
        <div class= "card"; style="flex: 1; min-width: 280px; padding: 1rem; background: linear-gradient(135deg, #ACD7FF, #ACAEFF); border-radius: 10px;"> 
            <img src="data:image/png;base64,{base64_HydEn}"
                alt="Hydraulic Energy Visualization"
                style="width:100%;
                max-width:4rem;
                border-radius:10px;">
            <h5 style="color: #000000; margin-top:2rem;">Smart Sensors.</h5>
            <p style="color: #1e1e1e;"> HydroPredict employs a systematic approach to ensure optimal performance. It uses data from sensors in the industry to detect anomalies</p>
        </div>  
        <!-- Second div: HydroPredict process -->
        <div class= "card"; style="flex: 1; min-width: 280px; padding: 1rem; background: linear-gradient(135deg, #ACD7FF, #ACAEFF); border-radius: 10px;">
            <img src ="data:image/png;base64,{base64_step}"
                alt="Step-by-step icon"
                style="width:100%;
                max-width:4rem;
                border-radius:10px;">
            <h5 style="color: #000000;  margin-top:2rem;">AI Predictions.</h5>
            <p style="color: #1e1e1e;">HydroPredict monitors key metrics and alerts before costly failures occur.</p>
        </div>
        <!-- Third div: Insights -->
        <div class= "card"; style="flex: 1; min-width: 280px; padding: 1rem; background: linear-gradient(135deg, #ACD7FF, #ACAEFF); border-radius: 10px;">
            <img src ="data:image/png;base64, {base64_insights}"
                alt="insights data"
                style="width:100%;
                max-width:4rem;
                border-radius:10px;">
            <h5 style="color: #000000; margin-top:2rem;">Dashboard Integration.</h5>
            <p style="color: #1e1e1e;">HydroPredict utilizes advanced machine learning techniques to forecast potential failures, and visualizing them using PowerBI or Tableau</p>
        </div>          
    </div>
</div>
""", unsafe_allow_html= True)



# == 3. ML Model

st.markdown('<div id="model"></div>', unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    base_path = os.path.join(os.getcwd(), "models")
    try:
        model = jbl.load(os.path.join(base_path, "xgb_means_model.pkl"))
        scaler = jbl.load(os.path.join(base_path, "means_scaler.pkl"))
        mean_columns = jbl.load(os.path.join(base_path, "mean_columns.pkl"))
        default_input = jbl.load(os.path.join(base_path, "default_input_mean.pkl"))
        return model, scaler, mean_columns, default_input
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None, None

# Load the saved model, scaler, and feature names
model, scaler, mean_columns, default_input = load_artifacts()

#define columns
col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("## :blue[🤖 Test Our Model: Predict Leakage with Your Data]")

    st.markdown("""
        <h3 style='color:#7678ff;'>🧠 What Does the Model Do?</h3>
        HydroPredict is built to <strong>predict internal pump leakage</strong> in hydraulic systems using machine learning.  
        The model was trained on statistical summaries of sensor data—specifically the <strong>mean values</strong> over time.

        <br>

        <h3 style='color:#7678ff;'>📊 Why Use Mean Values?</h3>
        Instead of feeding the model full time-series data, we use the <strong>mean</strong> of each sensor to:

        - 🧩 <strong>Reduce complexity</strong> from hundreds of raw sensor columns  
        - ⚡ <strong>Speed up predictions</strong> for real-time decision-making  
        - 🛡️ <strong>Improve generalization</strong> by smoothing noise and outliers

        These features (e.g., <code>ps1_mean</code>, <code>vs2_mean</code>, etc.) represent the <strong>average sensor readings</strong> collected during equipment operation.

        <br>

        <h3 style='color:#7678ff;'>🏭 How This Works in Industry</h3>
        In a real-world factory or plant, sensors like pressure and vibration detectors would:

        - Continuously measure values during machine operation  
        - Automatically calculate and stream mean values over fixed intervals  
        - Feed these values into HydroPredict for <strong>instant leakage detection</strong>

        But here, you can <strong>manually input the same values</strong> to test how the model works. Just enter the mean values below and get a prediction in real-time.

        <br><br>
    """, unsafe_allow_html=True)

    highlights = [
        "🔍 Input your own data to test leakage prediction",
        "🚨 Instantly know if the pump is healthy or leaking",
        "🛠️ Explore the power of predictive maintenance"
    ]

    st.write("♣ Test our Machine Learning Model:")
    for item in highlights:
        st.markdown(f"- {item}")




with col2:
    st.markdown("""
    <h3 style='color:#7678ff;'>Hydraulic System Failure Prediction</h3>
""", unsafe_allow_html=True)
    # App title
    st.write("The model uses **mean sensor values** to predict the internal condition of the pump.")
    # Input form
    with st.form("input_form"):
        inputs = []
        st.markdown("### Input Mean Sensor Values")

        for i, col_name in enumerate(mean_columns):
            default_val = default_input[i] if default_input else 0.0
            val = st.number_input(f"{col_name}", value=float(default_val))
            inputs.append(val)
        submitted = st.form_submit_button("Predict")

    # Prediction logic
    if submitted:
        X_input = pd.DataFrame([inputs], columns=mean_columns)
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]

        if prediction == 0:
            message = "No Leakage"
        elif prediction == 1:
            message = "Weak Leakage"
        else:
            message = "Severe Leakage"

        st.success(f"Predicted internal pump leakage class: **{prediction}** : **{message}**")



# == 4.tableau dashboard
st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)

st.markdown("## :blue[📊 Visual Insights: Dashboard]")

tableau_url = "https://public.tableau.com/views/InternalPumpLeakage/Dashboard1?:showVizHome=no&:embed=true"

st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <iframe src="{tableau_url}" width="100%" height="900" style="border:none;"></iframe>
    </div>
    """,
    unsafe_allow_html=True
)



# == Section 7: Quote from an important person
base64_makini = load_base64_image("assets/makini_logo.png", label="Makini logo")
st.markdown(f"""
<!-- Quote -->
<div class= "card"; style="
    display: flex;
    flex-direction: column;
    background: linear-gradient(135deg, #ACD7FF, #ACAEFF);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    min-height: 40vh;
    margin-top: 5rem;
    margin-bottom: 5rem;
    justify-content: center;
    ">
    <div style="text-align: center; padding-bottom: 3rem;">
    <a href="https://www.makini.io/" target="_blank">
        <img src="data:image/png;base64,{base64_makini}" 
            alt="Makini logo" 
            style="width:100%; max-width:150px;">
    </a>
    </div>
    <div style="padding: 1rem;">
    <p style="
        color: #DDDDDD;
        line-height: 1.5;
        font-style: italic;
        margin:0;
        font-size: 2rem;
        text-align: center;">
        <a href="https://www.makini.io/educational-resources/implementing-effective-preventive-maintenance-strategies-a-comprehensive-guide" 
            target="_blank"
            style="color:#000000; text-decoration: none;">
            " Preventive maintenance is no longer optional—it’s a strategic necessity for staying competitive. "
        </a>
    </p>
    </div>               
</div>         
""", unsafe_allow_html= True)

# ----------------------------
# Call to Action
# ----------------------------
st.markdown('<div id="contact"></div>', unsafe_allow_html=True)

st.markdown("""
    <div style='color:#7678ff; font-size:22px;'>
        ♦ Curious how predictive maintenance could transform your workflow?
    </div>
""", unsafe_allow_html=True)

# Option 2 – More conversational
st.markdown("""
    <div style='color:#7678ff; font-size:22px;'>
        ♣ Ready to explore predictive maintenance in your own operations?
    </div>
""", unsafe_allow_html=True)

# Option 3 – Clear and professional
st.markdown("""
    <div style='color:#7678ff; font-size:22px;'>
        ♠ Discover how predictive maintenance can improve efficiency and reliability.
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 50px;">
    <a href="https://www.linkedin.com/in/sarahmab" target="_blank" class="button-link">
        Contact Me
    </a>
</div>
""", unsafe_allow_html=True)

# == Footer
## Encoding the logos
base64_linkedin = load_base64_image("assets/linkedin.png", label="linkedin logo")
base64_linktree = load_base64_image("assets/linktree.png", label="linktree logo")
base64_tableau = load_base64_image("assets/tableau.png", label="tableau logo")
base64_github = load_base64_image("assets/github.png", label="github logo")

## Footer itself
st.markdown(f"""
<hr style="margin-top: 5rem; margin-bottom: 2rem; border: none; border-top: 1px solid #555555;">

<div style="
    text-align: center;
    font-size: 0.95rem;
    color: #888888;
    padding: 2rem;
    background-color: #ffffff;
    border-radius: 8px;
">

  <p>Connect with me:</p>

  <a href="https://github.com/sarah6mabrouk/Hydraulic-System-Failure-Prediction-Using-XGBoost-andPCA" target="_blank" style="color: #7678ff; text-decoration: none; margin: 0 1rem;">
    <img src="data:image/png;base64,{base64_github}" 
         alt="GitHub logo" 
         style="max-width: 18px; vertical-align: middle; margin-right: 0.5rem;">
    GitHub Project
  </a> |

  <a href="https://www.linkedin.com/in/sarahmab" target="_blank" style="color: #7678ff; text-decoration: none; margin: 0 1rem;">
    <img src="data:image/png;base64,{base64_linkedin}" 
         alt="LinkedIn logo" 
         style="max-width: 18px; vertical-align: middle; margin-right: 0.5rem;">
    LinkedIn
  </a> |

  <a href="https://linktr.ee/sarahmabrouk" target="_blank" style="color: #7678ff; text-decoration: none; margin: 0 1rem;">
    <img src="data:image/png;base64,{base64_linktree}" 
         alt="Linktree logo" 
         style="max-width: 18px; vertical-align: middle; margin-right: 0.5rem;">
    Linktree
  </a> |

  <a href="https://public.tableau.com/app/profile/sarah.mabrouk7137/vizzes" target="_blank" style="color: #7678ff; text-decoration: none; margin: 0 1rem;">
    <img src="data:image/png;base64,{base64_tableau}" 
         alt="Tableau logo" 
         style="max-width: 18px; vertical-align: middle; margin-right: 0.5rem;">
    Tableau Profile
  </a>

  <p style="margin-top: 1rem; color: #888888;">© 2025 Built by Sarah Mabrouk, Chemical Engineer & Data Scientist.</p>

</div>
""", unsafe_allow_html=True)
