import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Multiverse Predictor", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🔮"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Ubuntu:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        background-color: #0f0c29;
        color: #f0f0f0;
        font-family: 'Ubuntu', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        color: #ff6e9f !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(110, 72, 170, 0.4);
        font-family: 'Orbitron', sans-serif;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(110, 72, 170, 0.6);
    }
    
    .stSelectbox, .stNumberInput, .stTextInput {
        border-radius: 8px !important;
    }
    
    input, select {
        background-color: #1a1a2e !important;
        color: white !important;
        border: 1px solid #4e4e8d !important;
    }
    
    .st-bb, .st-at, .st-ag, .st-ah, .st-ae, .st-af, .st-div {
        border-color: #4e4e8d !important;
    }
    
    .css-1aumxhk {
        background-color: #1a1a2e;
        background-image: linear-gradient(to bottom, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .success-prediction {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.2) 0%, rgba(56, 142, 60, 0.3) 100%);
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0;
    }
    
    .failure-prediction {
        background: linear-gradient(135deg, rgba(198, 40, 40, 0.2) 0%, rgba(211, 47, 47, 0.3) 100%);
        border-left: 5px solid #F44336;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0;
    }
    
    .quote-box {
        font-style: italic;
        border-left: 3px solid #6e48aa;
        padding-left: 15px;
        margin: 15px 0;
        color: #b8b8ff;
    }
    
    .model-selector {
        background: linear-gradient(135deg, rgba(110, 72, 170, 0.2) 0%, rgba(157, 80, 187, 0.3) 100%);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .timeline-params {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.1) 0%, rgba(33, 150, 243, 0.2) 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #ff6e9f, #6e48aa);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 2px 10px rgba(110, 72, 170, 0.3);">
            🔮 Doctor Strange's Multiverse Predictor
        </h1>
        <p style="font-size: 1.2rem; color: #b8b8ff; max-width: 800px; margin: 0 auto;">
            Out of 14,000,605 timelines, only a few lead to <strong>victory</strong>. 
            Use ancient mysticism combined with machine learning to find the optimal path.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Model Selection ---
with st.container():
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        model_choice = st.selectbox(
            "🧠 Choose Prediction Algorithm", 
            ["Logistic Regression", "Random Forest", "Neural Network"],
            help="Select the mystical algorithm to scan the multiverse"
        )
    with col2:
        st.markdown("""
            <div style="padding-top: 10px;">
                <p style="color: #b8b8ff; margin: 0;">
                    <strong>Logistic Regression:</strong> Basic probability magic<br>
                    <strong>Random Forest:</strong> Ensemble of mystical predictions<br>
                    <strong>Neural Network:</strong> Deep arcane knowledge (recommended)
                </p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Load Model ---
model_paths = {
    "Logistic Regression": "multiverse_logistic.pkl",
    "Random Forest": "multiverse_random_forest.pkl",
    "Neural Network": "multiverse_neural_net.pkl"
}
model = joblib.load(model_paths.get(model_choice, "multiverse_random_forest.pkl"))

# --- Input Form ---
with st.form("timeline_form"):
    with st.container():
        st.markdown('<div class="timeline-params">', unsafe_allow_html=True)
        st.subheader("🌀 Timeline Parameters")
        st.markdown("Adjust the parameters to explore different timelines in the multiverse:")
        
        col1, col2 = st.columns(2)
        with col1:
            team_strength = st.slider("Team Strength", 0, 100, 60, help="Overall power level of your team")
            team_coordination = st.slider("Team Coordination", 0.0, 1.0, 0.75, 0.01, 
                                         help="How well your team works together")
            strategic_plan_complexity = st.slider("Strategic Plan Complexity", 1, 10, 6, 
                                                help="Complexity of your battle plan (1=simple, 10=complex)")
            diversion_success_rate = st.slider("Diversion Success Rate", 0.0, 1.0, 0.4, 0.01,
                                            help="Chance your diversions will work")
            intel_accuracy = st.slider("Intel Accuracy", 0.0, 1.0, 0.85, 0.01,
                                     help="Accuracy of your information about the enemy")
            previous_failures = st.slider("Previous Failures", 0, 10, 1,
                                        help="Number of previous failed attempts")
            universe_variability = st.slider("Universe Variability", 0.0, 1.0, 0.6, 0.01,
                                           help="How much the universe changes between timelines")

        with col2:
            enemy_strength = st.slider("Enemy Strength", 0, 100, 80, help="Overall power level of enemy")
            num_heroes = st.slider("Number of Heroes", 1, 20, 5, help="How many heroes in your team")
            num_enemies = st.slider("Number of Enemies", 1, 50, 10, help="How many enemies you face")
            enemy_stone_count = st.slider("Enemy Infinity Stones", 0, 6, 4,
                                        help="How many infinity stones the enemy has")
            has_time_stone = st.selectbox("Has Time Stone?", ["yes", "no"], 
                                         help="Do you have control of the Time Stone?")
            has_surprise_element = st.selectbox("Has Surprise Element?", ["yes", "no"],
                                              help="Do you have an element of surprise?")
            terrain_advantage = st.selectbox("Terrain Advantage?", ["yes", "no"],
                                           help="Does the terrain favor your team?")
            enemy_mind_state = st.selectbox("Enemy Mind State", ["confident", "hesitant", "arrogant", "fearful"],
                                          help="Current psychological state of the main enemy")
            has_ironman = st.selectbox("Has Ironman?", ["yes", "no"],
                                     help="Is Tony Stark involved in this timeline?")
            sacrifice_possible = st.selectbox("Sacrifice Possible?", ["yes", "no"],
                                            help="Is there potential for a heroic sacrifice?")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        submit = st.form_submit_button("🔮 Scan Multiverse", help="Initiate the multiverse prediction sequence")

# --- Prediction Logic ---
if submit:
    # --- Categorical Encoding ---
    binary_map = {"yes": 1, "no": 0}
    mind_state_map = {"confident": 0, "hesitant": 1, "arrogant": 2, "fearful": 3}

    input_dict = {
        "team_strength": team_strength,
        "enemy_strength": enemy_strength,
        "team_coordination": team_coordination,
        "strategic_plan_complexity": strategic_plan_complexity,
        "diversion_success_rate": diversion_success_rate,
        "intel_accuracy": intel_accuracy,
        "universe_variability": universe_variability,
        "previous_failures": previous_failures,
        "num_heroes": num_heroes,
        "num_enemies": num_enemies,
        "enemy_stone_count": enemy_stone_count,
        "has_time_stone": binary_map[has_time_stone],
        "has_surprise_element": binary_map[has_surprise_element],
        "terrain_advantage": binary_map[terrain_advantage],
        "enemy_mind_state": mind_state_map[enemy_mind_state],
        "has_ironman": binary_map[has_ironman],
        "sacrifice_possible": binary_map[sacrifice_possible]
    }

    try:
        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[0][1]
        outcome = "Victory" if prob >= 0.3 else "Defeat"

        # --- Animated Result Display ---
        st.markdown("""
            <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .result-animation {
                animation: fadeIn 1s ease-out;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.subheader("🔍 Multiverse Scan Results")
        
        if outcome == "Victory":
            st.markdown(f"""
                <div class="success-prediction result-animation">
                    <h3 style="color: #4CAF50;">✨ VICTORY TIMELINE DETECTED ✨</h3>
                    <p style="font-size: 1.1rem;">Probability of success: <strong style="color: #4CAF50; font-size: 1.3rem;">{prob*100:.1f}%</strong></p>
                    <div class="quote-box">
                        "We're in the endgame now." — <strong>Tony Stark</strong><br>
                        "I see only one possible path to victory." — <strong>Doctor Strange</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="failure-prediction result-animation">
                    <h3 style="color: #F44336;">💀 DEFEAT TIMELINE DETECTED 💀</h3>
                    <p style="font-size: 1.1rem;">Probability of success: <strong style="color: #F44336; font-size: 1.3rem;">{prob*100:.1f}%</strong></p>
                    <div class="quote-box">
                        "Dread it. Run from it. Destiny arrives all the same." — <strong>Thanos</strong><br>
                        "I'm sorry, Tony. There was no other way." — <strong>Doctor Strange</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # --- Interactive Timeline Visualization ---
        st.subheader("🌌 Multiverse Probability Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Probability Matrix", "Timeline Radar", "Outcome Distribution"])
        
        with tab1:
            # Bar Chart with custom design
            fig, ax = plt.subplots(figsize=(10, 4))
            values = [prob, 1 - prob]
            labels = ["Victory", "Defeat"] if outcome == "Victory" else ["Defeat", "Victory"]
            colors = ["#4CAF50", "#F44336"] if outcome == "Victory" else ["#F44336", "#4CAF50"]
            
            bars = ax.barh(labels, values, color=colors, height=0.6)
            for bar in bars:
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{bar.get_width()*100:.1f}%", 
                        va='center', color='white', fontsize=14, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_facecolor("#0f0c29")
            fig.patch.set_facecolor("#0f0c29")
            ax.spines['bottom'].set_color('#6e48aa')
            ax.spines['top'].set_color('#0f0c29') 
            ax.spines['right'].set_color('#0f0c29')
            ax.spines['left'].set_color('#6e48aa')
            ax.tick_params(axis='x', colors='#b8b8ff')
            ax.tick_params(axis='y', colors='#b8b8ff')
            ax.xaxis.label.set_color('#b8b8ff')
            plt.title('Timeline Outcome Probability', color='#ff6e9f', pad=20)
            st.pyplot(fig)
        
        with tab2:
            # Radar chart for key parameters
            categories = ['Team Strength', 'Enemy Strength', 'Coordination', 
                         'Intel Accuracy', 'Stones', 'Surprise']
            values_radar = [
                team_strength/100, 
                enemy_strength/100, 
                team_coordination,
                intel_accuracy,
                (6 - enemy_stone_count)/6,
                binary_map[has_surprise_element]
            ]
            
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            plt.xticks(angles[:-1], categories, color='white', size=12)
            ax.tick_params(axis='x', colors='#b8b8ff')
            ax.tick_params(axis='y', colors='#b8b8ff')
            
            values_radar += values_radar[:1]
            ax.plot(angles, values_radar, linewidth=2, linestyle='solid', 
                   color='#6e48aa', label="This Timeline")
            ax.fill(angles, values_radar, color='#6e48aa', alpha=0.4)
            
            ax.set_rlabel_position(30)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1], ["20%", "40%", "60%", "80%", "100%"], 
                       color="#b8b8ff", size=10)
            plt.ylim(0, 1)
            plt.title('Timeline Parameter Radar', color='#ff6e9f', pad=30, size=14)
            st.pyplot(fig)
        
        with tab3:
            # Animated pie chart
            fig = plt.figure(figsize=(8, 8), facecolor='none')
            ax = fig.add_subplot(111)
            
            if outcome == "Victory":
                colors = ['#4CAF50', '#F44336']
                explode = (0.1, 0)
            else:
                colors = ['#F44336', '#4CAF50']
                explode = (0.1, 0)
            
            wedges, texts, autotexts = ax.pie(
                [prob, 1-prob],
                explode=explode,
                labels=['Victory', 'Defeat'],
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                textprops={'color':"white", 'fontsize':14},
                wedgeprops = {'linewidth': 3, 'edgecolor': "#0f0c29"}
            )
            
            plt.setp(autotexts, size=16, weight="bold")
            plt.title('Multiverse Outcome Distribution', color='#ff6e9f', pad=30, size=14)
            st.pyplot(fig)
        
        # --- Timeline Recommendation ---
        st.subheader("🧭 Recommended Actions")
        
        if outcome == "Victory":
            if prob >= 0.7:
                st.success("""
                **Optimal Timeline Found!**  
                - Proceed with current strategy  
                - Maintain team coordination  
                - Protect the Time Stone at all costs  
                - Prepare for potential sacrifices
                """)
            else:
                st.warning("""
                **Marginal Victory Timeline**  
                - Consider small adjustments to improve odds  
                - Increase diversion tactics  
                - Gather more intelligence  
                - Secure additional allies if possible
                """)
        else:
            if prob <= 0.1:
                st.error("""
                **Critical Failure Timeline**  
                - Abandon this path immediately  
                - Re-evaluate all strategic assumptions  
                - Seek alternative approaches  
                - Consider unconventional solutions
                """)
            else:
                st.info("""
                **Potential Recovery Possible**  
                - Major strategic changes needed  
                - Consider sacrificing the Time Stone  
                - Look for unexpected allies  
                - Exploit enemy weaknesses more aggressively
                """)

    except Exception as e:
        st.error(f"""
            <div style="background-color: #330000; padding: 20px; border-radius: 12px;">
                <h3 style="color: #ff4444;">🛑 Multiverse Scan Failed</h3>
                <p>The Eye of Agamotto could not process these parameters:</p>
                <code style="color: #ff9999;">{e}</code>
                <p>Please adjust your inputs and try again.</p>
            </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #6e48aa; font-size: 0.9rem;">
        <hr style="border: 1px solid #6e48aa; opacity: 0.3;">
        <p>Doctor Strange's Multiverse Predictor v2.0 • Powered by the Time Stone • Sanctum Sanctorum Technologies</p>
        <p>Warning: Viewing alternate timelines may cause temporal disorientation</p>
    </div>
""", unsafe_allow_html=True)