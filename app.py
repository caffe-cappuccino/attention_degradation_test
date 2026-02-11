import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Cognitive Attention Assessment System", layout="centered")

# -------------------------
# Load & Train Model
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("attention_dataset_3200_realistic.csv")

df = load_data()

X = df.drop(columns=["session_id", "attention_label"])
y = df["attention_label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Session State Init
# -------------------------
if "stage" not in st.session_state:
    st.session_state.stage = 1

if "reaction_times" not in st.session_state:
    st.session_state.reaction_times = []

if "lapses" not in st.session_state:
    st.session_state.lapses = 0

if "switch_errors" not in st.session_state:
    st.session_state.switch_errors = 0

if "trial_count" not in st.session_state:
    st.session_state.trial_count = 0

# -------------------------
st.title("🧠 Multi-Test Attention Assessment System")

# =========================
# TEST 1 – Reaction Speed
# =========================
if st.session_state.stage == 1:

    st.header("Test 1: Reaction Speed")

    if st.button("Start Trial"):
        wait = random.uniform(1, 3)
        time.sleep(wait)
        start = time.time()
        st.write("Click NOW!")

        if st.button("Click"):
            rt = (time.time() - start) * 1000
            st.session_state.reaction_times.append(rt)
            st.success(f"RT: {round(rt,2)} ms")

            if len(st.session_state.reaction_times) >= 6:
                st.session_state.stage = 2
                st.rerun()

# =========================
# TEST 2 – Sustained Focus
# =========================
elif st.session_state.stage == 2:

    st.header("Test 2: Sustained Focus")

    number = random.randint(1, 9)
    st.write(f"Click ONLY if number is EVEN: {number}")

    if st.button("Respond"):
        if number % 2 != 0:
            st.session_state.lapses += 1
            st.warning("Incorrect (Lapse)")
        else:
            st.success("Correct")

        st.session_state.trial_count += 1

    if st.session_state.trial_count >= 6:
        st.session_state.stage = 3
        st.rerun()

# =========================
# TEST 3 – Task Switching
# =========================
elif st.session_state.stage == 3:

    st.header("Test 3: Task Switching")

    rule = random.choice(["Color Rule", "Number Rule"])
    value = random.randint(1, 9)

    st.write(f"Rule: {rule}")
    st.write(f"Value: {value}")

    if st.button("Respond to Rule"):
        if rule == "Number Rule" and value % 2 != 0:
            st.session_state.switch_errors += 1
        if rule == "Color Rule" and value % 2 == 0:
            st.session_state.switch_errors += 1

        st.session_state.trial_count += 1

    if st.session_state.trial_count >= 12:
        st.session_state.stage = 4
        st.rerun()

# =========================
# FINAL EVALUATION
# =========================
elif st.session_state.stage == 4:

    st.header("Final Cognitive Evaluation")

    mean_rt = np.mean(st.session_state.reaction_times)
    rt_std = np.std(st.session_state.reaction_times)
    lapse_ratio = st.session_state.lapses / 6
    switch_error_ratio = st.session_state.switch_errors / 6

    # Fatigue index
    first_half = np.mean(st.session_state.reaction_times[:3])
    second_half = np.mean(st.session_state.reaction_times[3:])
    fatigue_index = (second_half - first_half) / first_half

    consistency_score = 1 / (1 + rt_std)

    degradation_score = (
        0.35 * mean_rt +
        0.25 * rt_std +
        0.20 * lapse_ratio * 1000 +
        0.20 * fatigue_index * 1000
    )

    input_data = pd.DataFrame([{
        "mean_rt": mean_rt,
        "rt_std": rt_std,
        "max_rt": mean_rt + rt_std,
        "min_rt": mean_rt - rt_std,
        "lapse_count": lapse_ratio * 6,
        "lapse_ratio": lapse_ratio,
        "fatigue_index": fatigue_index,
        "consistency_score": consistency_score,
        "degradation_score": degradation_score
    }])

    prediction = model.predict(input_data)
    label = le.inverse_transform(prediction)[0]

    st.success(f"Detected Attention State: {label}")

    # ---------------- Advice
    st.subheader("Personalized Advice")

    if label == "Normal":
        st.write("✔ Maintain structured work blocks.")
    elif label == "Mild":
        st.write("⚠ Introduce 25/5 focus cycles.")
        st.write("Reduce task switching.")
    else:
        st.write("❗ Take 20–30 min rest.")
        st.write("Avoid complex cognitive tasks.")

    # ---------------- Routine Generator
    st.subheader("Generated Routine")

    if label == "Normal":
        st.write("• 40 min focus / 10 min break")
        st.write("• High complexity in morning")
    elif label == "Mild":
        st.write("• 25 min focus / 5 min break")
        st.write("• Medium complexity tasks only")
    else:
        st.write("• 15 min light tasks")
        st.write("• 10 min recovery breaks")
        st.write("• Resume heavy tasks next session")

    if st.button("Restart"):
        st.session_state.clear()
        st.rerun()
