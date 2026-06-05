import streamlit as st
import pandas as pd
import numpy as np
import random
import os

st.set_page_config(page_title="RL Dynamic Pricing", layout="centered")
st.title("💰 Dynamic Pricing using Reinforcement Learning")

# ---- LOAD DATA ----
file_name = "Online Retail.csv"

if not os.path.exists(file_name):
    st.error("Dataset not found")
    st.stop()

data = pd.read_csv(file_name, encoding='latin1')
data = data[['UnitPrice', 'Quantity']].dropna()
data = data[data['Quantity'] > 0].head(100)

# ---- Q TABLE ----
q_table = np.zeros((len(data), 3))

alpha = 0.1
gamma = 0.9
epsilon = 0.9   # HIGH exploration

episodes = st.sidebar.slider("Episodes", 100, 1000, 500)

# ---- TRAIN ----
if st.button("Train Model"):
    for _ in range(episodes):
        state = random.randint(0, len(data)-1)

        for _ in range(10):

            # RANDOM ACTION MOSTLY
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                action = np.argmax(q_table[state])

            price = data.iloc[state]['UnitPrice']
            quantity = data.iloc[state]['Quantity']

            # BALANCED EFFECT (KEY FIX 🔥)
            if action == 0:  # decrease
                new_price = price * 0.95
                reward = new_price * quantity

            elif action == 2:  # increase
                new_price = price * 1.05
                reward = new_price * quantity

            else:  # same
                new_price = price
                reward = new_price * quantity

            # SMALL RANDOM BOOST (IMPORTANT)
            reward += random.uniform(-5, 5)

            # UPDATE
            q_table[state][action] += alpha * (
                reward - q_table[state][action]
            )

    st.success("Training Done!")

# ---- PREDICT ----
if st.button(" Suggest Price"):

    state = random.randint(0, len(data)-1)

    # SOFTMAX (IMPORTANT 🔥)
    exp_q = np.exp(q_table[state])
    probs = exp_q / np.sum(exp_q)

    action = np.random.choice([0,1,2], p=probs)

    price = data.iloc[state]['UnitPrice']

    if action == 0:
        suggestion = price * 0.95
        decision = "Decrease Price 📉"
    elif action == 2:
        suggestion = price * 1.05
        decision = "Increase Price 📈"
    else:
        suggestion = price
        decision = "Keep Same ➖"

    st.subheader("Result")
    st.write("Original Price:", round(price,2))
    st.write("Suggested Price:", round(suggestion,2))
    st.write("Decision:", decision)

    st.write("Probabilities:", {
        "Decrease": round(probs[0],2),
        "Same": round(probs[1],2),
        "Increase": round(probs[2],2)
    })