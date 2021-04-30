import streamlit as st
import SessionState
st.title("Welcome to Waste Classifier site!")
st.header("Type wastes: glass, cardboard, paper, metal, paper-cup, battery")
st.text("Upload a image of your waste and we will tell you which bin it should be put into")

uploaded_file = st.file_uploader("Pick image of your trash", type=["jpg", "jpeg"])

# Remember the state of the program
session_state = SessionState.get(pred_button=False)

if not uploaded_file:
    st.warning("Please upload an image")
    st.stop()
else:
    session_state.upload_image = uploaded_file.read()
    st.image(session_state.upload_image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    session_state.pred_button = True
