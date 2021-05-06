import streamlit as st
import SessionState
import os
import tensorflow as tf
from cloud_service import predict_json, update_logger

CLASSES = ['battery',
           'brown-glass',
           'cardboard',
           'green-glass',
           'metal',
           'paper',
           'paper-cup', 'white-glass']

RECYCLE_BIN = [
    "green-glass", "brown-glass", "white-glass", "glass", "cardboard", "metal", "paper"
]
NON_RECYCLE_BIN = [
    "battery", "paper-cup", "clothes", "metal-cap", "plastic-bag"
]
# Configure Cloud variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "waste-classifier-312002-cb19c8542cc6.json"
PROJECT = "waste-classifier-312002"
REGION = "asia-east1"
MODEL = "sepm_waste_vision"
def load_and_prep_image(filename, img_shape=299, rescale=False):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (299, 299, 3).
  """
  # Decode it into a tensor
#   img = tf.io.decode_image(filename) # no channels=3 means model will break for some PNG's (4 channels)
  img = tf.io.decode_image(filename, channels=3) # make sure there's 3 colour channels (for PNG's)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  # Rescale the image (get all values between 0 and 1)
  if rescale:
      return img/255.
  else:
      return img
@st.cache
def make_prediction(image, model, class_names):
    """
    Take input image and use utils function to send request to Cloud-based model
    and return a prediction
    :param image: streamlit image read type
    :param model: model name in the cloud
    :param class_names: class names
    :return:
        image (preprocessed),
        pred_class (predicted class),
        pre_conf (confidence)
    """
    image_tensor = load_and_prep_image(image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    preprocessed_img = tf.cast(tf.expand_dims(image_tensor, axis=0), tf.int16)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=preprocessed_img)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    return image_tensor, pred_class, pred_conf

def get_bin(pred_class):
    if pred_class in "\t".join(NON_RECYCLE_BIN):
        st.write("belongs to **Non-Recycle bin**")
    elif pred_class in RECYCLE_BIN:
        st.write("belongs to **Recycle bin**")
    else:
        st.write("belongs to **General bin**")

# STREAMLIT APP STARTS HERE
st.title("Welcome to Waste Classifier site!")
st.header("Upload a image of your waste and we will tell you which bin it should be put into")
# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose {MODEL}, these are  classes of material or object it can identify:\n", CLASSES)

uploaded_file = st.file_uploader("Pick image of your trash", type=["jpg", "jpeg", "png"])

# Remember the state of the program
session_state = SessionState.get(pred_button=False)

if not uploaded_file:
    st.warning("Please upload an image")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# did they just press prediction button
if pred_button:
    session_state.pred_button = True

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(
        session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.3f}")
    # TODO Predicted class to correct bin
    get_bin(session_state.pred_class)

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Is this correct?",
        ("Select an option", "Yes", "No"))
    if session_state.feedback == "Select an option":
        pass
    elif session_state.feedback == "Yes":
        st.write("Thank you for your feedback!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=session_state.image,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "No":
        session_state.correct_class = st.text_input("What should the correct label be?")
        if session_state.correct_class:
            st.write("Thank you for that, we'll use your help to make our model better!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))
            # map the correct label object to correct bin
            get_bin(session_state.correct_class)
