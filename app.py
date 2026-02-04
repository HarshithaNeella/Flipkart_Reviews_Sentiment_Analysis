import streamlit as st
import pickle
import base64


# ------------------ Page Config ------------------

st.set_page_config(
    page_title="Flipkart Reviews",
    page_icon="🛒",
    layout="centered"
)


# ------------------ Background Image ------------------

def set_background(image_file):

    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Make sure this file exists in project folder
set_background("flipkart_image.png")


# ------------------ Load Model ------------------

@st.cache_resource
def load_model():

    with open("flipkart_LSVM.pkl", "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# ------------------ UI Header ------------------

st.markdown(
    """
    <h1 style='
        color:#2874F0;
        font-family: Arial, Helvetica, sans-serif;
        font-weight:600;
        text-align:left;
    '>
     🛒&nbsp;Flipkart&nbsp;Reviews Classification
    </h1>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <p style='
        color:#2874F0;
        text-align:center;
        font-size:18px;
        margin-top:-10px;
    '>
    Predict the sentiment from Flipkart product reviews
    </p>
    """,
    unsafe_allow_html=True
)


# ------------------ Input ------------------

user_input = st.text_area(
    "✍️ Enter Review",
    placeholder="Good quality product..."
)


# ------------------ Prediction ------------------

if st.button("🔍 Predict"):

    if user_input.strip() == "":

        st.warning("⚠️ Please enter some text.")

    else:

        text = user_input.strip().lower()


        # Rule phrases

        positive_phrases = [
            "i like", "i love", "very good", "excellent",
            "awesome", "perfect", "good quality",
            "high quality", "worth it"
        ]


        negative_phrases = [
            "don't like", "do not like", "not good",
            "waste of", "worst", "very bad",
            "poor quality", "not worth", "bad quality"
        ]


        with st.spinner("Analyzing review..."):

            override = False


            # Priority: Negative first

            if any(p in text for p in negative_phrases):

                prediction = "Negative"
                confidence = 0.95
                override = True


            elif any(p in text for p in positive_phrases):

                prediction = "Positive"
                confidence = 0.95
                override = True


            else:

                prediction = model.predict([user_input])[0]

                score = model.decision_function([user_input])[0]

                confidence = min(1.0, abs(score) / 3)


        # ------------------ Result ------------------

        if prediction == "Positive":

            st.success(
                f"✅ Sentiment: **{prediction}**  \n📊 Confidence: {confidence*100:.1f}%"
            )

        else:

            st.error(
                f"❌ Sentiment: **{prediction}**  \n📊 Confidence: {confidence*100:.1f}%"
            )


        if override:

            st.info("ℹ️ Rule-based override applied.")
