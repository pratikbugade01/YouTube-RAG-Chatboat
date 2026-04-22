import streamlit as st
import requests

# Backend URL
BASE_URL = "http://127.0.0.1:8000/"

st.set_page_config(page_title="YouTube RAG Chat", layout="centered")

st.title("🎥 YouTube Chatbot (RAG)")
st.write("Ask questions from any YouTube video transcript")

# -------------------------
# Load Video Section
# -------------------------
video_id = st.text_input("Enter YouTube Video ID")

if st.button("Load Video"):
    if video_id:
        with st.spinner("Loading transcript..."):
            response = requests.post(
                f"{BASE_URL}/load_video",
                json={"video_id": video_id}
            )

            data = response.json()

            if data["success"]:
                st.success(data["message"])
                st.session_state["loaded"] = True
            else:
                st.error(data["message"])
    else:
        st.warning("Please enter a video ID")

# -------------------------
# Ask Question Section
# -------------------------
if st.session_state.get("loaded", False):

    st.subheader("Ask a Question")

    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{BASE_URL}/ask",
                    json={"question": question}
                )

                answer = response.json()["answer"]

                st.markdown("### 💡 Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question")