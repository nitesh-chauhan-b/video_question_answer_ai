import streamlit as st
import tempfile
import os
#For removing the error
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNING"] = "true"

# The trick is to import before the chromadb is imported in any files
# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import helper

st.set_page_config(layout="wide")  # Enable wide layout
st.title("🎥 Question Answer With Video")

# Upload video
uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    # Get video path
    video_path = tfile.name

    # Layout: Two columns
    col1, col2 = st.columns([2, 2])  # 2:1 width ratio

    # Column 1: Video preview
    with col1:
        st.video(video_path)
        st.success(f"Uploaded: {uploaded_file.name}")

    # Column 2: Q&A section
    with col2:

        #Getting video ready for question and answers
        try:
            # Getting transcription from the video
            video_transcript = helper.get_video_content(video_path)
            # print(video_transcript)

            # Storing the data
            helper.store_data(video_transcript)

            # Getting Question answer chain
            # if db:
            chain = helper.get_qa_chain()

            print("Success!!")

            st.header("Question & Answer")
            query = st.text_input("Enter Your Question")
            submit = st.button("Submit")

            if query and submit:

                #Getting the answer
                response = chain(query)

                st.header("Answer")

                st.markdown(response["result"],unsafe_allow_html=True)

        except Exception as e:
            print("Error : ", e)


