import gradio as gr
import whisper
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
from langchain_chroma import Chroma
import textwrap

#Using faster whisper model
from faster_whisper import WhisperModel

#Loading model
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",

)

#Embeddings for vector db
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    # temperature=0.6,
    max_tokens=1000
)

qa_pipeline = pipeline("question-answering")

# Storage for transcribed text
transcript_text = ""

#Chian
chain = None

def get_video_content(path):
    # Original Whisper model
    #It uses the segments and to divide video into segments
    segments,info = model.transcribe(
        path,
        beam_size=1,
        word_timestamps=False
    )

    #Combining segments to get the transcribes test
    video_transcript = "".join([seg.text for seg in segments])

    print(video_transcript)
    return video_transcript


def store_data(video_transcription):
    splitter = RecursiveCharacterTextSplitter(
        separators=[".", "\n", "\n\n"],
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_text(video_transcription)

    # db = Chroma.from_texts(chunks, embedding, collection_name="video_transcription", persist_directory="./chroma_db")
    db = Chroma.from_texts(chunks, embedding, collection_name="video_transcription")

    return db

def get_qa_chain(db):
    # if os.path.exists("vectore_db.pkl"):
    #     with open("vectore_db.pkl", "rb") as file:
    #         # db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    #         db = pickle.load(file)

    # Getting db as retriever
    retriever = db.as_retriever()

    # Creating a template for getting answers
    # template = """Given the following context and a question, generate an answer based on this context only.
    #     In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    #     If context doesn't contain answer then answer in short form about i don't know.
    #     Try to give examples to explain the question better.
    #
    #     CONTEXT: {context}
    #
    #     QUESTION: {question}
    # """

    template = """Given the question, provide a detailed, easy-to-understand answer based on the content in the video.
        The answer should be as direct and informative as possible without referencing the specific context.
        If an example is provided, format it clearly to make it easier to follow.
        If the content doesn’t answer the question directly, provide a concise and polite response indicating that the answer is unknown.

        Content: {context}

        QUESTION: {question}

        Ensure the response is clear, with simple explanations and structured examples if necessary. Avoid making unnecessary references to the source material directly.
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Creating a question answer chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        # return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain

def handle_video_change(video_input):
    global transcript_text
    transcript_text = get_video_content(video_input)

    #Storing the data
    db = store_data(transcript_text)

    #Getting question answer chain
    global  chain
    chain = get_qa_chain(db)

    return transcript_text

def answer_question(question):
    if not transcript_text:
        return "Please upload a video and wait for transcription first."

    #getting the global chain
    global  chain

    if chain is not None:

        result = chain(question)

        #For formatting the answer
        answer_text = result["result"]
        md = f"""# Answer
        
{answer_text}

---
_This answer is auto-generated from the video transcription._"""

        return textwrap.dedent(md)
        # return

    return "Sorry, Something went wrong"

with gr.Blocks() as app:
    gr.Markdown("## 🎥 Video QA App")

    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Upload Video")
            transcribed_output = gr.Textbox(label="Transcribed Text", lines=10)

        with gr.Column(scale=2):
            question_input = gr.Textbox(label="Ask a question about the video")
            ask_button = gr.Button("Get Answer")
            answer_output = gr.Markdown(label="Answer")

    video_input.change(fn=handle_video_change, inputs=video_input, outputs=transcribed_output)
    ask_button.click(fn=answer_question, inputs=question_input, outputs=answer_output)


if __name__ =="__main__":
    app.launch()
