from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def get_transcript(video_id,language):
   
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id=video_id, languages=[language])

    # Support both dict-style and object-style transcript snippets
        transcript = " ".join(
            chunk["text"] if isinstance(chunk, dict) else chunk.text
            for chunk in transcript_list
        )
        return transcript

    except TranscriptsDisabled:
        return "No captions available for this video."
    except Exception as e:
        return f"Failed to fetch transcript: {e}"


def split_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    return chunks

def create_retriever(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever   


def initialize_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key
        )
    return llm

def create_prompt():
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        context:
        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )
    return prompt


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


def build_chain(retriever,prompt,llm):

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain


def load_transcript(video_id,language):

    video_id = video_id.strip()

    transcript = get_transcript(video_id,language)

    if transcript.startswith("Failed") or transcript.startswith("No captions"):
        return None

    chunks = split_transcript(transcript)

    retriever = create_retriever(chunks)

    llm = initialize_llm()
    prompt = create_prompt()

    chain = build_chain(retriever, prompt, llm)
    return chain


def answer(question,chain):
    answer = chain.invoke(question)
    return answer