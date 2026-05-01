from fastapi import FastAPI
from pydantic import BaseModel
from rag import load_transcript, answer

app = FastAPI()

chain = None  # store chain in memory

class VideoRequest(BaseModel):
    video_id: str
    language: str

class QuestionRequest(BaseModel):
    question: str

@app.get('/')
def home():
    return {'message':'This is a YouTube RAG Chatboat API'}
    

@app.get('/health')
def health_check():
    return {
        'status':'ok',
        'version': '1.0.0',
        }


@app.post("/load_video")
def load_video(request: VideoRequest):
    global chain
    chain = load_transcript(request.video_id,request.language)  # ← your function
    print(chain)
    if chain is None:
        return {"success": False, "message": "Could not fetch transcript."}

    return {"success": True, "message": "Video loaded successfully!"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    global chain

    if chain is None:
        return {"answer": "Please load a video first."}

    result = answer(request.question, chain)   # ← your function
    return {"answer": result}