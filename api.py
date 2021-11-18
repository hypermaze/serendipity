import json
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sentence_transformers import SentenceTransformer


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8081",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("roam_embeddings.json", "r") as file:
    roam_index = json.loads(f.read())
    embeddings = np.array([obj.get("embedding") for obj in roam_index], dtype=np.float32)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# PAYLOAD CLASSES
class RoamParams(BaseModel):
    sentence_text: str
    k: int


class ExportParams(BaseModel):
    data: dict
    user: str


# HELPERS
def wrap_response(response):
    return json.dumps({"payload": response})

@app.post("/similar")
def similar_sentences(params: RoamParams):
    """returns the k most similar sentences in the graph"""
    print(f"Sentence: {params.sentence_text}")
    results = query_index(params.sentence_text, model, roam_embeddings, index, k=params.k)

    return wrap_response([{"uid": result.get("uid"), "sentence": result.get("sentence")} for result in results])


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    load_dotenv('.env')
    uvicorn.run(app, host="0.0.0.0", port=8080)
