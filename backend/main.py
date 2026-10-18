import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

load_dotenv()

app = FastAPI(title="Mixr")

# Comma-separated frontend origins, e.g. "https://mixr.vercel.app,http://localhost:3000".
cors_origins = [
    o.strip() for o in os.getenv("MIXR_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # Optional, for Vercel preview URLs, e.g. "https://mixr-.*\.vercel\.app".
    allow_origin_regex=os.getenv("MIXR_CORS_ORIGIN_REGEX") or None,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


app.include_router(router, prefix="/api")
