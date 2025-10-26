from fastapi import FastAPI
from .routes import router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://10.10.7.75:5173",
        "http://10.10.7.75:8003",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"Hello": "World"}



app.include_router(router=router, prefix="/api/v1")