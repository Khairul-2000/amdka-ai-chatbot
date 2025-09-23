from fastapi import FastAPI
from .routes import router


app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}



app.include_router(router=router, prefix="/api/v1")