from fastapi import APIRouter, File, UploadFile, HTTPException
import os
from uuid import uuid4
import sys
from pydantic import BaseModel
from Image_Analysis.Image_search.image import image_analysis
from Chatbot.Main import main
import json

router = APIRouter()


# ---------------------------
class ChatRequest(BaseModel):
    thread_id: str
    user_input: str

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)






@router.post("/image-analyze")
async def upload_file(file: UploadFile = File(...)):
    # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
    
        # Generate unique filename
        unique_filename = f"{uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        file_content = await file.read()
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Process the image and get analysis result
        result = image_analysis(file_path)

        result_json = json.loads(result)

        return {"data": result_json}

@router.post("/chat")
async def chat_with_bot(chat_request: ChatRequest):
    # Here you would integrate with your chatbot logic

    response = main(chat_request.thread_id, chat_request.user_input)

    response_json = json.loads(response)

    return {"data": response_json}
