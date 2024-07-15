from fastapi import APIRouter, File, HTTPException, status, UploadFile
from fastapi.responses import JSONResponse
import aiofiles
from src.chat.local_llm import ask_llm
from src.utils.misc import generate_session_id, validate_session_id
from src.utils.s3 import upload_to_s3
from src.models.chat_message import ChatMessage


v1_router = APIRouter(prefix="/v1")


@v1_router.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        async with aiofiles.open(file.filename, 'wb') as ff:
            while contents := await file.read(1024 * 1024):
                await ff.write(contents)
        file_name, s3_path = await upload_to_s3(file.filename)
        response = {
            "file_name": file_name,
            "file_path": s3_path
        }
    except ModuleNotFoundError:
        # Use local file system
        response = {
            "file_name": file.filename.split("/")[-1],
            "file_path": file.filename
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    return JSONResponse(content=response)


@v1_router.post("/chat")
async def create_chat_msg(chats: ChatMessage):
    try:
        if not chats.session_id:
            chats.session_id = await generate_session_id()

        payload = ChatMessage(
            session_id=chats.session_id,
            user_input=chats.user_input,
            data_source=chats.data_source
        )
        data = payload.model_dump()
        response = ask_llm(data.get("data_source"),
                           data.get("user_input"))
        return JSONResponse(
            content={
                "session_id": chats.session_id,
                "response": response
            }
        )
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT, detail=str(ex))
