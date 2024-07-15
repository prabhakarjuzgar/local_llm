from pydantic import BaseModel, Field


class ChatMessage(BaseModel):

    session_id: str = Field(default=None)
    user_input: str
    data_source: str
