import uuid

from pydantic import ValidationError


async def generate_session_id() -> str:
    return str(uuid.uuid4())


async def validate_session_id(session_id: str) -> str:
    try:
        session_id = str(uuid.UUID(session_id))
    except Exception:
        raise ValidationError("Not a valid uuid")

    return session_id
