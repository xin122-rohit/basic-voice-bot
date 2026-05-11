import os
from dotenv import load_dotenv
from livekit import api

load_dotenv()


def create_token(email: str, room: str):
    token = (
        api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
        .with_identity(email)
        .with_name("User Name")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )

    return token
