from dotenv import load_dotenv

load_dotenv()

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    get_job_context,
    cli,
    room_io,
    AutoSubscribe,
)
from livekit.agents.llm import ImageContent

# from livekit_bot.livekit_agent_functions import ClaimTools
from jinja2 import Environment, FileSystemLoader
from livekit.plugins import azure, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import os
from pathlib import Path
import asyncio
import json
import base64
import uuid

TMP_DIR_BASE = "tmp_uploaded_images"
os.makedirs(TMP_DIR_BASE, exist_ok=True)

# ---------------------------------------------------------------------------
# Session-isolation helpers
# ---------------------------------------------------------------------------


def _sanitize_path_component(value: str) -> str:
    """Replace characters unsafe for directory names with underscores."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _session_image_dir(room_name: str, participant_identity: str) -> str:
    """
    Return a filesystem path that is unique per (room, participant) pair.

    Format: tmp_uploaded_images/<room>/<participant>/
    This prevents images from two different users/sessions sharing the same
    directory even if the agent worker is reused across back-to-back jobs.
    """
    safe_room = _sanitize_path_component(room_name)
    safe_participant = _sanitize_path_component(participant_identity)
    return os.path.join(TMP_DIR_BASE, safe_room, safe_participant)


def _resolve_user_email(
    default_email: str,
    room_name: str,
    remote_participants: dict,
) -> tuple[str, str]:
    """
    Derive the primary user email and participant identity for this room.

    Priority:
    1. Room name contains the email tag — rooms are named `voice-<emailTag>-<sessionId>`
       by the frontend, so we first look for a remote participant whose identity
       starts with the email tag extracted from the room name.
    2. Exact email match — participant identity contains "@".
    3. Fallback to `default_email`.

    Returns (user_email, participant_identity).
    """
    # Try to extract an email-tag prefix from the room name (format: voice-<tag>-<sid>)
    # parts = room_name.split("-")
    # room_email_tag = parts[1].lower() if len(parts) >= 3 and parts[0] == "voice" else ""

    best_email = default_email
    best_identity = default_email

    for identity in remote_participants:
        if "@" in identity:
            # candidate_tag = (
            #     identity.split("@")[0].lower().replace(".", "").replace("_", "")
            # )
            # if room_email_tag and candidate_tag.startswith(room_email_tag[:8]):
            #     # Strongest match: email tag from room name aligns with participant
            #     return identity, identity
            best_email = identity
            best_identity = identity

    return best_email, best_identity


env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "prompts"), autoescape=True
)
prompt = env.get_template("livekit_prompt.jinja2").render()

azure_llm = openai.LLM.with_azure(
    model=os.getenv("AZURE_OPENAI_MODEL"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
    # parallel_tool_calls=False,
    timeout=120.0,
)

SUPPORTED_LANGUAGES = [
    "en-US",
]

stt = azure.STT(
    speech_key=os.getenv("AZURE_SPEECH_KEY"),
    speech_region=os.getenv("AZURE_SPEECH_REGION"),
    language="en-US",
)

tts = azure.TTS(
    speech_key=os.getenv("AZURE_SPEECH_KEY"),
    speech_region=os.getenv("AZURE_SPEECH_REGION"),
    voice="en-US-JennyNeural",
)


BATCH_WAIT_SECONDS = 3.0


class Assistant(Agent):
    def __init__(
        self,
        user_email: str,
        session_image_dir: str,
        active_participant_identity: str,
    ) -> None:
        self.session_image_dir = session_image_dir
        os.makedirs(self.session_image_dir, exist_ok=True)
        # Store the identity of the human participant this assistant instance
        # should serve.  Byte-stream uploads from other identities are ignored
        # to prevent cross-session image mixing.
        self.active_participant_identity = active_participant_identity
        # self.claim_tools = ClaimTools(user_email, image_dir=session_image_dir)
        self._tasks = []
        self._pending_images: list[str] = []
        self._batch_timer: asyncio.TimerHandle | None = None
        self._batch_lock = asyncio.Lock()
        super().__init__(
            instructions=prompt,
            tools=[
                # self.claim_tools.get_user_policies,
                # self.claim_tools.initialize_claim,
                # self.claim_tools.save_incident_details,
                # # self.claim_tools.process_images,
                # self.claim_tools.assess_damage,
            ],
        )
        print("[ASSISTANT] Initialized with tools")
        print(f"[ASSISTANT] User email: {user_email}")
        print(f"[ASSISTANT] Active participant identity: {active_participant_identity}")

    async def on_enter(self):
        await super().on_enter()
        print("[ASSISTANT] on_enter called")

    async def _image_received(self, reader, participant_identity):
        # Drop images from participants that do not belong to this assistant's session.
        # This is the key guard that prevents cross-session image contamination when
        # multiple users are connected to the LiveKit cluster simultaneously.
        if participant_identity != self.active_participant_identity:
            print(
                f"[IMAGE] Ignoring byte stream from {participant_identity!r} — "
                f"expected {self.active_participant_identity!r} (cross-session guard)"
            )
            return

        try:
            image_bytes = bytes()
            async for chunk in reader:
                image_bytes += chunk

            if not image_bytes:
                print("[IMAGE] Received empty byte stream, ignoring")
                return

            ext = self._detect_image_format(image_bytes) or "png"

            filename = f"{uuid.uuid4().hex}.{ext}"
            stored_path = os.path.join(self.session_image_dir, filename)
            with open(stored_path, "wb") as f:
                f.write(image_bytes)

            print(f"[IMAGE] Saved to {stored_path}")

            async with self._batch_lock:
                if not self._pending_images:
                    for old_file in os.listdir(self.session_image_dir):
                        old_path = os.path.join(self.session_image_dir, old_file)
                        if old_path != stored_path:
                            os.remove(old_path)

                self._pending_images.append(stored_path)

                if self._batch_timer is not None:
                    self._batch_timer.cancel()

                loop = asyncio.get_running_loop()
                self._batch_timer = loop.call_later(
                    BATCH_WAIT_SECONDS,
                    lambda: asyncio.ensure_future(self._process_image_batch()),
                )

            print(
                f"[IMAGE] Queued ({len(self._pending_images)} pending), waiting for more..."
            )

        except Exception as e:
            print(f"[IMAGE] Error handling image: {e}")
            import traceback

            traceback.print_exc()

    async def _process_image_batch(self):
        async with self._batch_lock:
            paths = list(self._pending_images)
            self._pending_images.clear()
            self._batch_timer = None

        if not paths:
            return

        print(f"[IMAGE] Processing batch of {len(paths)} images")

        self.claim_tools.latest_image_data = {
            "paths": paths,
            "type": "image",
        }

        chat_ctx = self.chat_ctx.copy()
        content = [f"User uploaded {len(paths)} image(s) for the insurance claim."]
        for path in paths:
            with open(path, "rb") as f:
                img_bytes = f.read()
            ext = self._detect_image_format(img_bytes) or "png"
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
            }.get(ext, "image/png")
            b64_url = (
                f"data:{mime};base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            )
            content.append(ImageContent(image=b64_url))
        chat_ctx.add_message(role="user", content=content)
        await self.update_chat_ctx(chat_ctx)

        try:
            result = await self.claim_tools.process_images(images=paths)
            print(f"[IMAGE] process_images result: {result}")
        except Exception as e:
            print(f"[IMAGE] Error in process_images: {e}")
            import traceback

            traceback.print_exc()

    @staticmethod
    def _detect_image_format(image_bytes: bytes) -> str | None:
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "jpg"
        elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        elif image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
            return "gif"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
            return "webp"
        return None


async def entrypoint(ctx: JobContext):
    print("Agent received job - Connecting to room...")

    await ctx.connect(
        auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL,
    )

    room_name = ctx.room.name
    print(f"Successfully joined room: {room_name}")

    # ---------------------------------------------------------------------------
    # Resolve the owning user using session-isolation helpers.
    #
    # Rooms are named `voice-<emailTag>-<sessionId>` by the frontend, so we can
    # cross-reference the room name against remote participant identities to find
    # the exact human this agent instance should serve.
    # ---------------------------------------------------------------------------
    DEFAULT_EMAIL = "rohit.kaushal@xebia.com"
    active_participant_identity = _resolve_user_email(
        default_email=DEFAULT_EMAIL,
        room_name=room_name,
        remote_participants=ctx.room.remote_participants,
    )
    user = ctx.room.remote_participants
    # user_email = user[0]
    user_email = next(iter(user))
    # active_participant_identity = next(iter(user))

    print(f"[ENTRYPOINT] room={room_name}")
    print(f"[ENTRYPOINT] user_email={user_email}")
    print(f"[ENTRYPOINT] active_participant_identity={active_participant_identity}")

    # Image directory is scoped to (room, participant) — no two sessions ever
    # share the same path even on a single worker process.
    session_image_dir = _session_image_dir(room_name, active_participant_identity)

    assistant = Assistant(
        user_email=user_email,
        session_image_dir=session_image_dir,
        active_participant_identity=active_participant_identity,
    )

    def _image_received_handler(reader, participant_identity):
        task = asyncio.create_task(
            assistant._image_received(reader, participant_identity)
        )
        assistant._tasks.append(task)
        task.add_done_callback(
            lambda t: assistant._tasks.remove(t) if t in assistant._tasks else None
        )

    ctx.room.register_byte_stream_handler("my-topic", _image_received_handler)
    print("[ENTRYPOINT] Registered byte stream handler for 'my-topic'")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt,
        llm=azure_llm,
        tts=tts,
        turn_handling={
            "turn_detection": MultilingualModel(),
            "interruption": {"mode": "vad"},
        },
        preemptive_generation=True,
    )

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=room_io.RoomInputOptions(
            text_enabled=True, audio_enabled=True
        ),
        # room_options=room_io.RoomOptions(close_on_disconnect=False),
    )

    print("AgentSession started successfully!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, load_threshold=0.9))
