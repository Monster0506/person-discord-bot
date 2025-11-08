import asyncio
import logging
import os
import pickle
import tempfile
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import nextcord
from google import genai
from google.genai import types  # adjust to your SDK version
from nextcord.ext import commands


# -------------------- Config --------------------

STATE_PATH = os.path.join("data", "state.pkl")
AUTOSAVE_DEBOUNCE_SEC = 0.75
MAX_HISTORY_TURNS = 80  # per-channel total turns (user + bot)
MODEL_NAME = "gemini-2.0-flash"

# -------------------- Logging --------------------

def setup_logging(level: int = logging.INFO) -> None:
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler(
        os.path.join("logs", "bot.log"),
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(ch)
    root.addHandler(fh)

setup_logging(logging.INFO)
log = logging.getLogger("bot")

# -------------------- Bot / Client --------------------

description = "my bot"
intents = nextcord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", description=description, intents=intents)
client = genai.Client()

# Channel-scoped conversation state
# channel_histories: channel_id(str) -> list[ {role, content, timestamp, author_id?, author_name?} ]
channel_histories: Dict[str, List[Dict[str, Any]]] = {}
# Channel-scoped chat sessions (non-persistent)
channel_sessions: Dict[str, Any] = {}

def utcnow():
    return datetime.now(timezone.utc)

def truncate(msg: str, limit: int = 1900) -> str:
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    return msg[:limit] + "…"

# -------------------- Persistence --------------------

class StateManager:
    def __init__(self, path: str, debounce_sec: float = 1.0):
        self.path = path
        self.debounce_sec = debounce_sec
        self._save_task: Optional[asyncio.Task] = None
        self._save_scheduled_at: Optional[float] = None
        self._lock = asyncio.Lock()

    def _ensure_dir(self):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _atomic_write(self, data: bytes):
        self._ensure_dir()
        dir_name = os.path.dirname(self.path) or "."
        with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, self.path)

    async def save(self):
        async with self._lock:
            try:
                # Only persist serializable structures
                serializable = {
                    "channel_histories": channel_histories,
                }
                data = pickle.dumps(serializable, protocol=pickle.HIGHEST_PROTOCOL)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._atomic_write, data)
                log.info(
                    "State saved | file=%s size=%d channels=%d",
                    self.path,
                    len(data),
                    len(channel_histories),
                )
            except Exception:
                log.exception("Failed to save state to %s", self.path)

    async def schedule_save(self):
        now = asyncio.get_running_loop().time()
        self._save_scheduled_at = now

        async def _runner(expected_when: float):
            try:
                await asyncio.sleep(self.debounce_sec)
                if self._save_scheduled_at != expected_when:
                    return
                await self.save()
            finally:
                if self._save_scheduled_at == expected_when:
                    self._save_task = None

        if self._save_task is None:
            self._save_task = asyncio.create_task(_runner(now))

    def load(self):
        if not os.path.exists(self.path):
            log.warning("State file not found at %s; starting fresh.", self.path)
            return
        try:
            with open(self.path, "rb") as f:
                obj = pickle.load(f)
            loaded = obj.get("channel_histories", {})
            if not isinstance(loaded, dict):
                raise ValueError("Invalid state format for channel_histories")
            channel_histories.clear()
            # Clamp by MAX_HISTORY_TURNS per channel
            for ch_id, hist in loaded.items():
                if isinstance(hist, list):
                    channel_histories[ch_id] = hist[-MAX_HISTORY_TURNS:]
            log.info(
                "Loaded state | file=%s channels=%d",
                self.path,
                len(channel_histories),
            )
        except Exception:
            log.exception("Failed to load state from %s; starting fresh.", self.path)

state = StateManager(STATE_PATH, AUTOSAVE_DEBOUNCE_SEC)

# -------------------- Prompts --------------------

def load_prompts():
    try:
        with open("system.txt", "r", encoding="utf-8") as f1, open(
            "all_stories.txt", "r", encoding="utf-8"
        ) as f2:
            content = f"{f1.read()}\n\n{f2.read()}"
            log.info("Loaded system and stories prompts.")
            return content
    except FileNotFoundError as e:
        log.exception("Required prompt file missing: %s", e)
        raise
    except Exception:
        log.exception("Failed to load prompts.")
        raise

base_system_instruction = load_prompts()

# -------------------- Helpers --------------------

def is_direct_mention_first(message: nextcord.Message) -> bool:
    if bot.user is None:
        return False
    return message.content.strip().startswith(f"<@{bot.user.id}>")

def get_channel_id(message: nextcord.Message) -> str:
    # DM channels also have IDs; fallback to "DM" if missing
    ch_id = getattr(message.channel, "id", None)
    return str(ch_id) if ch_id is not None else "DM"

def ensure_channel_history(ch_id: str):
    if ch_id not in channel_histories:
        channel_histories[ch_id] = []

def trim_channel_history(ch_id: str):
    hist = channel_histories.get(ch_id, [])
    while len(hist) > MAX_HISTORY_TURNS:
        hist.pop(0)

def append_user_turn(ch_id: str, author: nextcord.Member | nextcord.User, content: str, ts):
    ensure_channel_history(ch_id)
    channel_histories[ch_id].append(
        {
            "role": "user",
            "author_id": str(author.id),
            "author_name": getattr(author, "display_name", None) or author.name,
            "content": content,
            "timestamp": ts,
        }
    )
    trim_channel_history(ch_id)

def append_bot_turn(ch_id: str, content: str):
    ensure_channel_history(ch_id)
    channel_histories[ch_id].append(
        {"role": "bot", "content": content, "timestamp": utcnow()}
    )
    trim_channel_history(ch_id)

async def ensure_channel_session(ch_id: str):
    if ch_id in channel_sessions and channel_sessions[ch_id] is not None:
        return
    try:
        channel_sessions[ch_id] = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=base_system_instruction
            ),
        )
        log.info("Created chat session | channel=%s", ch_id)
    except Exception:
        log.exception("Failed to create chat session | channel=%s", ch_id)
        raise

async def reset_channel(ch_id: str):
    # Clear history and reset session
    channel_histories.pop(ch_id, None)
    if ch_id in channel_sessions:
        try:
            # If the SDK has a close/teardown, call it; otherwise just drop it
            channel_sessions.pop(ch_id, None)
        except Exception:
            channel_sessions.pop(ch_id, None)
    log.info("Channel reset | channel=%s", ch_id)
    await state.save()

# -------------------- Events --------------------

@bot.event
async def on_connect():
    try:
        state.load()
    except Exception:
        pass

@bot.event
async def on_ready():
    log.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "unknown")

@bot.event
async def on_message(message: nextcord.Message):
    if message.author.bot:
        return

    ch_id = get_channel_id(message)
    guild_id = getattr(message.guild, "id", "DM")
    content_len = len(message.content or "")
    log.debug(
        "Msg in | guild=%s channel=%s user=%s len=%d",
        guild_id,
        ch_id,
        message.author.id,
        content_len,
    )

    # Record every message to channel-scoped history
    append_user_turn(ch_id, message.author, message.content, message.created_at)
    await state.schedule_save()

    # Only reply if directly mentioned first
    if not is_direct_mention_first(message):
        await bot.process_commands(message)
        return

    # Ensure session for this channel exists
    try:
        await ensure_channel_session(ch_id)
    except Exception:
        await message.channel.send(
            f"{message.author.mention} Sorry, I couldn't start the chat session."
        )
        await bot.process_commands(message)
        return

    # Address the triggering user but keep shared per-channel memory
    mention = message.author.mention

    # Call the model (offload sync call)
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        response = await loop.run_in_executor(
            None, channel_sessions[ch_id].send_message, message.content
        )
        latency = loop.time() - start
        log.info(
            "GenAI call ok | guild=%s channel=%s latency=%.3fs",
            guild_id,
            ch_id,
            latency,
        )
    except Exception:
        latency = loop.time() - start
        log.exception(
            "GenAI call failed | guild=%s channel=%s latency=%.3fs",
            guild_id,
            ch_id,
            latency,
        )
        await message.channel.send(f"{mention} Error generating a response.")
        await bot.process_commands(message)
        return

    text = (getattr(response, "text", None) or str(response) or "").strip()
    if not text:
        text = "I couldn't generate a response."
    addressed = truncate(f"{mention} {text}", 1900)

    try:
        await message.channel.send(addressed)
    except Exception:
        log.exception("Failed to send message | guild=%s channel=%s", guild_id, ch_id)
        await bot.process_commands(message)
        return

    append_bot_turn(ch_id, addressed)
    await state.schedule_save()

    await bot.process_commands(message)

# -------------------- Commands --------------------

@bot.command(name="new")
async def new(ctx: commands.Context):
    """Start a fresh conversation for this channel."""
    ch_id = str(ctx.channel.id)
    await reset_channel(ch_id)
    await ctx.send("Started a new conversation for this channel.")

@bot.command(name="history")
async def history(ctx: commands.Context, n: int = 12):
    """Show the last n turns from this channel's shared history."""
    ch_id = str(ctx.channel.id)
    ensure_channel_history(ch_id)
    n = max(1, min(n, 30))
    hist = channel_histories[ch_id][-n:]
    parts = []
    for t in hist:
        role = t.get("role", "?")
        if role == "user":
            who = t.get("author_name", "user")
            parts.append(f"user:{who}: {t.get('content','')[:140]}")
        else:
            parts.append(f"bot: {t.get('content','')[:140]}")
    if parts:
        await ctx.send("Recent turns:\n" + "\n".join(parts))
    else:
        await ctx.send("No history yet for this channel.")

@bot.command(name="forget")
async def forget(ctx: commands.Context):
    """Clear this channel's history and session."""
    ch_id = str(ctx.channel.id)
    await reset_channel(ch_id)
    await ctx.send("Cleared this channel's history.")

@bot.command(name="setlog")
@commands.has_permissions(administrator=True)
async def setlog(ctx: commands.Context, level: str = "INFO"):
    """Set log level: DEBUG, INFO, WARNING, ERROR."""
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        await ctx.send("Invalid level. Use DEBUG, INFO, WARNING, ERROR.")
        return
    logging.getLogger().setLevel(lvl)
    await ctx.send(f"Log level set to {level.upper()}")
    log.info("Log level changed via command | level=%s guild=%s", level.upper(), getattr(ctx.guild, 'id', 'DM'))

# -------------------- Startup --------------------

def load_token() -> str:
    try:
        with open("TOKEN.txt", "r", encoding="utf-8") as f:
            token = f.read().strip()
            if not token:
                raise ValueError("Empty token")
            log.info("Token loaded.")
            return token
    except Exception:
        log.exception("Failed to load TOKEN.txt")
        raise

if __name__ == "__main__":
    log.info("Starting bot...")
    try:
        bot.run(load_token())
    except KeyboardInterrupt:
        log.info("Shutdown requested by user.")
    except Exception:
        log.exception("bot.run failed")
    finally:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(state.save())
        except Exception:
            log.exception("Final save failed")
        finally:
            log.info("Bot stopped.")
