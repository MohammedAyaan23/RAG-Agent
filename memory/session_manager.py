from pathlib import Path
from datetime import datetime
from typing import Optional
import uuid
import logging
import json

logger = logging.getLogger(__name__)

# Directory where all session files live
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


# -------------------------
# Session ID handling
# -------------------------

def generate_session_id() -> str:
    """Generate a unique session ID."""
    return uuid.uuid4().hex


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """
    Return an existing session_id if provided,
    otherwise generate a new one.
    """

    sid = session_id or generate_session_id()
    if not session_id:
        logger.info(f"Generated new session ID: {sid}")
    else:
        logger.info(f"Using existing session ID: {sid}")
    return sid


def session_file_path(session_id: str) -> Path:
    """Return the file path for a session."""
    return SESSIONS_DIR / f"{session_id}.txt"


# -------------------------
# Writing to session file
# -------------------------

def append_event(
    session_id: str,
    role: str,
    content: str,
):
    """
    Append a single event to the session file.

    role examples:
    - user
    - assistant
    - planner
    - state
    - router
    """
    timestamp = datetime.utcnow().isoformat()
    path = session_file_path(session_id)

    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {role.upper()}:\n")
        f.write(content.strip())
        f.write("\n\n")
    
    logger.info(f"Appended event '{role}' to session {session_id}")


def append_state(session_id: str, state: dict):
    """Convenience wrapper for writing conversation state."""
    append_event(session_id, "state", json.dumps(state))


# -------------------------
# Reading from session file
# -------------------------

def load_last_state(
    session_id: str,
    default: str = "start",
) -> str:
    """
    Load the most recent conversation state from the session file.
    If none exists, return the default state.
    """
    path = session_file_path(session_id)

    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Walk backwards to find the last STATE entry
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().endswith("STATE:"):
            if i + 1 < len(lines):
                return lines[i + 1].strip()

    return default


def load_last_n_turns(session_id: str, n: int = 5) -> list[dict]:
    """
    Load the last N (user, assistant) turns from the session file.
    Returns a list of dicts: { "user": ..., "assistant": ... }
    """
    path = session_file_path(session_id)
    if not path.exists():
        return []

    turns = []
    current_user = None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()

        if line.endswith("USER:"):
            if i + 1 < len(lines):
                current_user = lines[i + 1].strip()

        elif line.endswith("ASSISTANT:") and current_user is not None:
            if i + 1 < len(lines):
                assistant = lines[i + 1].strip()
                turns.append(
                    {
                        "user": current_user,
                        "assistant": assistant,
                    }
                )
                current_user = None

    return turns[-n:]
