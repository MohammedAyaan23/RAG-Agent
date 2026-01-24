import logging

logger = logging.getLogger(__name__)

class ConversationBuffer:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.buffer = []

    def load_turns(self, turns: list[dict]):
        self.buffer = turns[-self.max_turns :]
        logger.info(f"Loaded {len(self.buffer)} turns into buffer")

    def add_turn(self, user: str, assistant: str):
        self.buffer.append({"user": user, "assistant": assistant})
        logger.info("Added new turn to buffer")
        if len(self.buffer) > self.max_turns:
            self.buffer.pop(0)

    def as_messages(self) -> list[dict]:
        messages = []
        for turn in self.buffer:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages
