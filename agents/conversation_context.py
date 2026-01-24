from dataclasses import dataclass
from typing import Optional

@dataclass
class ConversationContext:
    conversation_id: str
    user_id: Optional[str] = None
    last_planner: Optional[str] = None
    last_intent: Optional[str] = None
    turn_count: int = 0
    unresolved: bool = False    
