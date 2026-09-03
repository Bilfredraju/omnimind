from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ConversationTurn:
    """
    Represents one user/assistant interaction.
    """

    user: str
    assistant: str
    timestamp: str

    @classmethod
    def create(cls, user: str, assistant: str):
        return cls(
            user=user,
            assistant=assistant,
            timestamp=datetime.utcnow().isoformat(),
        )

    def to_dict(self) -> dict:
        return asdict(self)