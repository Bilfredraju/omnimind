from memory.store import ConversationStore


class MemoryManager:
    """
    High-level interface for OmniMind memory.
    """

    def __init__(
        self,
        store: ConversationStore | None = None,
    ):
        self.store = store or ConversationStore()

    def remember(
        self,
        user_message: str,
        assistant_message: str,
    ) -> dict:
        return self.store.add_turn(
            user=user_message,
            assistant=assistant_message,
        )

    def recall(
        self,
        limit: int = 5,
    ) -> list[dict]:
        return self.store.get_recent(limit)

    def count(self) -> int:
        return self.store.count()

    def clear(self) -> None:
        self.store.clear()