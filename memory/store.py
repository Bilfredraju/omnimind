import json
from pathlib import Path

from memory.conversation import ConversationTurn


class ConversationStore:
    """
    Persistent JSON-backed conversation memory.
    """

    def __init__(
        self,
        path: str = "data/memory/conversations.json",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.path.exists():
            self._save([])

    def _load(self) -> list[dict]:
        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if isinstance(data, list):
                return data

        except (json.JSONDecodeError, OSError):
            pass

        return []

    def _save(self, data: list[dict]) -> None:
        with self.path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def add_turn(
        self,
        user: str,
        assistant: str,
    ) -> dict:
        turn = ConversationTurn.create(
            user=user,
            assistant=assistant,
        )

        data = self._load()
        data.append(turn.to_dict())
        self._save(data)

        return turn.to_dict()

    def get_recent(
        self,
        limit: int = 5,
    ) -> list[dict]:
        data = self._load()

        if limit <= 0:
            return []

        return data[-limit:]

    def count(self) -> int:
        return len(self._load())

    def clear(self) -> None:
        self._save([])