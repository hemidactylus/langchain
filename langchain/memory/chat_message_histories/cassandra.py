import json
from typing import List


from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)


DEFAULT_TABLE_NAME = 'langchain_chat_history'
DEFAULT_TTL_SECONDS = None


class CassandraChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, session_id, session, keyspace,
                 tableName=DEFAULT_TABLE_NAME, ttl_seconds=DEFAULT_TTL_SECONDS):
        try:
            from cassio.history import StoredBlobHistory
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        self.sessionId = session_id
        self.ttlSeconds = ttl_seconds
        self.blobHistory = StoredBlobHistory(session, keyspace, tableName)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all session messages from DB"""
        messageBlobs = self.blobHistory.retrieve(
            self.sessionId,
        )
        items = [json.loads(messageBlob) for messageBlob in messageBlobs]
        messages = messages_from_dict(items)
        return messages

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Write a message to the table"""
        self.blobHistory.store(
            self.sessionId,
            json.dumps(_message_to_dict(message)),
            self.ttlSeconds
        )

    def clear(self) -> None:
        """Clear session memory from DB"""
        self.blobHistory.clearSessionId(self.sessionId)
