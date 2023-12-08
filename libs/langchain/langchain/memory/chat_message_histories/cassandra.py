"""Cassandra-based chat message history, based on cassIO."""
from __future__ import annotations

import json
import typing
import uuid
from typing import List, Optional

if typing.TYPE_CHECKING:
    from cassandra.cluster import Session as CassandraSession

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

DEFAULT_TABLE_NAME = "message_store"
DEFAULT_TTL_SECONDS = None


class CassandraChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Cassandra.

    Args:
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        session: a Cassandra `Session` object (an open DB connection)
        keyspace: name of the keyspace to use.
        table_name: name of the table to use.
        ttl_seconds: time-to-live (seconds) for automatic expiration
            of stored entries. None (default) for no expiration.
    """

    def __init__(
        self,
        session_id: str,
        table_name: str = DEFAULT_TABLE_NAME,
        session: Optional[CassandraSession] = None,
        keyspace: Optional[str] = None,
        ttl_seconds: Optional[int] = DEFAULT_TTL_SECONDS,
        skip_provisioning: bool = False,
    ) -> None:
        try:
            from cassio.table.tables import ClusteredCassandraTable
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        self.session_id = session_id
        self.session = session
        self.keyspace = keyspace
        self.ttl_seconds = ttl_seconds
        self.table_name = table_name
        self.blob_history = ClusteredCassandraTable(
            table=self.table_name,
            session=self.session,
            keyspace=self.keyspace,
            partition_id=self.session_id,
            primary_key_type=["TEXT", "TIMEUUID"],
            ordering_in_partition="DESC",
            ttl_seconds=self.ttl_seconds,
            skip_provisioning=skip_provisioning,
        )

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all session messages from DB"""
        message_blobs = [row["body_blob"] for row in self.blob_history.get_partition()][
            ::-1
        ]
        items = [json.loads(message_blob) for message_blob in message_blobs]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Write a message to the table"""
        this_message_id = uuid.uuid1()
        self.blob_history.put(
            row_id=this_message_id,
            body_blob=json.dumps(message_to_dict(message)),
        )

    def clear(self) -> None:
        """Clear session memory from DB"""
        self.blob_history.delete_partition()
        return None
