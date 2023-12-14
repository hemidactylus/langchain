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

        table_name (str, default "message_store"): name of the database table
            that'll be created, if not present yet, to contain the
            message history entries.
        session (cassandra.cluster.Session, default None): database connection.
            If not supplied, falls back to the global connection previously set
            through cassio.init(...). If that is also not set, an error is raised.
        keyspace (str, default None): keyspace for the database table.
            If not supplied, falls back to the global value previously set
            through cassio.init(...). If that is also not set, an error is raised.
        ttl_seconds (int, default None): Time-to-live for storing entries in
            seconds. If not supplied, entries will persist indefinitely.
        skip_provisioning (bool, default False): do not bother creating
            the database table and indexes, assuming they exist already.
            Use only when you know the chat history table has been already
            created on DB.
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
