import json
import os

from cassandra.cluster import Cluster

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.cassandra import (
    CassandraChatMessageHistory,
)
from langchain.schema import _message_to_dict


def _chat_message_history(
    session_id: str = "test-session",
    drop: bool = True,
) -> CassandraChatMessageHistory:
    keyspace = "cmh_test_keyspace"
    table_name = "cmh_test_table"
    # get db connection
    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = os.environ["CONTACT_POINTS"].split(",")
        cluster = Cluster(contact_points)
    else:
        cluster = Cluster()
    #
    session = cluster.connect()
    # ensure keyspace exists
    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )
    # drop table if required
    if drop:
        session.execute(f"DROP TABLE IF EXISTS {keyspace}.{table_name}")
    #
    return CassandraChatMessageHistory(
        session_id=session_id,
        session=session,
        keyspace=keyspace,
        table_name=table_name,
    )


def test_memory_with_message_store() -> None:
    """Test the memory with a message store."""
    # setup cassandra as a message store
    message_history = _chat_message_history()
    memory = ConversationBufferMemory(
        memory_key="baz", chat_memory=message_history, return_messages=True
    )

    # add some messages
    memory.chat_memory.add_ai_message("This is me, the AI")
    memory.chat_memory.add_user_message("This is me, the human")

    # get the message history from the memory store and turn it into a json
    messages = memory.chat_memory.messages
    messages_json = json.dumps([_message_to_dict(msg) for msg in messages])

    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # remove the record from Cassandra, so the next test run won't pick it up
    memory.chat_memory.clear()

    assert memory.chat_memory.messages == []
