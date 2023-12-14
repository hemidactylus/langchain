import os
import time
from typing import Optional

from cassandra.cluster import Cluster
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_message_histories.cassandra import (
    CassandraChatMessageHistory,
)


def _chat_message_history(
    session_id: str = "test-session",
    drop: bool = True,
    ttl_seconds: Optional[int] = None,
) -> CassandraChatMessageHistory:
    keyspace = "cmh_test_keyspace"
    table_name = "cmh_test_table"
    # get db connection
    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = os.environ["CASSANDRA_CONTACT_POINTS"].split(",")
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
    if ttl_seconds is None:
        return CassandraChatMessageHistory(
            session_id=session_id,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
        )
    else:
        return CassandraChatMessageHistory(
            session_id=session_id,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
        )


def test_message_history_base() -> None:
    """Test the memory with a message store."""
    message_history = _chat_message_history()

    assert message_history.messages == []

    # add some messages
    message_history.add_ai_message("This is me, the AI")
    message_history.add_user_message("This is me, the human")

    messages = message_history.messages
    expected = [
        AIMessage(content="This is me, the AI"),
        HumanMessage(content="This is me, the human"),
    ]
    assert messages == expected

    # clear the store
    message_history.clear()

    assert message_history.messages == []


def test_message_history_separate_ids() -> None:
    """Test that separate session IDs do not share entries."""
    message_history1 = _chat_message_history(session_id="test-session1")
    message_history2 = _chat_message_history(session_id="test-session2")

    message_history1.add_ai_message("Just saying.")

    assert message_history2.messages == []

    message_history1.clear()
    message_history2.clear()


def test_message_history_with_ttl() -> None:
    """Test time-to-live feature of the memory."""
    message_history = _chat_message_history(ttl_seconds=5)

    assert message_history.messages == []
    message_history.add_ai_message("Nothing special here.")
    time.sleep(2)
    assert message_history.messages != []
    time.sleep(5)
    assert message_history.messages == []
