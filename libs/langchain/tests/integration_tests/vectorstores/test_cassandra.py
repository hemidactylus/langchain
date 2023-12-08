"""Test Cassandra functionality."""
import os
import time
from typing import List, Optional, Type

import pytest
from cassandra.cluster import Cluster

from langchain_core.documents import Document
from langchain.vectorstores import Cassandra
from tests.integration_tests.vectorstores.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    ConsistentFakeEmbeddings,
    Embeddings,
)


def _vectorstore_from_texts(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    embedding_class: Type[Embeddings] = ConsistentFakeEmbeddings,
    drop: bool = True,
    table_suffix: Optional[str] = None,
    partitioned: bool = False,
    partition_id: str = "my_part",
    skip_provisioning: bool = False,
) -> Cassandra:
    keyspace = "vector_test_keyspace"
    _p_prefix = "" if not partitioned else "p_"
    if table_suffix:
        table_name = f"{_p_prefix}vec_table_{table_suffix}"
    else:
        table_name = f"{_p_prefix}vec_table"
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
    return Cassandra.from_texts(
        texts=texts,
        embedding=embedding_class(),
        metadatas=metadatas,
        session=session,
        keyspace=keyspace,
        table_name=table_name,
        partitioned=partitioned,
        partition_id=(partition_id if partitioned else None),
        skip_provisioning=skip_provisioning,
    )


@pytest.mark.parametrize("partitioned", [False, True])
def test_cassandra(partitioned: bool) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = _vectorstore_from_texts(
        texts, table_suffix="c", partitioned=partitioned
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.parametrize("partitioned", [False, True])
def test_cassandra_with_score(partitioned: bool) -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(
        texts, metadatas=metadatas, table_suffix="wc", partitioned=partitioned
    )
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": "0.0"}),
        Document(page_content="bar", metadata={"page": "1.0"}),
        Document(page_content="baz", metadata={"page": "2.0"}),
    ]
    assert scores[0] > scores[1] > scores[2]


@pytest.mark.parametrize("partitioned", [False, True])
def test_cassandra_max_marginal_relevance_search(partitioned: bool) -> None:
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==3 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order).
    """
    texts = ["-0.124", "+0.127", "+0.25", "+1.0"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(
        texts,
        metadatas=metadatas,
        embedding_class=AngularTwoDimensionalEmbeddings,
        table_suffix="mmr",
        partitioned=partitioned,
    )
    # time.sleep(0.5)
    output = docsearch.max_marginal_relevance_search("0.0", k=2, fetch_k=3)
    output_set = {
        (mmr_doc.page_content, mmr_doc.metadata["page"]) for mmr_doc in output
    }
    assert output_set == {
        ("+0.25", "2.0"),
        ("-0.124", "0.0"),
    }


@pytest.mark.parametrize("partitioned", [False, True])
def test_cassandra_add_extra(partitioned: bool) -> None:
    """Test end to end construction with further insertions."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(
        texts, metadatas=metadatas, table_suffix="x", partitioned=partitioned
    )

    texts2 = ["foo2", "bar2", "baz2"]
    metadatas2 = [{"page": i + 3} for i in range(len(texts))]
    docsearch.add_texts(texts2, metadatas2)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


@pytest.mark.parametrize("partitioned", [False, True])
def test_cassandra_no_drop(partitioned: bool) -> None:
    """Test end to end construction and re-opening the same index."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(
        texts, metadatas=metadatas, table_suffix="nd", partitioned=partitioned
    )
    del docsearch

    texts2 = ["foo2", "bar2", "baz2"]
    docsearch = _vectorstore_from_texts(
        texts2,
        metadatas=metadatas,
        drop=False,
        table_suffix="nd",
        partitioned=partitioned,
    )

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


@pytest.mark.parametrize("partitioned", [False, True])
def test_cassandra_delete(partitioned: bool) -> None:
    """Test delete methods from vector store."""
    texts = ["foo", "bar", "baz", "gni"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(
        [], metadatas=metadatas, table_suffix="dl", partitioned=partitioned
    )

    ids = docsearch.add_texts(texts, metadatas)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 4

    docsearch.delete_by_document_id(ids[0])
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 3

    docsearch.delete(ids[1:3])
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 1

    docsearch.delete(["not-existing"])
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 1

    docsearch.clear()
    time.sleep(1.8)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 0


def test_partitioning() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = _vectorstore_from_texts(
        texts[:1], table_suffix="pt", drop=True, partitioned=True, partition_id="alpha"
    )
    _ = _vectorstore_from_texts(
        texts[:1],
        table_suffix="pt",
        drop=False,
        partitioned=True,
        partition_id="omega",
        skip_provisioning=True,
    )
    output = docsearch.similarity_search("foo", k=2)
    # if partition Id does not get applied to this search, we would get _two_ rows back:
    assert output == [Document(page_content="foo")]


if __name__ == "__main__":
    test_cassandra(partitioned=False)
    test_cassandra_with_score(partitioned=False)
    test_cassandra_max_marginal_relevance_search(partitioned=False)
    test_cassandra_add_extra(partitioned=False)
    test_cassandra_no_drop(partitioned=False)
    test_cassandra_delete(partitioned=False)
    test_cassandra(partitioned=True)
    test_cassandra_with_score(partitioned=True)
    test_cassandra_max_marginal_relevance_search(partitioned=True)
    test_cassandra_add_extra(partitioned=True)
    test_cassandra_no_drop(partitioned=True)
    test_cassandra_delete(partitioned=True)
    test_partitioning()
