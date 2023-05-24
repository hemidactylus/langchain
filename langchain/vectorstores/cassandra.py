"""Wrapper around Cassandra vector-store capabilities, based on cassIO."""

import uuid
from typing import TypeVar, Type, Iterable, Optional, List, Any, Tuple

from cassandra.cluster import Session

from cassio.vector import VectorDBTable

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.docstore.document import Document


CVST = TypeVar("CVST", bound="Cassandra")

# How many multiples of K are retrieved in a search
CASSANDRA_VECTORSTORE_DEFAULT_OVERFETCH_FACTOR = 3
# Default number of documents ultimately returned in a search
CASSANDRA_VECTORSTORE_DEFAULT_K = 12


class Cassandra(VectorStore):
    """Wrapper around Cassandra embeddings platform.

    There is no notion of a default table name, since each embedding
    function implies its own vector dimension, which is part of the schema.

    Example:
        .. code-block:: python

                from langchain.vectorstores import Cassandra
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                session = ...
                keyspace = 'my_keyspace'
                vectorstore = Cassandra(session, keyspace, 'my_doc_archive', embeddings)
    """

    def _getEmbeddingDimension(self):
        if self._embedding_dimension is None:
            self._embedding_dimension = len(self.embedding.embed_query(
                "This is a sample sentence."
            ))
        return self._embedding_dimension

    def __init__(
        self,
        embedding: Embeddings,
        session: Session,
        keyspace: str,
        table_name: str,
        overfetch_factor = CASSANDRA_VECTORSTORE_DEFAULT_OVERFETCH_FACTOR,
    ) -> None:
        """Create a vector table."""
        self.embedding = embedding
        self.session = session
        self.keyspace = keyspace
        self.table_name = table_name
        self.overfetch_factor = overfetch_factor
        #
        self._embedding_dimension = None
        #
        self.table = VectorDBTable(
            session,
            keyspace,
            table_name,
            self._getEmbeddingDimension(),
            autoID=False, # the `add_texts` contract admits user-provided ids
        )

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.table.clear()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        if metadatas is None:
            metadats = [{} for _ in texts]
        #
        embedding_vectors = self.embedding.embed_documents(texts)
        for text, embedding_vector, text_id, metadata in zip(texts, embedding_vectors, ids, metadatas):
            self.table.put(
                text,
                embedding_vector,
                text_id,
                metadata,
            )
        #
        return ids

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = CASSANDRA_VECTORSTORE_DEFAULT_K,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        No support for `filter` query (on metadata) along with vector search.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query vector.
        """
        hits = self.table.search(
            embedding_vector=embedding,
            topK=k,
            maxRowsToRetrieve=k*self.overfetch_factor,
            metric='cos',
            metricThreshold=None,
        )
        # We stick to 'cos' distance as it can be normalized on a 0-1 axis
        # (1=most relevant), as required by this class' contract.
        return [
            (
                Document(
                    page_content=hit['document'],
                    metadata=hit['metadata'],
                ),
                0.5 + 0.5*hit['distance'],
            )
            for hit in hits
        ]

    def similarity_search(
        self,
        query: str,
        k: int = CASSANDRA_VECTORSTORE_DEFAULT_K,
        **kwargs: Any,
    ) -> List[Document]:
        #
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = CASSANDRA_VECTORSTORE_DEFAULT_K,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding,
                k,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = CASSANDRA_VECTORSTORE_DEFAULT_K,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding_vector,
            k,
        )

    # Even though this is a `_`-method,
    # it is apparently used by VectorSearch parent class
    # in an exposed method (`similarity_search_with_relevance_scores`).
    # So we implement it (hmm).
    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score(
            query,
            k,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls: Type[CVST],
        texts: List[str],
        metadatas: List[dict],
        embedding: Embeddings,
        session: Session,
        keyspace: str,
        table_name: str,
        overfetch_factor = CASSANDRA_VECTORSTORE_DEFAULT_OVERFETCH_FACTOR,
        **kwargs: Any,
    ) -> CVST:
        """Create a Cassandra vectorstore from raw texts.

        No support for specifying text IDs

        Returns:
            a Cassandra vectorstore.
        """
        cassandraStore = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            overfetch_factor=overfetch_factor,
        )
        cassandraStore.add_texts(texts=texts, metadatas=metadatas)
        return cassandraStore

    @classmethod
    def from_documents(
        cls: Type[CVST],
        documents: List[Document],
        embedding: Embeddings,
        session: Session,
        keyspace: str,
        table_name: str,
        overfetch_factor = CASSANDRA_VECTORSTORE_DEFAULT_OVERFETCH_FACTOR,
        **kwargs: Any,
    ) -> CVST:
        """Create a Cassandra vectorstore from a document list.

        No support for specifying text IDs

        Returns:
            a Cassandra vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            overfetch_factor=overfetch_factor,
        )
