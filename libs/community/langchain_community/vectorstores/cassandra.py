from __future__ import annotations

import typing
import uuid
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

if typing.TYPE_CHECKING:
    from cassandra.cluster import Session as CassandraSession

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

CVST = TypeVar("CVST", bound="Cassandra")


class Cassandra(VectorStore):
    """Wrapper around Apache Cassandra(R) for vector-store workloads.

    To use it, you need a recent installation of the `cassio` library
    and a Cassandra cluster / Astra DB instance supporting vector capabilities.

    Visit the cassio.org website for extensive quickstarts and code examples.

    Constructor args:

        embedding (langchain_core.embeddings.Embeddings): the embedding
            model to use to turn input texts into embedding vectors.
        table_name (str): name of the database table that'll be created, if not
            present yet, to contain the vector store entries.
        session (cassandra.cluster.Session, default None): database connection.
            If not supplied, falls back to the global connection previously set
            through cassio.init(...). If that is also not set, an error is raised.
        keyspace (str, default None): keyspace for the database vector table.
            If not supplied, falls back to the global value previously set
            through cassio.init(...). If that is also not set, an error is raised.
        ttl_seconds (int, default None): Time-to-live for storing entries in
            seconds. If not supplied, entries will persist indefinitely.
        non_searchable_metadata_fields (List[str], default []):
            Optionally specify a list of field names in the metadata
            that will *not* be indexed for later filtering during searches.
            Useful e.g. for very large pieces of metadata that you'll never
            need to use as part of the query and you only need to retrieve
            back with the rest of the document from a search.
            Important: once you commit to a choice for a given vector store,
            changing this value later might lead to inconsistently-stored data.
        partitioned (bool, default False): whether this is a partitioned
            vector store. If you'll have entries grouped in distinct
            *partitions* (e.g. a set of vectors per each user_id),
            and anticipate running most ANN searches on a single partition,
            this would improve ANN performance greatly.
        partition_id: (str, default None): on partitioned vector stores,
            you can specify a partition_id for the whole instance - as opposed
            to supplying the partition_id parameter to method invocations
            (in which case the latter takes precedence)
        skip_provisioning (bool, default False): do not bother creating
            the database table and indexes, assuming they exist already.
            Use only when you know the vector store has been already created
            on DB.

    Example:
        .. code-block:: python

                from langchain.vectorstores import Cassandra
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                session = ...             # create your Cassandra session object
                vectorstore = Cassandra(
                    embeddings=embeddings,
                    session=session,
                    keyspace="my_keyspace",  # it must exist on DB already
                    table_name="my_store",
                )
    """

    _embedding_dimension: Union[int, None]

    # Additional 'kwargs' to let through to CassIO "*ann_search" calls
    CASSIO_SEARCH_KWARGS = {"partition_id"}

    @staticmethod
    def _filter_ann_search_kwargs(args_dict: Dict[str, Any]) -> Dict[str, Any]:
        discarded_args = set(args_dict.keys()) - Cassandra.CASSIO_SEARCH_KWARGS
        if discarded_args:
            warnings.warn(
                f"Warning: unrecognized arguments are being purged from a call "
                f"to the CassIO vector store class: {','.join(discarded_args)}."
            )
        return {
            k: v for k, v in args_dict.items() if k in Cassandra.CASSIO_SEARCH_KWARGS
        }

    @staticmethod
    def _filter_to_metadata(filter_dict: Optional[Dict[str, str]]) -> Dict[str, Any]:
        if filter_dict is None:
            return {}
        else:
            return filter_dict

    def _get_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            self._embedding_dimension = len(
                self.embedding.embed_query("This is a sample sentence.")
            )
        return self._embedding_dimension

    def __init__(
        self,
        embedding: Embeddings,
        table_name: str,
        session: Optional[CassandraSession] = None,
        keyspace: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        non_searchable_metadata_fields: List[str] = [],
        partitioned: bool = False,
        partition_id: Optional[str] = None,
        skip_provisioning: bool = False,
    ) -> None:
        try:
            from cassio.table import (
                ClusteredMetadataVectorCassandraTable,
                MetadataVectorCassandraTable,
            )
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        """Create a vector table."""
        self.embedding = embedding
        self.session = session
        self.keyspace = keyspace
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        #
        self._embedding_dimension = None
        #
        if not partitioned:
            if partition_id is not None:
                raise ValueError(
                    "`partition_id` cannot be supplied on a non-partitioned store."
                )
            self.vector_table = MetadataVectorCassandraTable(
                session=session,
                keyspace=keyspace,
                table=table_name,
                vector_dimension=self._get_embedding_dimension(),
                primary_key_type="TEXT",
                metadata_indexing=(
                    "default_to_searchable",
                    non_searchable_metadata_fields,
                ),
                skip_provisioning=skip_provisioning,
            )
        else:
            if partition_id is None:
                raise ValueError(
                    "`partition_id` must be supplied for a partitioned store."
                )
            self.vector_table = ClusteredMetadataVectorCassandraTable(
                session=session,
                keyspace=keyspace,
                table=table_name,
                vector_dimension=self._get_embedding_dimension(),
                primary_key_type=["TEXT", "TEXT"],
                metadata_indexing=(
                    "default_to_searchable",
                    non_searchable_metadata_fields,
                ),
                partition_id=partition_id,
                skip_provisioning=skip_provisioning,
            )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @staticmethod
    def _dont_flip_the_cos_score(similarity0to1: float) -> float:
        # the identity
        return similarity0to1

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The underlying *CassandraTable already returns a "score proper",
        i.e. one in [0, 1] where higher means more *similar*,
        so here the final score transformation is not reversing the interval:
        """
        return self._dont_flip_the_cos_score

    def delete_collection(self) -> None:
        """
        Just an alias for `clear`
        (to better align with other VectorStore implementations).
        """
        self.clear()

    def clear(self) -> None:
        """Empty the collection."""
        self.vector_table.clear()

    def delete_by_document_id(self, document_id: str) -> None:
        return self.vector_table.delete(row_id=document_id)

    def delete(
        self, ids: Optional[List[str]] = None, batch_size: int = 50, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
            batch_size (int, default=50): size of delete batches

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]

            futures = [
                self.vector_table.delete_async(row_id=document_id)
                for document_id in batch_ids
            ]
            for future in futures:
                future.result()

        return True

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 16,
        ttl_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.
            batch_size (int): Number of concurrent requests to send to the server.
            ttl_seconds (Optional[int], optional): Optional time-to-live
                for the added texts.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        _texts = list(texts)  # lest it be a generator or something
        if ids is None:
            ids = [uuid.uuid4().hex for _ in _texts]
        if metadatas is None:
            metadatas = [{} for _ in _texts]
        #
        ttl_seconds = ttl_seconds or self.ttl_seconds
        #
        embedding_vectors = self.embedding.embed_documents(_texts)
        #
        for i in range(0, len(_texts), batch_size):
            batch_texts = _texts[i : i + batch_size]
            batch_embedding_vectors = embedding_vectors[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            futures = [
                self.vector_table.put_async(
                    row_id=text_id,
                    body_blob=text,
                    vector=embedding_vector,
                    metadata=metadata or {},
                    ttl_seconds=ttl_seconds,
                    **kwargs,
                )
                for text, embedding_vector, text_id, metadata in zip(
                    batch_texts, batch_embedding_vectors, batch_ids, batch_metadatas
                )
            ]
            for future in futures:
                future.result()
        return ids

    # id-returning search facilities
    def similarity_search_with_score_id_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score, id), the most similar to the query vector.
        """
        search_metadata = self._filter_to_metadata(filter)
        #
        hits = self.vector_table.metric_ann_search(
            vector=embedding,
            n=k,
            metric="cos",
            metric_threshold=None,
            metadata=search_metadata,
            **self._filter_ann_search_kwargs(kwargs),
        )
        # We stick to 'cos' distance as it can be normalized on a 0-1 axis
        # (1=most relevant), as required by this class' contract.
        return [
            (
                Document(
                    page_content=hit["body_blob"],
                    metadata=hit["metadata"],
                ),
                0.5 + 0.5 * hit["distance"],
                hit["row_id"],
            )
            for hit in hits
        ]

    def similarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, str]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_id_by_vector(
            embedding=embedding_vector,
            k=k,
            filter=filter,
            **kwargs,
        )

    # id-unaware search facilities
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, doc_id) in self.similarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k,
            filter=filter,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding,
                k,
                filter=filter,
                **kwargs,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding_vector,
            k,
            filter=filter,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        search_metadata = self._filter_to_metadata(filter)

        prefetch_hits = list(
            self.vector_table.metric_ann_search(
                vector=embedding,
                n=fetch_k,
                metric="cos",
                metric_threshold=None,
                metadata=search_metadata,
                **self._filter_ann_search_kwargs(kwargs),
            )
        )
        # let the mmr utility pick the *indices* in the above array
        mmr_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [pf_hit["vector"] for pf_hit in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_hits = [
            pf_hit
            for pf_index, pf_hit in enumerate(prefetch_hits)
            if pf_index in mmr_indices
        ]
        return [
            Document(
                page_content=hit["body_blob"],
                metadata=hit["metadata"],
            )
            for hit in mmr_hits
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Optional.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls: Type[CVST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> CVST:
        """Create a Cassandra vectorstore from raw texts.

        Returns:
            a Cassandra vectorstore.
        """
        cassandra_store = cls(
            embedding=embedding,
            **kwargs,
        )
        cassandra_store.add_texts(
            texts=texts, metadatas=metadatas, batch_size=batch_size
        )
        return cassandra_store

    @classmethod
    def from_documents(
        cls: Type[CVST],
        documents: List[Document],
        embedding: Embeddings,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> CVST:
        """Create a Cassandra vectorstore from a document list.

        Returns:
            a Cassandra vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embedding,
            batch_size=batch_size,
            **kwargs,
        )
