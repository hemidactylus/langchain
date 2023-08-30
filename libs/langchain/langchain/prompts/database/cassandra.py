"""
A prompt template that automates retrieving rows from multiple tables in
Cassandra and making their content into variables in a prompt.
"""
from __future__ import annotations

import typing
from typing import Any, Callable, Dict, Tuple, Union

if typing.TYPE_CHECKING:
    from cassandra.cluster import Session

from langchain.prompts.database.convertor_prompt_template import ConvertorPromptTemplate
from langchain.pydantic_v1 import root_validator

RowToValueType = Union[str, Callable[[Any], Any]]
FieldMapperType = Dict[str, Union[Tuple[str, RowToValueType], Tuple[str, RowToValueType, bool], Tuple[str, RowToValueType, bool, Any]]]

DEFAULT_ADMIT_NULLS = True


class CassandraReaderPromptTemplate(ConvertorPromptTemplate):
    session: Any  # Session

    keyspace: str

    field_mapper: FieldMapperType

    admit_nulls: bool = DEFAULT_ADMIT_NULLS

    @root_validator(pre=True)
    def check_and_provide_convertor(cls, values: Dict) -> Dict:
        convertor_info = cls._prepare_reader_info(
            values["session"],
            values["keyspace"],
            values["field_mapper"],
            values.get("admit_nulls", DEFAULT_ADMIT_NULLS),
        )
        for k, v in convertor_info.items():
            values[k] = v
        return values

    @staticmethod
    def _prepare_reader_info(
        session: Session,
        keyspace: str,
        field_mapper: FieldMapperType,
        admit_nulls: bool,
    ):
        try:
            from cassio.db_extractor import CassandraExtractor
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        #
        _convertor = CassandraExtractor(
            session=session,
            keyspace=keyspace,
            field_mapper=field_mapper,
            admit_nulls=admit_nulls,
        )

        return {
            "convertor": _convertor.dictionary_based_call,
            "convertor_output_variables": _convertor.output_parameters,
            "convertor_input_variables": _convertor.input_parameters,
        }

    @property
    def _prompt_type(self) -> str:
        return "cassandra-reader-prompt-template"
