from __future__ import annotations

import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain.prompts.database.convertor_prompt_template import ConvertorPromptTemplate
from langchain.pydantic_v1 import root_validator


# if typing.TYPE_CHECKING:
from cassandra.cluster import Session

FieldMapperType = Dict[str, Tuple[str, Union[str, Callable[[Any], Any]]]]

DEFAULT_ADMIT_NULLS = True


class CassandraReaderPromptTemplate(ConvertorPromptTemplate):

    session: Session

    keyspace: str

    field_mapper: FieldMapperType

    admit_nulls: bool = DEFAULT_ADMIT_NULLS

    @root_validator(pre=True)
    def check_and_provide_convertor(cls, values: Dict) -> Dict:
        print("VALIDATING IN CRPT")
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
    def _prepare_reader_info(session: Session, keyspace: str, field_mapper: FieldMapperType, admit_nulls: bool):
        try:
            from cassio.db_extractor import CassandraExtractor
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        #
        _conv = CassandraExtractor(
            session=session,
            keyspace=keyspace,
            field_mapper=field_mapper,
            literal_nones=admit_nulls,
        )

        return {
            "convertor": lambda kws: _conv(**kws),
            "convertor_output_variables": list(field_mapper.keys()),  # TODO: infer these
            "convertor_input_variables": _conv.requiredParameters,
        }

    @property
    def _prompt_type(self) -> str:
        return "cassandra-reader-prompt-template"
