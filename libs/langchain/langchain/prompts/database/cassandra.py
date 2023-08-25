"""
A prompt template that automates retrieving rows from Cassandra and making their
content into variables in a prompt.
"""
from __future__ import annotations

import typing
from typing import Any, Callable, Dict, List, Tuple, Union

if typing.TYPE_CHECKING:
    from cassandra.cluster import Session

from langchain.prompts.dependencyful_prompt import DependencyfulPromptTemplate

# Since subclassing for thins one is a mess, with pydantic and so many changed parameters,
# we opt for a factory function

FieldMapperType = Dict[str, Tuple[str, Union[str, Callable[[Any], Any]]]]


def createCassandraPromptTemplate(
    session: Session,
    keyspace: str,
    template: str,
    input_variables: List[str],
    field_mapper: FieldMapperType,
    literal_nones: bool = False,
) -> DependencyfulPromptTemplate:
    try:
        from cassio.db_extractor import CassandraExtractor
    except (ImportError, ModuleNotFoundError):
        raise ValueError(
            "Could not import cassio python package. "
            "Please install it with `pip install cassio`."
        )

    # we need a callable that receives a 'dependencies' dict argument and other keyword args = columns in primary keys
    # and returns the values as in the field_mapper provided dict
    dataExtractor = CassandraExtractor(session, keyspace, field_mapper, literal_nones)

    def _dataGetter(deps: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        # we ignore dependencies in this case, knowing it's not required
        # by the extractor contract
        return dataExtractor(**kwargs)

    # let's store the extractor in the dependencies.
    # Should it ever need to be accessed (but no immediate reason to)
    cptDependencies = {"extractor": dataExtractor}

    cassandraPromptTemplate = DependencyfulPromptTemplate(
        template=template,
        dependencies=cptDependencies,
        getter=_dataGetter,
        input_variables=input_variables,
        forceGetterArguments=dataExtractor.requiredParameters,
    )

    return cassandraPromptTemplate
