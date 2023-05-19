"""
A prompt template that automates retrieving rows from Cassandra and making their
content into variables in a prompt.
"""

from typing import List, Any, Dict, Callable

from langchain.prompts.dependencyful_prompt import DependencyfulPromptTemplate

from cassio.db_extractor import CassandraExtractor

# Since subclassing for thins one is a mess, with pydantic and so many changed parameters,
# we opt for a factory function

def createCassandraPromptTemplate(session, keyspace, template, input_variables, field_mapper, literal_nones=False):

    # we need a callable that receives a 'dependencies' dict argument and other keyword args = columns in primary keys
    # and returns the values as in the field_mapper provided dict
    dataGetter = CassandraExtractor(session, keyspace, field_mapper, literal_nones)
    
    cassandraPromptTemplate = DependencyfulPromptTemplate(
        template=template,
        dependencies={'session': session, 'keyspace': keyspace},
        getter=dataGetter,
        input_variables=input_variables,
        forceGetterArguments=dataGetter.requiredParameters,
    )
    
    return cassandraPromptTemplate
