"""
A prompt template that automates retrieving rows from Feast and making their
content into variables in a prompt.
"""

from functools import reduce
from typing import List, Any, Dict, Callable

from langchain.prompts.dependencyful_prompt import DependencyfulPromptTemplate

import typing
if typing.TYPE_CHECKING:
    from feast.feature_store import FeatureStore
    from feast.entity import Entity
    from feast.feature_view import FeatureView

# def _feast_get_entity_by_name(store: FeatureStore, entityName: str) -> Entity:
def _feast_get_entity_by_name(store, entityName: str):
    return [
        ent
        for ent in store.list_entities()
        if ent.name == entityName
    ][0]


# def _feast_get_entity_join_keys(entity: Entity) -> List[str]:
def _feast_get_entity_join_keys(entity) -> List[str]:
    if hasattr(entity, 'join_keys'):
        # Feast plans to replace `join_key: str` with `join_keys: List[str]`
        return list(entity.join_keys)
    else:
        return [entity.join_key]


# def _feast_get_feature_view_by_name(store: FeatureStore, featureViewName: str) -> FeatureView:
def _feast_get_feature_view_by_name(store, featureViewName: str):
    return [
        fview
        for fview in store.list_feature_views()
        if fview.name == featureViewName
    ][0]


def createFeastPromptTemplate(store, template, input_variables, field_mapper):

    try:
        from feast.feature_store import FeatureStore
        from feast.entity import Entity
        from feast.feature_view import FeatureView
    except (ImportError, ModuleNotFoundError):
        raise ValueError(
            "Could not import feast python package. "
            "Please install it with `pip install \"feast>=0.26\"`."
        )

    # inspection of the store allows to reconstruc this:
    neededFeatureViews = [
        _feast_get_feature_view_by_name(store, fvName)
        for fvName in {
            fview
            for fview, _ in field_mapper.values()
        }
    ]
    neededEntityNames = {
        ent
        for fView in neededFeatureViews
        for ent in fView.entities
    }
    allRequiredJoinKeys = sorted(
        jk
        for entName in neededEntityNames
        for jk in _feast_get_entity_join_keys(
            _feast_get_entity_by_name(store, entName)
        )
    )
    #
    def _getter(deps, **kwargs):
        _store = deps['store']
        #
        feature_vector = _store.get_online_features(
            features=[
                f'{fview}:{fname}'
                for _, (fview, fname) in field_mapper.items()
            ],
            entity_rows=[kwargs]
        ).to_dict()
        #
        retrieved_variables = {
            vname: feature_vector[fname][0]
            for vname, (_, fname) in field_mapper.items()
        }
        return retrieved_variables
    
    feastPromptTemplate = DependencyfulPromptTemplate(
        template=template,
        dependencies={'store': store},
        getter=_getter,
        input_variables=input_variables,
        forceGetterArguments=allRequiredJoinKeys,
    )
    
    return feastPromptTemplate
