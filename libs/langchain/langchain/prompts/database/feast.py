"""
A prompt template that automates retrieving rows from Feast and making their
content into variables in a prompt.
"""
from __future__ import annotations

import typing
from typing import Any, Dict, List, Tuple

from langchain.prompts.dependencyful_prompt import DependencyfulPromptTemplate

if typing.TYPE_CHECKING:
    from feast.entity import Entity
    from feast.feature_store import FeatureStore
    from feast.feature_view import FeatureView


FieldMapperType = Dict[str, Tuple[str, str]]


def _feast_get_entity_by_name(store: FeatureStore, entityName: str) -> Entity:
    return [ent for ent in store.list_entities() if ent.name == entityName][0]


def _feast_get_entity_join_keys(entity: Entity) -> List[str]:
    if hasattr(entity, "join_keys"):
        # Feast plans to replace `join_key: str` with `join_keys: List[str]`
        return list(entity.join_keys)
    else:
        return [entity.join_key]


def _feast_get_feature_view_by_name(
    store: FeatureStore, featureViewName: str
) -> FeatureView:
    return [
        fview for fview in store.list_feature_views() if fview.name == featureViewName
    ][0]


def createFeastPromptTemplate(
    store: FeatureStore,
    template: str,
    input_variables: List[str],
    field_mapper: FieldMapperType,
) -> DependencyfulPromptTemplate:
    try:
        pass
    except (ImportError, ModuleNotFoundError):
        raise ValueError(
            "Could not import feast python package. "
            'Please install it with `pip install "feast>=0.26"`.'
        )

    # inspection of the store allows to reconstruc this:
    neededFeatureViews = [
        _feast_get_feature_view_by_name(store, fvName)
        for fvName in {fview for fview, _ in field_mapper.values()}
    ]
    neededEntityNames = {ent for fView in neededFeatureViews for ent in fView.entities}
    allRequiredJoinKeys = sorted(
        jk
        for entName in neededEntityNames
        for jk in _feast_get_entity_join_keys(_feast_get_entity_by_name(store, entName))
    )

    #
    def _getter(deps: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        _store = deps["store"]
        #
        feature_vector = _store.get_online_features(
            features=[f"{fview}:{fname}" for _, (fview, fname) in field_mapper.items()],
            entity_rows=[kwargs],
        ).to_dict()
        #
        retrieved_variables = {
            vname: feature_vector[fname][0]
            for vname, (_, fname) in field_mapper.items()
        }
        return retrieved_variables

    feastPromptTemplate = DependencyfulPromptTemplate(
        template=template,
        dependencies={"store": store},
        getter=_getter,
        input_variables=input_variables,
        forceGetterArguments=allRequiredJoinKeys,
    )

    return feastPromptTemplate
