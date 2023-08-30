"""
A prompt template that automates retrieving rows from Feast and making their
content into variables in a prompt.
"""
from __future__ import annotations

import typing
from typing import Any, Dict, List, Tuple

if typing.TYPE_CHECKING:
    from feast.entity import Entity
    from feast.feature_store import FeatureStore
    from feast.feature_view import FeatureView

from langchain.prompts.database.convertor_prompt_template import ConvertorPromptTemplate
from langchain.pydantic_v1 import root_validator


FieldMapperType = Dict[str, Tuple[str, str]]


def _feast_get_entity_by_name(store: FeatureStore, entity_name: str) -> Entity:
    return [ent for ent in store.list_entities() if ent.name == entity_name][0]


def _feast_get_entity_join_keys(entity: Entity) -> List[str]:
    if hasattr(entity, "join_keys"):
        # Feast plans to replace `join_key: str` with `join_keys: List[str]`
        return list(entity.join_keys)
    else:
        return [entity.join_key]


def _feast_get_feature_view_by_name(
    store: FeatureStore, feature_view_name: str
) -> FeatureView:
    return [
        fview for fview in store.list_feature_views() if fview.name == feature_view_name
    ][0]


class FeastReaderPromptTemplate(ConvertorPromptTemplate):
    feature_store: Any  # FeatureStore

    field_mapper: FieldMapperType

    @root_validator(pre=True)
    def check_and_provide_convertor(cls, values: Dict) -> Dict:

        convertor_info = cls._prepare_reader_info(
            values["feature_store"],
            values["field_mapper"],
        )
        for k, v in convertor_info.items():
            values[k] = v
        return values

    @staticmethod
    def _prepare_reader_info(
        feature_store: FeatureStore,
        field_mapper: FieldMapperType,
    ):
        try:
            from feast.entity import Entity
            from feast.feature_store import FeatureStore
            from feast.feature_view import FeatureView
        except (ImportError, ModuleNotFoundError):
            raise ValueError(
                "Could not import feast python package. "
                'Please install it with `pip install "feast>=0.26"`.'
            )
        # inspection of the store to build the getter and the var names:
        required_f_views = [
            _feast_get_feature_view_by_name(feature_store, f_view_name)
            for f_view_name in {fwn for fwn, _ in field_mapper.values()}
        ]
        required_entity_names = {ent for f_view in required_f_views for ent in f_view.entities}
        join_keys = sorted({
            join_key
            for entity_name in required_entity_names
            for join_key in _feast_get_entity_join_keys(_feast_get_entity_by_name(feature_store, entity_name))
        })

        def _convertor(args_dict: Dict[str, Any]) -> Dict[str, Any]:
            feature_vector = feature_store.get_online_features(
                features=[f"{fview}:{fname}" for _, (fview, fname) in field_mapper.items()],
                entity_rows=[args_dict],
            ).to_dict()
            #
            retrieved_variables = {
                vname: feature_vector[fname][0]
                for vname, (_, fname) in field_mapper.items()
            }
            return retrieved_variables

        return {
            "convertor": _convertor,
            "convertor_output_variables": list(field_mapper.keys()),
            "convertor_input_variables": join_keys,
        }

    @property
    def _prompt_type(self) -> str:
        return "feast-reader-prompt-template"
