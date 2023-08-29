"""
A "convertor"-based prompt template
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    StringPromptTemplate,
)
from langchain.pydantic_v1 import Extra, root_validator


class ConvertorPromptTemplate(StringPromptTemplate):
    @property
    def lc_serializable(self) -> bool:
        return False

    template: str
    """
    TODO: instead of adding `template` and `input_variables`,
    probably can inherit form PromptTemplate (which however
    adds a few methods: check if these withstand the convertor stuff).
    """

    validate_template: bool = True

    input_variables: List[str]

    convertor: Callable[[Dict[str, Any]], Dict[str, Any]]

    convertor_input_variables: List[str]

    convertor_output_variables: List[str]

    template_format: str = "f-string"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=False)
    def check_convertor_and_template(cls, values: Dict) -> Dict:
        # this is e.g. to ensure partialing knows what to do for ChatPromptTemplate
        values["input_variables"] = list(
            set(values["input_variables"]) | set(values["convertor_input_variables"])
        )
        return values

    def format(self, **kwargs: Any) -> str:
        kwargs = self._merge_partial_and_user_variables(**kwargs)

        convertor_kwargs = {
            k: v for k, v in kwargs.items() if k in self.convertor_input_variables
        }
        prompt_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.convertor_input_variables
        }

        _converted = self.convertor(convertor_kwargs)
        # restrict to those which are featured in the prompt
        converted_kwargs = {k: _converted[k] for k in self.convertor_output_variables}

        full_kwargs = {**prompt_kwargs, **converted_kwargs}

        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](
            self.template, **full_kwargs
        )

    @property
    def _prompt_type(self) -> str:
        return "convertor-prompt-template"
