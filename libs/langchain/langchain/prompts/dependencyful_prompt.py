"""
Custom prompt template able to deal with:
1. arbitrary dependency injection (e.g. a db connection)
2. arguments passed to the function producing the value for the template variables
"""

import inspect
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Field, validator

from langchain.prompts.base import StringPromptTemplate

GetterType = Callable[[Any], Dict[str, Any]]


class DependencyfulPromptTemplate(StringPromptTemplate):
    dependencies: Dict[str, Any] = Field(exclude=True, default={})
    getter: GetterType
    template: str = Field(exclude=True)
    # needed for overriding when e.g. the getter is programmatically built
    forceGetterArguments: List[str] = Field(exclude=True, default=None)

    # @validator("input_variables")
    # def validate_input_variables(cls, v: dict) -> dict:
    #     # TODO: 'v' is the value of 'input_variables' at initialization
    #     # Raise a ValueError('...') if necessary, otherwise:
    #     return v

    # @validator("getter")
    # def validate_getter(cls, v: GetterType) -> GetterType:
    #     _getterArgs = inspect.getfullargspec(v).args
    #     if _getterArgs:
    #         cls.getterArguments = _getterArgs[1:]
    #         return v
    #     else:
    #         raise ValueError(
    #             "The getter must accept at least one argument (dependency dict)"
    #         )

    def format(self, **kwargs: Any) -> str:
        fullKwargs = {**kwargs, **self.partial_variables}
        # prepare the arguments
        getterArgs = (
            self.getterArguments
            if self.forceGetterArguments is None
            else self.forceGetterArguments
        )
        getterKwargs = {k: v for k, v in fullKwargs.items() if k in getterArgs}
        templateProvidedKwargs = {
            k: v for k, v in fullKwargs.items() if k not in getterArgs
        }
        gottenValues = self.getter(self.dependencies, **getterKwargs)
        #
        fullTemplateKwargs = {
            **templateProvidedKwargs,
            **gottenValues,
        }
        # Generate the prompt to be sent to the language model
        prompt = self.template.format(**fullTemplateKwargs)
        return prompt

    @property
    def _prompt_type(self) -> str:
        return "dependencyful-prompt-template"
