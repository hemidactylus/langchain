"""
Custom prompt template able to deal with:
1. arbitrary dependency injection (e.g. a db connection)
2. arguments passed to the function producing the value for the template variables
"""

import inspect
from typing import List, Any, Dict, Callable
from pydantic import BaseModel, validator, Field

from langchain.prompts.base import StringPromptTemplate

class DependencyfulPromptTemplate(StringPromptTemplate, BaseModel):

    dependencies: Dict[str, Any] = Field(exclude=True, default={})
    getter: Callable[Any, Dict[str, Any]]
    template: str = Field(exclude=True)
    # needed for overriding when e.g. the getter is programmatically built
    forceGetterArguments: List[str] = Field(exclude=True, default=None)

    @validator("input_variables")
    def validate_input_variables(cls, v):
        # TODO: 'v' is the value of 'input_variables' at initialization
        # Raise a ValueError('...') if necessary, otherwise:
        return v

    @validator("getter")
    def validate_getter(cls, v):
        _getterArgs = inspect.getfullargspec(v).args
        if _getterArgs:
            cls.getterArguments = _getterArgs[1:]
            return v
        else:
            raise ValueError('The getter must accept at least one argument (dependency dict)')
    
    def format(self, **kwargs) -> str:
        fullKwargs = {**kwargs, **self.partial_variables}
        # prepare the arguments
        getterArgs = self.getterArguments if self.forceGetterArguments is None else self.forceGetterArguments
        getterKwargs = {
            k: v
            for k, v in fullKwargs.items()
            if k in getterArgs
        }
        templateProvidedKwargs = {
            k: v
            for k, v in fullKwargs.items()
            if k not in getterArgs
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
    
    def _prompt_type(self):
        return "dependencyful-prompt-template"
