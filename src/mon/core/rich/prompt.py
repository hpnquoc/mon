#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extend ``rich.prompt.Prompt``."""

import rich
from rich.columns import Columns
from rich.prompt import *

from mon.core import type_extensions


# ----- Prompt Class -----
class SelectionOrInputPrompt(Prompt):
    """Extend ``rich.prompt.Prompt`` to allow for either selecting an index or directly
    entering value.
    """
    
    response_type: type = str
    
    def print_choices(self):
        """Print columns of choices to the console."""
        choices_ = []
        for i, choice in enumerate(self.choices):
            choices_.append(f"{f'{i}.':>6} {choice}")
        columns = Columns(choices_, equal=True, column_first=True)
        rich.print(columns)
    
    def render_default(self, default: DefaultType) -> Text:
        """Turn the supplied default in to a Text instance.

        Args:
            default: Default value.

        Returns:
            Text containing rendering of default value.
        """
        return Text(f"[{default}]", "prompt.default")
    
    def make_prompt(self, default: DefaultType) -> Text:
        """Make prompt text.

        Args:
            default: Default value.

        Returns:
            Text to display in prompt.
        """
        if self.show_choices and self.choices and len(self.choices) > 0:
            rich.print(self.prompt)
            self.print_choices()
            prompt = Text.from_markup("", style="prompt")
        else:
            prompt = self.prompt.copy()
        prompt.end = ""
        
        if (
            default != ...
            and self.show_default
            and isinstance(default, (str, self.response_type))
        ):
            prompt.append(" ")
            _default = self.render_default(default)
            prompt.append(_default)

        prompt.append(self.prompt_suffix)

        return prompt
    
    def check_choice(self, value: str) -> bool:
        """Check value is in the list of valid choices.

        Args:
            value: Value entered by user.

        Returns:
            ``True`` if choice was valid, otherwise ``False``.
        """
        assert self.choices is not None
        if self.case_sensitive:
            return value in self.choices
        return value.lower() in [choice.lower() for choice in self.choices]
    
    def process_response(self, value: str) -> PromptType:
        """Process response from user, convert to prompt type.

        Args:
            value: String typed by user.

        Raises:
            If ``value`` is invalid.

        Returns:
            The value to be returned from ask method.
        """
        value = value.strip() if isinstance(value, str) else value
        
        if self.choices is not None:
            if len(self.choices) == 0:
                return value
            if len(self.choices) > 0 and value == "":
                raise InvalidResponse(self.illegal_choice_message)
            
            # Convert index (if any) to choice
            value = type_extensions.to_list(value, sep=[",", ";"])
            value = [self.choices[int(d)] if type_extensions.is_int(d) else d for d in value]
            
            for i, v in enumerate(value):
                if not self.check_choice(v):
                    raise InvalidResponse(self.illegal_choice_message)
                if not self.case_sensitive:
                    # return the original choice, not the lower case version
                    value[i] = self.choices[[choice.lower() for choice in self.choices].index(v.lower())]
            
            # value = value[0] if len(value) == 1 else value
            
        return value
    
    def __call__(self, *, default: Any = ..., stream: Optional[TextIO] = None) -> Any:
        """Run the prompt loop.

        Args:
            default (Any, optional): Optional default value.

        Returns:
            PromptType: Processed value.
        """
        while True:
            self.pre_prompt()
            prompt = self.make_prompt(default)
            value  = self.get_input(self.console, prompt, self.password, stream=stream)
            if value == "" and default != ...:
                # return default
                value = default
            try:
                return_value = self.process_response(value)
            except InvalidResponse as error:
                self.on_validate_error(value, error)
                continue
            else:
                return return_value
