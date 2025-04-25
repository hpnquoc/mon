# Prompt for Code Refactoring

## Docstrings
- **Format**: Google-style, ≤88 chars, generate docstring if missing. Put __init_'s docstring at class level.
- **Elements**: Use double ticks to represent code elements such as variable names, function names, types, or literal values, but not for array/tuple shapes (e.g., [H, W, C], (1, 2)).
- **Defaults**: "Default is ..." (e.g., "Default is ``False``").
- **Style**: Active voice, concise, no verbose intros (e.g., avoid "This function...").
- **Raises**: Include if `raise` present, with condition.
- **Alignment**: no need to align in Args, Attributes, Returns, Raises.

## Code
- **Brackets**: Use `[ ]` in `raise`, `print`, `log` (e.g., `raise ValueError(f"[path] must be valid, got {path}").`).
- **Style**: Active voice, short statements.
- **Type Hints**: Add return hints (e.g., `-> int`), don't add `-> None` unless explicit `return None`.
- **Assertions**: Replace `assert` with `raise`, use ", but got" (e.g., `raise TypeError(f"[x] must be a dict, got [{type(x).__name__}]")`).
- **Alignment**: Align by `=`, `:`, `(`, `[`, `{` in multi-arg constructs (e.g., `key = value`).

## Formatting & Structure
- **Indentation**: Minimal (4 spaces, don't use tab).
- **Grouping**: Group related properties/methods.

## Optimization & Readability
- **Efficiency**: Optimize code (e.g., use comprehensions, reduce redundancy).
- **Readability**: Clear names, logical flow, avoid complex one-liners.
