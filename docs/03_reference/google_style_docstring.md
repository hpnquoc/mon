### 📘 Python Docstring Style Guide (Personal Preference)

#### ✅ Docstring Format
- **Style**: Google-style docstrings
- **Type hints**: ✅ Only in function signature, **not** in docstring

#### ✅ Formatting Defaults and Values
- Use **double backticks** (``value``) for:
    - Default values (e.g., ``None``, ``True``)
    - Literal strings (e.g., ``"utf-8"``)
    - Boolean or numeric values (e.g., ``0.0``, ``100``)
    - Parameter or keyword names (e.g., ``path``, ``file_format``)
    - Inline code or expressions (e.g., ``len(data) > 0``)

- **Don't** use backticks for:
    - Descriptive phrases
    - Full sentences
    - Generic explanations

#### ✅ Example (Good)

```python
def save_data(path: str, file_format: str = None):
    """
    Save data to file.

    Args:
        path: Output file path.
        file_format: Optional format to save in. Default is ``None``, which infers the format from ``path``.
    """
