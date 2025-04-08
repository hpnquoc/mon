# How to Structure a Python Package

## Package Structure

```text
package
  |_ __init__.py                        # Interface of the package.
  |_ base.py                            # Abstract base classes for the package.
  |_ core.py                            # Core logical units (i.e., base classes) and core properties.
  |_ io.py                              # I/O operations: load, save, batch i/o, metadata handling.
  |_ processing.py                      # Basic processing: format conversion, normalize, basic operations, etc.
  |_ utils.py                           # Utilities: accessors, validation checks, etc.
```
