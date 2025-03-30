# Data Types for Multiple Tasks: Design Decision

## Problem
Decide where to implement data types (e.g., custom classes, structs) for a project with multiple tasks or domains:
- **Option 1**: Within each task/domain module (e.g., `task1/types.py`, `task2/types.py`).
- **Option 2**: In a shared `data` module (e.g., `data/__init__.py`).

## Options Analyzed

### Option 1: Task-Specific Types
- **Structure**: Each task has its own `types.py` (e.g., `task1/types.py` with `DataTypeA`, `task2/types.py` with `DataTypeB`).
- **Pros**:
    - **High Cohesion**: Data types are closely tied to their task’s logic.
    - **Low Coupling**: No shared dependency between tasks.
    - **Encapsulation**: Task-specific data stays isolated.
- **Cons**:
    - **Duplication**: Shared data types (e.g., a common `Tensor` type) are repeated across tasks.
    - **Inconsistency**: Similar types might diverge in definition or behavior.

### Option 2: Shared `data` Module
- **Structure**: Single `data` module (e.g., `data/__init__.py`) holds all types (e.g., `DataTypeA`, `DataTypeB`).
- **Pros**:
    - **Reusability**: Common types are defined once and shared across tasks.
    - **Consistency**: Uniform type definitions for all tasks.
    - **Interoperability**: Easier to pass data between tasks.
- **Cons**:
    - **Increased Coupling**: Tasks depend on `data`, linking them together.
    - **Lower Cohesion**: `data` mixes types from unrelated tasks.

## Recommendation: Hybrid Approach
- **Structure**:

```text
project/
├── data/
│   ├── init.py  # Shared base types (e.g., BaseData)
│   └── common.py    # Common concrete types (e.g., SharedType)
├── task1/
│   ├── types.py     # Task1-specific types (e.g., Task1Data)
├── task2/
│   ├── types.py     # Task2-specific types (e.g., Task2Data)
...
```

- **Details**:
  - `data/` contains minimal shared types:
      - Base classes (e.g., `BaseData`) for inheritance.
      - Concrete types used across tasks (e.g., `SharedType`).
  - Task-specific types live in each task’s `types.py`.

- **Why**:
  - Balances cohesion (task-specific types stay local) and reusability (shared types avoid duplication).
  - Reduces coupling by keeping `data` lightweight.
  - Supports projects with multiple tasks and some overlapping data needs.

## Example
```python
# data/__init__.py
from dataclasses import dataclass

@dataclass
class BaseData:
  """Base class for all data types."""
  id: int

# data/common.py
from data import BaseData

@dataclass
class SharedType(BaseData):
  """Type shared across tasks."""
  value: float

# task1/types.py
from data import BaseData

@dataclass
class Task1Data(BaseData):
  """Task1-specific type."""
  field1: str

# task2/types.py
from data import BaseData

@dataclass
class Task2Data(BaseData):
  """Task2-specific type."""
  field2: int
