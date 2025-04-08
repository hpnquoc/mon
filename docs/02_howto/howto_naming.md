# How to Name Things

## Standard Conventions

| Type                          | Format       |
|-------------------------------|--------------|
| `module_name`                 | noun or verb |
| `package_name`                | noun or verb |
| `local_var_name`              | noun         |
| `global_var_name`             | noun         |
| `instance_var_name`           | noun         |
| `method_name`                 | noun or verb |
| `function_name`               | noun or verb |
| `function_parameter_name`     | noun         |
| `ClassName`                   | noun         |
| `ExceptionName`               | noun         |
| `GLOBAL_CONSTANT_NAME`        | noun         |
| `query_proper_noun_for_thing` | noun         |
| `send_acronym_via_https`      | noun         |

## Singular vs Plural

- Use `singular` nouns for domain or concept. Ex: `vision`, `image`, `classify`, etc.
- Use `plural` nouns for collections of things. Ex: `types`, `serializers`, `datasets`, etc.
- When in doubt, use `singular` nouns.

## Function & Method Names

### **[C]reate:**
- Use `create` when creating a resource. Ex: `create_dir()`.
- Use `write` when preserving data to an external source.

### **[R]ead/Retrieve/Access:**

| Prefix | Action                   | Processing  | Output       | Source                  | Examples                          |
|--------|--------------------------|-------------|--------------|-------------------------|-----------------------------------|
| `list` | Enumerate multiple items | Minimal     | List         | Any (files, memory)     | `list_datasets()`, `list_files()` |
| `read` | Fetch raw data           | Minimal     | Raw          | External (file, stream) |                                   |
| `get`  | Retrieve specific value  | May compute | Single/small | Object/memory           |                                   |
| `load` | Fetch + process data     | Significant | Structured   | External (file, stream) |                                   |
  
- Use `get`:
  - Use `get` when retrieving a **stored value or accessing a property already in memory**, often implying a simple lookup or minimal computation.
  - Omit `get`. Directly names the property (e.g., “area”), implying the function computes or returns it without emphasizing the action of retrieval. Ex: `bbox_area()`.
- Use `read` vs `load`:
  - Use `read` when fetching raw data from a source (e.g., file, stream, network) into memory with minimal processing.
  - Use `load` when bringing data into memory and preparing it for use, often with some processing or conversion to a specific format or structure.  

### **[U]pdate:**
- Use `update` when one or more of the components is updated as a result, and something new could also be added.
- Use `add` to add something into a group of the things.
- Use `append` similar as `add`. It could be used when it doesn't modify the original group of things, but produce the new group.
- Use `disable` to configure a resource an unavailable or inactive state.
- Use `split` when separating parts of a resource.
- Use `merge` when creating a single resource from multiple resources.
- Use `join` similar as `merge` but for data and values.

### **[D]elete:**
- Use `remove` when a given thing is removed from a group of the things.
- Use `delete` to eliminate the object or group of things.

### **Convert:**
- Use `to` when converting a variable from arbitrary types to the desired type. Ex: `to_list()`.
- Use `x_to_y` when converting a variable from type `a` to type `b`. Ex: `str_to_int()`.
- Use `X.from()` when creating an instance of class `X` from a value. Ex: `List.from_string()`.
- Use `parse` when transforming raw input into a structured representation.

### **Validity Check:**
- Use `is` when defining state of a resource. Ex: `is_available()`.
- Use `has` to define whether a resource contains a certain data. Ex: `has_name()`.
- Use `can` to define a certain ability of a resource.
- Use `should` to define a certain obligation of a resource.

### **Using noun for function name:**
- function is always expected to perform an action. **If it barely returns a value, it should be a property**.
- You have a hint that the function should be transformed into a property
  when:
  - The function barely contains a `return ...` statement,
  - The function's name, which comes naturally into your mind is `get_something`, as in `product.get_price()` --> `product.price()`.
