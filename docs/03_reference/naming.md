# Naming

## Standard Conventions

- `module_name`
- `package_name`
- `local_var_name`
- `global_var_name`
- `instance_var_name`
- `method_name`
- `function_name`
- `function_parameter_name`
- `ClassName`
- `ExceptionName`
- `GLOBAL_CONSTANT_NAME`
- `query_proper_noun_for_thing`
- `send_acronym_via_https`

## Singular vs Plural

- Use `singular` nouns for domain or concept. Ex: `vision`, `image`, `classify`, etc.
- Use `plural` nouns for collections of things. Ex: `types`, `serializers`, `datasets`, etc.
- When in doubt, use `singular` nouns.

## Function & Method Names

- **Creation:**
    - Use `create` when creating a resource. Ex: `create_dir()`.
    - Use `X.from()` when creating an instance of class `X` from a value. Ex: `List.from_string()`.
    - `write` when saving to disk. Use together with `read`. 

- **Accessing:**
    - `get` when retrieving a **stored value or accessing a property**, often implying a simple lookup or minimal computation.
    - Omitting `get`. Directly names the property (e.g., “area”), implying the function computes or returns it without emphasizing the action of retrieval. Ex: `bbox_area()`.
    - `read` when acquiring data from disk. Use together with `write`.

- **Updating:**
	- Use `change` when a whole thing, such as image, is replaced by something else.
	- Use `update` when one or more of the components is updated as a result, and something new could also be added.
	- Use `add` to add something into a group of the things.
	- Use `append` similar as `add`. It could be used when it doesn't modify the original group of things, but produce the new group.
	- Use `write` when preserving data to an external source. Use together with `read`.
	- Use `disable` to configure a resource an unavailable or inactive state.
	- Use `split` when separating parts of a resource.
	- Use `merge` when creating a single resource from multiple resources.
	- Use `join` similar as `merge` but for data and values.

- **Deletion:**
	- Use `remove` when a given thing is removed from a group of the things.
	- Use `delete` to eliminate the object or group of things.
