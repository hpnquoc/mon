#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory Module.

This module implements a factory method design pattern. It defines mechanisms
for registering classes and dynamically build them at run-time.
"""

from __future__ import annotations

__all__ = [
    "Factory",
    "ModelFactory",
]

import copy
import inspect
from typing import Any

import humps


# region Factory

class Factory(dict):
    """The base factory class for building arbitrary objects. It registers
    classes to a registry `dict` and then dynamically builds objects of
    the registered classes later.
    
    Notes:
        We inherit Python built-in `dict`.
    
    Args:
        name: The factory's name.
        
    Example:
        >>> MODEL = Factory("Model")
        >>> @MODEL.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet = MODEL.build(name="ResNet", **resnet_hparams)
    """
    
    def __init__(self, name: str, mapping: dict = None, *args, **kwargs):
        """Initialize the Factory with a name and an optional mapping.

        Args:
            name: The factory's name.
            mapping: An optional dictionary to initialize the factory with.
        """
        if not name:
            raise ValueError("`name` must be given to create a valid factory object.")
        self.name = name
        super().__init__(mapping or {})
    
    def __repr__(self) -> str:
        """Return a string representation of the Factory."""
        return f"{self.__class__.__name__}(name={self.name}, items={self})"
    
    def register(
        self,
        name   : str  = None,
        module : Any  = None,
        replace: bool = False
    ) -> callable:
        """Register a module/class.
        
        Args:
            name: A module/class name. If ``None``, automatically infer from the
                given ``module``.
            module: The registering module.
            replace: If ``True``, overwrite the existing module.
                Default: ``False``.
        
        Returns:
            callable: A decorator to register the module/class.
            
        Example:
            # >>> backbones = Factory("backbone")
            # >>>
            # >>> @backbones.register()
            # >>> class ResNet:
            # >>>     pass
            # >>>
            # >>> @backbones.register(name="mnet")
            # >>> class MobileNet:
            # >>>     pass
            # >>>
            # >>> class ResNet:
            # >>>     pass
            # >>> backbones.register(ResNet)
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"`name` must be a `str`, but got {type(name)}.")
        
        def _register(cls):
            self.register_module(module_cls=cls, module_name=name, replace=replace)
            return cls
        
        return _register(module) if module else _register
    
    def register_module(
        self,
        module_cls : Any,
        module_name: str  = None,
        replace    : bool = False
    ):
        """Register a module/class.

        Args:
            module_cls: The registering module/class.
            module_name: A module/class name. If ``None``, automatically infer
                from the given `module`.
            replace: If ``True``, overwrite the existing module. Default: ``False``.
        """
        if not inspect.isclass(module_cls):
            raise ValueError(f"`module_cls` must be a class interface, but got {type(module_name)}.")
        
        module_name = module_name or humps.kebabize(module_cls.__name__)
        if replace or module_name not in self:
            self[module_name] = module_cls
        
    def build(
        self,
        name   : str  = None,
        config : dict = None,
        to_dict: bool = False,
        **kwargs
    ):
        """Build an instance of the registered class.

        Args:
            name: Class name.
            config: Class arguments.
            to_dict: If True, return a dict of {name: instance}. Default: False.

        Returns:
            An instance of the registered class.
        """
        if not name and (not config or "name" not in config):
            return None
        if config:
            config_ = copy.deepcopy(config)
            name = name or config_.pop("name", None)
            kwargs |= config_
            
        # Loop through all possible naming conventions
        for n in [name,
                  humps.kebabize(name),
                  humps.depascalize(humps.pascalize(name)),
                  humps.pascalize(name)]:
            if n in self:
                name = n
                break
        else:
            raise ValueError(f"`name` must be a valid keyword inside the registry, but got {name}.")
        
        obj = self[name](**kwargs)
        if getattr(obj, "name", None) is None:
            obj.name = humps.depascalize(humps.pascalize(name))
        
        return {f"{name}": obj} if to_dict else obj
    
    def build_instances(
        self,
        configs: list[Any],
        to_dict: bool = False,
        **kwargs
    ):
        """Build multiple instances of different classes with the given
        `args`.

        Args:
            configs: A list of classes' arguments. Each item can be:
                - A name (str).
                - A dictionary of arguments containing the 'name' key.
            to_dict: If True, return a dict of {name: instance}. Default: False.

        Returns:
            A list, or a dictionary of instances.
        """
        if not isinstance(configs, list):
            raise ValueError(f"`configs` must be a `list`, but got {type(configs)}.")
        
        configs_ = copy.deepcopy(configs)
        objs     = {} if to_dict else []
        for config in configs_:
            if isinstance(config, str):
                name = config
            elif isinstance(config, dict):
                name = config.pop("name")
            else:
                raise ValueError(f"Item inside `configs` must be a `str` or `dict`, but got {type(config)}.")
    
            obj = self.build(name=name, to_dict=to_dict, **config)
            if obj:
                if to_dict:
                    objs |= obj
                else:
                    objs.append(obj)
    
        return objs if objs else None


class ModelFactory(Factory):
    """The factory for registering and building models.
    
    Notes:
        We inherit Python built-in `dict`.
    
    Example:
        >>> MODEL = ModelFactory("Model")
        >>> @MODEL.register(arch="resnet", name="resnet")
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODEL.build(name="resnet", **resnet_hparams)
    """
    
    @property
    def archs(self) -> list[str]:
        """List of registered architectures.

        Returns:
            list[str]: A list of architecture names.
        """
        return list(self)
    
    @property
    def models(self) -> list[str]:
        """List of registered models.

        Returns:
            list[str]: A list of model names.
        """
        return [model for models in self.values() if isinstance(models, dict) for model in models]
    
    def register(
        self,
        name   : str  = None,
        arch   : str  = None,
        module : Any  = None,
        replace: bool = False,
    ) -> callable:
        """Register a model.

        Args:
            name: Model's name. If None, infer from the module.
            arch: Architecture's name. If None, infer from the module.
            module: The registering module.
            replace: If True, overwrite the existing module. Default: False.

        Returns:
            callable: A decorator to register the module/class.
        """
        if name is not None and not isinstance(name, str):
            raise TypeError(f"`name` must be a `str`, but got {type(name)}.")
    
        if module:
            self.register_module(module_cls=module, module_name=name, arch_name=arch, replace=replace)
            return module
    
        def _register(cls):
            self.register_module(module_cls=cls, module_name=name, arch_name=arch, replace=replace)
            return cls
    
        return _register
    
    def register_module(
        self,
        module_cls : Any,
        module_name: str  = None,
        arch_name  : str  = None,
        replace    : bool = False
    ):
        """Register a module/class.

        Args:
            module_cls: The registering module/class.
            module_name: Module/class name. If None, infer from the module.
            arch_name: Architecture's name. If None, infer from the module.
            replace: If True, overwrite the existing module. Default: False.
        """
        if not inspect.isclass(module_cls):
            raise ValueError(f"`module_cls` must be a class, but got {type(module_name)}.")

        module_name = module_name or humps.kebabize(module_cls.__name__)
        arch_name   = (arch_name
                       or humps.kebabize(getattr(module_cls, "arch", None))
                       or humps.kebabize(module_cls.__name__))
        if arch_name not in self:
            self[arch_name] = {}
        if replace or module_name not in self[arch_name]:
            self[arch_name][module_name] = module_cls
    
    def build(
        self,
        name   : str  = None,
        arch   : str  = None,
        config : dict = None,
        to_dict: bool = False,
        **kwargs
    ):
        """Build an instance of the registered model's variant corresponding to
        the given name.

        Args:
            name: Model's name.
            arch: Architecture's name.
            config: The class's arguments.
            to_dict: If True, return a dict of {name: instance}. Default: False.
    
        Returns:
            An instance of the registered class.
        """
        if name is None and (config is None or "name" not in config):
            return None
        if config:
            config_ = copy.deepcopy(config)
            name = name or config_.pop("name", None)
            kwargs |= config_
        arch = arch or name
        
        # Loop through all possible naming conventions
        for n in [name,
                  humps.kebabize(name),
                  humps.depascalize(humps.pascalize(name)),
                  humps.pascalize(name)]:
            for a, models_dict in self.items():
                if n in models_dict:
                    name, arch = n, a
                    break
    
        if arch not in self or name not in self[arch]:
            raise ValueError(f"`arch` and `name` must be a valid keyword inside "
                             f"the registry, but got {arch} and {name}.")
    
        obj = self[arch][name](**kwargs)
        if getattr(obj, "name", None) is None:
            obj.name = humps.depascalize(humps.pascalize(name))
    
        return {f"{name}": obj} if to_dict else obj
    
    def build_instances(
        self,
        configs: list[Any],
        to_dict: bool = False,
        **kwargs
    ):
        """Build multiple instances of different classes with the given args.

        Args:
            configs: A list of classes' arguments. Each item can be:
                - A name (str).
                - A dictionary of arguments containing the 'name' key.
            to_dict: If True, return a dict of {name: instance}. Default: False.
    
        Returns:
            A list, or a dictionary of instances.
        """
        if not isinstance(configs, list):
            raise ValueError(f"`configs` must be a `list`, but got {type(configs)}.")
        
        configs_ = copy.deepcopy(configs)
        objs     = {} if to_dict else []
        for config in configs_:
            if isinstance(config, str):
                name, arch = config, None
            elif isinstance(config, dict):
                name, arch = config.pop("name", None), config.pop("arch", None)
            else:
                raise ValueError(f"Item inside `configs` must be a `str` or `dict`, but got {type(config)}.")
    
            obj = self.build(name=name, arch=arch, to_dict=to_dict, **config)
            if obj:
                if to_dict:
                    objs |= obj
                else:
                    objs.append(obj)
    
        return objs if objs else None

# endregion
