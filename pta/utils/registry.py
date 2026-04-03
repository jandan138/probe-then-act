"""Registry pattern for methods, environments, and models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A generic registry that maps string keys to callables.

    Useful for registering environment factories, model constructors,
    and evaluation method builders so they can be instantiated from
    configuration files.

    Parameters
    ----------
    name : str
        Human-readable name for the registry (used in error messages).

    Examples
    --------
    >>> env_registry = Registry("environments")
    >>> @env_registry.register("pick_place")
    ... def make_pick_place(**kwargs):
    ...     ...
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: Dict[str, Callable[..., Any]] = {}
        raise NotImplementedError

    def register(
        self,
        key: str,
        obj: Optional[Callable[..., Any]] = None,
    ) -> Callable[..., Any]:
        """Register an object under *key*, usable as a decorator.

        Parameters
        ----------
        key : str
            Unique identifier for the registered object.
        obj : callable, optional
            The object to register.  If ``None``, returns a decorator.

        Returns
        -------
        callable
            The registered object (or a decorator if *obj* is ``None``).
        """
        raise NotImplementedError

    def get(self, key: str) -> Callable[..., Any]:
        """Retrieve a registered object by key.

        Parameters
        ----------
        key : str
            Previously registered key.

        Returns
        -------
        callable
            The registered object.

        Raises
        ------
        KeyError
            If the key has not been registered.
        """
        raise NotImplementedError

    def list_keys(self) -> list[str]:
        """Return all registered keys.

        Returns
        -------
        list[str]
            Sorted list of registered keys.
        """
        raise NotImplementedError
