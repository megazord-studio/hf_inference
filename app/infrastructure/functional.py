"""Higher-order functions and functional programming utilities.

Provides composable, reusable functional patterns for the application.
"""

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")
E = TypeVar("E")


# Result type for error handling (functional alternative to exceptions)
Result = Tuple[Optional[R], Optional[E]]


def compose(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Compose functions from right to left.
    
    compose(f, g, h)(x) == f(g(h(x)))
    
    Args:
        *functions: Functions to compose
    
    Returns:
        Composed function
    
    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> add_then_double = compose(double, add_one)
        >>> add_then_double(3)  # (3 + 1) * 2
        8
    """
    def composed(arg: Any) -> Any:
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return composed


def pipe(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Pipe functions from left to right (Unix pipeline style).
    
    pipe(f, g, h)(x) == h(g(f(x)))
    
    Args:
        *functions: Functions to pipe
    
    Returns:
        Piped function
    
    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> process = pipe(add_one, double)
        >>> process(3)  # (3 + 1) * 2
        8
    """
    def piped(arg: Any) -> Any:
        result = arg
        for func in functions:
            result = func(result)
        return result
    return piped


def curry(func: Callable[..., R]) -> Callable[..., Any]:
    """
    Curry a function (partial application support).
    
    Args:
        func: Function to curry
    
    Returns:
        Curried function
    
    Example:
        >>> def add(a, b, c):
        ...     return a + b + c
        >>> curried_add = curry(add)
        >>> add_5 = curried_add(5)
        >>> add_5_and_3 = add_5(3)
        >>> add_5_and_3(2)  # 5 + 3 + 2
        10
    """
    def curried(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except TypeError:
            # Need more arguments, return partial
            def partial(*more_args: Any, **more_kwargs: Any) -> Any:
                return curried(*(args + more_args), **{**kwargs, **more_kwargs})
            return partial
    return curried


def safe_call(
    func: Callable[..., R], *args: Any, **kwargs: Any
) -> Result[R, str]:
    """
    Call a function and return (result, None) or (None, error_message).
    
    Functional error handling without exceptions in business logic.
    
    Args:
        func: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (result, error) where one is always None
    
    Example:
        >>> result, error = safe_call(int, "123")
        >>> if error is None:
        ...     print(f"Success: {result}")
        >>> result, error = safe_call(int, "invalid")
        >>> if error:
        ...     print(f"Error: {error}")
    """
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)


def maybe_map(func: Callable[[T], R], value: Optional[T]) -> Optional[R]:
    """
    Map a function over an Optional value (Maybe monad pattern).
    
    Args:
        func: Function to apply
        value: Optional value
    
    Returns:
        Transformed value or None
    
    Example:
        >>> double = lambda x: x * 2
        >>> maybe_map(double, 5)
        10
        >>> maybe_map(double, None)
        None
    """
    if value is None:
        return None
    return func(value)


def either(
    left_func: Callable[[E], R],
    right_func: Callable[[T], R],
    result: Result[T, E],
) -> R:
    """
    Handle a Result by applying left_func to error or right_func to value.
    
    Args:
        left_func: Function to apply to error
        right_func: Function to apply to success value
        result: Result tuple (value, error)
    
    Returns:
        Result of applying appropriate function
    
    Example:
        >>> handle_error = lambda e: f"Error: {e}"
        >>> handle_success = lambda v: f"Success: {v}"
        >>> either(handle_error, handle_success, (None, "failed"))
        'Error: failed'
        >>> either(handle_error, handle_success, (42, None))
        'Success: 42'
    """
    value, error = result
    if error is not None:
        return left_func(error)
    return right_func(value)  # type: ignore


def memoize(func: Callable[..., R]) -> Callable[..., R]:
    """
    Memoize a pure function (cache results by arguments).
    
    Only use for pure functions with hashable arguments.
    
    Args:
        func: Pure function to memoize
    
    Returns:
        Memoized version of function
    
    Example:
        >>> @memoize
        ... def expensive_calc(n):
        ...     print(f"Computing {n}")
        ...     return n * n
        >>> expensive_calc(5)  # Prints "Computing 5"
        25
        >>> expensive_calc(5)  # Uses cache, no print
        25
    """
    cache: Dict[Tuple[Any, ...], R] = {}
    
    def memoized(*args: Any) -> R:
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    return memoized


__all__ = [
    "Result",
    "compose",
    "pipe",
    "curry",
    "safe_call",
    "maybe_map",
    "either",
    "memoize",
]
