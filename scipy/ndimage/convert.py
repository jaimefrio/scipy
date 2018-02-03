"""Utilities to convert and validate arguments of a function."""
from __future__ import division, print_function, absolute_import

import functools
import inspect


class ArgConverter(object):
    """Decorator for arg converter functions."""

    def __init__(self, converter, arg=None, extra_args=()):
        self.converter = converter
        self.arg = arg
        self.extra_args = extra_args

    def with_arg(self, arg):
        """Returns a new ArgConverter with the provided arg."""
        return ArgConverter(self.converter, arg, self.extra_args)

    def of(self, *extra_args):
        """Returns a new ArgConverter with the provided extra args."""
        return ArgConverter(self.converter, self.arg, extra_args)

    def update_callargs(self, callargs):
        """Updates a dictionary of call args by applying this converter."""
        assert self.arg is not None
        converter_args = [callargs[self.arg]]
        converter_args.extend(callargs[extra_arg]
                              for extra_arg in self.extra_args)
        callargs[self.arg] = self(*converter_args)

    def __call__(self, *args, **kwargs):
        return self.converter(*args, **kwargs)


def order_converters(converters):
    """Returns a list of converters with a safe application order.

    Does a topological sort, based on the converter dependencies, to determine
    an order of application of the converters that will avoid items being used
    before conversion, or will raise an error if not possible. See:

    https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search

    for a detailed explanation of the algorithm.
    """
    arg_converters = {converter.arg: converter for converter in converters}
    ordered_converters = []
    visited = set()
    ordered = set()

    def visit(converter):
        arg = converter.arg
        if arg in ordered:
            return
        if arg in visited:
            raise RuntimeError('converters have cyclic dependencies')
        visited.add(arg)
        for extra_arg in converter.extra_args:
            visit(arg_converters[extra_arg])
        ordered_converters.append(converter)
        ordered.add(arg)

    for converter in converters:
        visit(converter)

    return ordered_converters




def convert_args(**arg_converters):
    """Decorator that applies converters to a function's arguments."""
    converters = [converter.with_arg(arg) for arg, converter in arg_converters.items()]
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            callargs = inspect.getcallargs(f, *args, **kwargs)
            for converter in order_converters(converters):
                converter.update_callargs(callargs)
            return f(**callargs)
        return wrapper
    return decorator













