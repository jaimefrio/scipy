"""Utilities to convert and validate arguments of a function."""
from __future__ import division, print_function, absolute_import

import collections
import functools
import inspect
import numpy as np


# This module simplifies sanitizing function inputs.
#
# Say we want to write a function foo that takes an n-d array and an
# axis of that array, and returns a 1-d array of the same size as the
# dimension of the first array indicated by the axis. Because we want
# to allow reusing existing arrays for the output, the signature of
# this function could look something like:
#
#   def foo(array, axis=-1, output=None):
#       ...
#
# The first not-so-few lines of code of such a function would likely be
# dealing with validating and sanitizing its inputs. Especially if we
# have more than one function with a similar pattern, it would be good
# to extract these conversions and validations to dedicated functions,
# e.g. we could have:
#
#   def convert_to_input_array(array):
#       ...
#
#   def convert_to_valid_axis(axis, array):
#       ...
#
#   def convert_to_1d_output(output, array, axis):
#       ...
#
# And then call these at the beginning of our foo function:
#
#   def foo(array, axis=-1, output=None):
#       array = convert_to_input_array(array)
#       axis = convert_to_valid_axis(axis, array)
#       output = convert_to_1d_output(output, array, axis)
#       ...
#
# Note that the order of application of the converters is important, so
# that e.g. the axis converter can rely on its second argument being an
# array, and not a list of lists, and for similar reasons the output
# converter should be applied last.
#
# For functions with many interdependent inputs, even after moving the
# conversion logic to dedicated functions, these conversions can end up
# taking a substantial chunk of the function's code, making it harder
# to understand at first glance what the actual purpose of the function
# is.
#
# This module provides two things:
#
#  * a set of converters tailored to the needs of ndimage, to avoid
#    repetition and unify behavior, and
#  * machinery to simplify application of converters, removing the need
#    to call them explicitly, and ensuring that they are called in the
#    proper order.
#
# For the above example, we would move the three converter functions
# into this module, decorate them with the @Converter decorator, and
# give them a slightly shorter name:
#
#   @Converter
#   def to_array(array_like):
#       ...
#
#   @Converter
#   def to_valid_axis(axis, array):
#       ...
#
#   @Converter
#   def to_1d_output(output, array, axis):
#       ...
#
# Then in the module where foo lives, we would do the following:
#
#   from scipy.ndimage import convert
#
#   @convert.function_args(
#       array=convert.to_array,
#       axis=convert.to_valid_axis.using('array'),
#       output=convert.to_1d_output.using('array', 'axis'))
#   def foo(array, axis=-1):
#       ...
#
# The decorator will make sure that the function inputs are sanitized
# by applying the converters in the right order, so the code in foo only
# needs to deal with doing whatever foo is supposed to do.
#
# This example also shows the naming convention used in this module: if
# a function or class is to be used elsewhere, it has been named so that
# it makes sense with the module's name (convert) prepended to it.


class Converter(object):
    """Decorator for arg converter functions.

    Given a main function taking several arguments, a converter function
    is one that returns its first argument converted and validated so
    that it is safe to use inside the main function. The converter may
    take extra arguments, but these are also expected to be arguments
    of the main function.

    A decorated converter function carries metadata on the names of the
    arguments of the main function it is supposed to act on, that is
    used by the function_args decorator to automatically apply all
    converters and call the main function with sanitized arguments.

    Single argument converter functions, with no extra args, will be
    automatically wrapped if used as a converter, so they need not be
    explicitly decorated.
    """

    def __init__(self, converter, arg=None, extra_args=()):
        """Initializes a new converter.

        Parameters
        ----------
        converter : callable
            The converter function.
        arg: str
            The name of the argument of the main function that the
            converter should be applied to. This metadata is added by
            the ``function_args`` decorator, by calling the ``with_arg``
            method of the provided instance, so it should normally not
            be passed to the constructor directly.
        extra_args: sequence of str
            The names of the other arguments of the main function that
            should be passed to the converter function. This metadata is
            normally not passed to the constructor directly, but added
            by calling the ``with_extra_args`` method (or its far more
            readable alias ``using``) of an existing instance.
        """
        if not isinstance(converter, collections.Callable):
            raise TypeError('converter function must be callable')
        self.converter = converter
        self.arg = arg
        self.extra_args = extra_args

    def with_arg(self, arg):
        """Returns a new Converter with the provided arg."""
        return Converter(self.converter, arg, self.extra_args)

    def with_extra_args(self, *extra_args):
        """Returns a new Converter with the provided extra args."""
        return Converter(self.converter, self.arg, extra_args)

    # An alias of with_extra_args that enables more readable code.
    using = with_extra_args

    def update_callargs(self, callargs):
        """Updates a dictionary of call args by applying this converter."""
        if (self.arg not in callargs or
                any(arg not in callargs for arg in self.extra_args)):
            raise RuntimeError()
        converter_args = [callargs[self.arg]]
        converter_args.extend(callargs[extra_arg]
                              for extra_arg in self.extra_args)
        callargs[self.arg] = self(*converter_args)

    def __call__(self, *args, **kwargs):
        return self.converter(*args, **kwargs)


def sorted_converters(converters):
    """Sorts a list of converters to a safe application order.

    Does a topological sort, based on the converter dependencies, to
    determine an order of application of the converters that will avoid
    items being used before being converted, or will raise an error if
    no such order exists. See:

    https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search

    for a detailed description of the algorithm.
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

    for converter in arg_converters.values():
        visit(converter)

    return ordered_converters


def function_args(**arg_converters):
    """Decorator that applies converters to a function's arguments."""
    converters = sorted_converters(
        converter.with_arg(arg) if isinstance(converter, Converter)
        else Converter(converter, arg=arg)
        for arg, converter in arg_converters.items())
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            callargs = inspect.getcallargs(f, *args, **kwargs)
            for converter in converters:
                converter.update_callargs(callargs)
            return f(**callargs)
        return wrapper
    return decorator


# Converters #


SUPPORTED_TYPECODES = set('?bBhHiIlLqQpPfd')


@Converter
def to_input_array(array_like):
    """Converter to input array.

    An input array is an aligned, not swapped array of supported dtype.
    """
    ans_array = _to_aligned_not_swapped_array(array_like)
    _validate_array_dtype(ans_array)
    return ans_array


@Converter
def to_output_array(array_or_dtype, input_array=None):
    """Converter to output array.

    If an input array is provided this function behaves as a normal
    converter, either validating an array against the shape of the input
    or creating an array of the requested dtype matching the input's
    shape. Even after this conversion, if the array was provided by the
    user it may be misaligned or swapped.

    To make sure that we work with an aligned, not swapped array, this
    function can also be used as a context manager, by leaving out the
    input array argument. It will provide an aligned, not swapped array
    to work on and, if needed, copy the contents back to the original
    array on exit.

    The typical use case needs both behaviors and will look like:

        @convert.function_args(
            ...
            output=convert.to_output_array.using('input'),
            ...)
        def function(input, ..., output, ...):
            ...
            with convert.to_output_array(output) as output_array:
                ...
                # Operate on output_array within the context.
                ...
            return output

    The output array inside the context manager will be a writeable,
    aligned, not swapped array of a supported dtype.
    """
    if input_array is None:
        return _to_aligned_not_swapped_writeback_array(array_or_dtype, False)
    elif isinstance(array_or_dtype, np.ndarray):
        if not array_or_dtype.flags.writeable:
            raise ValueError('output array is read-only')
        if array_or_dtype.shape != input_array.shape:
            raise ValueError('output array shape incompatible with input')
        output_array = array_or_dtype
    else:
        if array_or_dtype is None:
            dtype = input_array.dtype
        else:
            try:
                dtype = np.dtype(array_or_dtype)
            except TypeError:
                raise TypeError('output must be an array or dtype')
        output_array = np.empty_like(input_array, dtype=dtype)
    _validate_array_dtype(output_array)
    return output_array


@Converter
def to_1d_weights_array(array_like):
    """Converter to 1-d weights array.

    A one-dimensional weights array is a non-empty, one-dimensional,
    contiguous, aligned, not swapped array of float64 dtype.
    """
    weights = _to_contiguous_aligned_not_swapped_array(array_like, float)
    if weights.ndim != 1:
        raise ValueError('weights array must be 1-dimensional')
    if weights.size == 0:
        raise ValueError('weights array is empty')
    return weights


@Converter
def to_valid_axis(axis, array):
    """Converter to valid axis of array."""
    if axis < -array.ndim or axis >= array.ndim:
        raise ValueError('invalid axis ({}) for {}-dimensional '
                         'array'.format(axis, array.ndim))
    if axis < 0:
        axis += array.ndim
    return axis


@Converter
def to_valid_origin(origin, weights_1d):
    """Converter to valid origin of weights_1d."""
    _validate_origin(origin, weights_1d.size)
    return origin


def _validate_array_dtype(array):
    """Checks that the array's dtype is supported."""
    dtype = array.dtype
    if not dtype.isbuiltin or dtype.char not in SUPPORTED_TYPECODES:
        raise TypeError('unsupported dtype {}'.format(dtype))


def _validate_origin(origin, size):
    """Checks that an origin is valid for the given size."""
    base_index = origin + size // 2
    if base_index < 0 or base_index >= size:
        raise ValueError('invalid origin {} for size {}'.format(origin, size))


def _to_aligned_not_swapped_array(array_like):
    """Converts an array-like to an aligned, not-swapped array."""
    ans_array = np.asarray(array_like)
    if not ans_array.flags.aligned or not ans_array.dtype.isnative:
        dtype = ans_array.dtype.newbyteorder('=')
        ans_array = np.array(ans_array, dtype=dtype)
    return ans_array


def _to_contiguous_aligned_not_swapped_array(array_like, dtype=None):
    """Converts an array-like to a contiguous, aligned, not-swapped array."""
    cans_array = np.ascontiguousarray(array_like, dtype=dtype)
    return _to_aligned_not_swapped_array(cans_array)


class _to_aligned_not_swapped_writeback_array(object):
    """Context manager that provides an aligned, not-swapped output array.

    This makes functionality similar to the WRITEBACKIFCOPY mechanism of
    NumPy's C API available from Python: if the passed array is not
    aligned or is byte-swapped, it will create an empty, aligned,
    not-swapped array of the same shape an dtype, and optionally copy
    the contents of the original array into it. Any changes made to this
    array will be copied back into the original array on exit.
    """

    def __init__(self, array, copy):
        """Instance initialization.

        Parameters
        ----------
        array: ndarray
            The array to convert, if needed, into a write-back aligned,
            not swapped one.
        copy: bool
            Whether the contents of array should be copied into the
            write-back array on creation

        """
        self.array = array
        self.copy = copy

    def __enter__(self):
        if self.array.flags.aligned and self.array.dtype.isnative:
            self.ans_array = self.array
        else:
            dtype = self.array.dtype.newbyteorder('=')
            self.ans_array = np.empty_like(self.array, dtype=dtype)
            if self.copy:
                self.ans_array[:] = self.array
        return self.ans_array

    def __exit__(self, exc_type, exc_value, traceback):
        if self.array is not self.ans_array:
            self.array[:] = self.ans_array










@Converter
def to_weights_array(weights_like, input_array):
    """Returns a contiguous, aligned, non-swapped, double input array."""
    weights = _to_contiguous_aligned_not_swapped_array(weights_like, float)
    if weights.ndim != input_array.ndim:
        raise ValueError('weights array shape incompatible with input')
    return weights


@Converter
def to_int_size(size):
    """Returns the size if it is a positive integer."""
    if not size == int(size) or size < 1:
        raise ValueError('size ({}) must be a positive integer'.format(size))
    return int(size)





