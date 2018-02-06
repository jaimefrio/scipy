"""Tests for argument conversion utilities."""
from __future__ import division, print_function, absolute_import

import collections
import itertools
import numpy as np
import pytest

from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises

from scipy.ndimage import convert


def test_Converter_decorator_and_factories():
    @convert.Converter
    def f(u, v, w, x, y, z):
        return u + v + w + x + y + z

    assert_(f.arg is None)
    assert_equal(f.extra_args, ())
    assert_equal(f(2, 3, 5, 7, 11, 13), 41)

    g = f.with_arg('u')
    assert_equal('u', g.arg)
    assert_equal(g.extra_args, ())
    assert_(g is not f)
    assert_equal(g(2, 3, 5, 7, 11, 13), 41)

    extra_args = ('v', 'w', 'x', 'y', 'z')
    for h in g.with_extra_args(*extra_args), g.using(*extra_args):
        assert_equal(h.arg, 'u')
        assert_equal(h.extra_args, ('v', 'w', 'x', 'y', 'z'))
        assert_(h is not g)
        assert_equal(h(2, 3, 5, 7, 11, 13), 41)


def test_Converter_input_type_check():
    with assert_raises(TypeError):
        converter = convert.Converter('a string is not callable!')


def test_Converter_update_callargs():

    @convert.Converter
    def double(x):
        return 2 * x

    @convert.Converter
    def add_together(x, y):
        return x + y

    double = double.with_arg('a')
    add_together = add_together.with_arg('b').with_extra_args('a')

    callargs = {'a': 1, 'b': 2}
    double.update_callargs(callargs)
    add_together.update_callargs(callargs)
    assert_equal(callargs, {'a': 2, 'b': 4})

    callargs = {'a': 1, 'b': 2}
    add_together.update_callargs(callargs)
    double.update_callargs(callargs)
    assert_equal(callargs, {'a': 2, 'b': 3})

    callargs = {'b': 2}
    with assert_raises(RuntimeError):
        double.update_callargs(callargs)
    with assert_raises(RuntimeError):
        add_together.update_callargs(callargs)


def test_sorted_converters():
    a = convert.Converter(lambda x: x).with_arg('a').with_extra_args('b')
    b = convert.Converter(lambda x: x).with_arg('b').with_extra_args('c', 'd')
    c = convert.Converter(lambda x: x).with_arg('c').with_extra_args('d')
    d = convert.Converter(lambda x: x).with_arg('d')
    for converters in itertools.permutations((a, b, c, d)):
        converters = [converter.arg
                      for converter in convert.sorted_converters(converters)]
        assert_equal(['d', 'c', 'b', 'a'], converters)

    # This creates a dependency cycle, a -> b -> c -> a.
    d = convert.Converter(lambda x: x).with_arg('d').with_extra_args('a')
    for converters in itertools.permutations((a, b, c, d)):
        with assert_raises(RuntimeError):
            convert.sorted_converters(converters)


def test_function_args():

    @convert.Converter
    def double(x):
        return 2 * x

    # Undecorated functions taking a single argument need not be decorated.
    def treble(x):
        return 3 * x

    @convert.Converter
    def add_together(x, y, z):
        return x + y + z

    @convert.function_args(
        a=add_together.using('b', 'c'),
        b=double,
        c=treble)
    def multiply_together(a, b, c):
        return a * b * c

    assert_equal(multiply_together(2, 3, 5), (2 + 2*3 + 3*5) * 2*3 * 3*5)


@pytest.mark.parametrize('typecode', convert.SUPPORTED_TYPECODES)
def test_validate_array_dtype_ok(typecode):
    convert._validate_array_dtype(np.empty((3,), dtype=typecode))


@pytest.mark.parametrize('typecode', 'egFDGMmO')
def test_validate_array_dtype_nok(typecode):
    with assert_raises(TypeError):
        convert._validate_array_dtype(np.empty((3,), dtype=typecode))


@pytest.mark.parametrize('size, first_good, first_bad',
                         [(3, -1, 2), (4, -2, 2)])
def test_validate_origin(size, first_good, first_bad):
    for origin in range(2 * first_good, 2 * first_bad):
        if first_good <= origin < first_bad:
            convert._validate_origin(origin, size)
        else:
            with assert_raises(ValueError):
                convert._validate_origin(origin, size)


def generate_test_arrays(shape):
    """Yields arrays with different flag and dtype combinations."""
    typecodes = convert.SUPPORTED_TYPECODES - {'?'}
    test_cases = itertools.product([False, True], repeat=3)
    for aligned, swapped, contiguous in test_cases:
        size = np.prod(shape)
        if not contiguous:
            size *= 2
        if not aligned:
            size += 1
        for typecode in typecodes:
            dtype = np.dtype(typecode)
            if dtype.itemsize == 1 and not aligned:
                # No can do: single byte dtype arrays are always aligned.
                continue
            if swapped:
                dtype = dtype.newbyteorder()
            array = np.empty(size, dtype=dtype)
            if not aligned:
                array = array.view(np.uint8)[1:-dtype.itemsize + 1].view(dtype)
            if not contiguous:
                array = array[::2]
            for permutation in itertools.permutations(range(len(shape))):
                permuted_shape = [None] * len(shape)
                inv_permutation = [None] * len(shape)
                for i, p in enumerate(permutation):
                    permuted_shape[i] = shape[p]
                    inv_permutation[p] = i
                array = array.reshape(permuted_shape).transpose(inv_permutation)
                array[:] = np.arange(array.size, dtype=dtype).reshape(shape)

                yield array


@pytest.mark.parametrize('array', generate_test_arrays((2, 3, 5)))
def test_to_aligned_not_swapped_array(array):
    ans_array = convert._to_aligned_not_swapped_array(array)
    if array.flags.aligned and array.dtype.isnative:
        assert_(ans_array is array)
    else:
        assert_(ans_array.flags.aligned)
        assert_(ans_array.dtype.isnative)
        assert_equal(ans_array, array)


@pytest.mark.parametrize('array', generate_test_arrays((2, 3, 5)))
def test_to_contiguous_aligned_not_swapped_array(array):
    cans_array = convert._to_contiguous_aligned_not_swapped_array(array)
    if array.flags.contiguous and array.flags.aligned and array.dtype.isnative:
        assert_(cans_array is array)
    else:
        assert_(cans_array.flags.contiguous)
        assert_(cans_array.flags.aligned)
        assert_(cans_array.dtype.isnative)
        assert_equal(cans_array, array)


@pytest.mark.parametrize('output', generate_test_arrays((2, 3, 5)))
def test_to_output_array_as_context_manager(output):
    with convert.to_output_array(output) as output_array:
        if output.flags.aligned and output.dtype.isnative:
            assert_(output_array is output)
        else:
            assert_(output_array is not output)
        assert_(output_array.flags.aligned)
        assert_(output_array.dtype.isnative)
        output_array[:] = np.arange(1, output.size + 1).reshape(output.shape)
    assert_equal(output, np.arange(1, output.size + 1).reshape(output.shape))













































