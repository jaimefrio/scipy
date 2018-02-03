"""Tests for argument conversion utilities."""
from __future__ import division, print_function, absolute_import

import itertools

from numpy.testing import assert_equal
from pytest import raises as assert_raises

from scipy.ndimage import convert


def test_ArgConverter_decorator_and_factories():
    @convert.ArgConverter
    def f(x, y, z):
        return x + y + z

    assert_equal(f(2, 3, 5), 10)

    g = f.of('y', 'z')
    assert_equal(g(2, 3, 5), 10)
    assert_equal(g.extra_args, ('y', 'z'))

    h = g.with_arg('x')
    assert_equal(h(2, 3, 5), 10)
    assert_equal(h.arg, 'x')
    assert_equal(h.extra_args, ('y', 'z'))


def test_ArgConverter_update_callargs():

    @convert.ArgConverter
    def double(x):
        return 2 * x

    @convert.ArgConverter
    def add_together(x, y):
        return x + y

    double = double.with_arg('a')
    add_together = add_together.with_arg('b').of('a')

    callargs = {'a': 1, 'b': 2}
    double.update_callargs(callargs)
    add_together.update_callargs(callargs)
    assert_equal(callargs, {'a': 2, 'b': 4})

    callargs = {'a': 1, 'b': 2}
    add_together.update_callargs(callargs)
    double.update_callargs(callargs)
    assert_equal(callargs, {'a': 2, 'b': 3})


def test_order_converters():
    a = convert.ArgConverter(None).with_arg('a').of('b')
    b = convert.ArgConverter(None).with_arg('b').of('c', 'd')
    c = convert.ArgConverter(None).with_arg('c')
    d = convert.ArgConverter(None).with_arg('d')
    for converters in itertools.permutations((a, b, c, d)):
        converters = [converter.arg
                      for converter in convert.order_converters(converters)]
        assert_equal(converters[-2:], ['b', 'a'])
        assert_equal(sorted(converters[:2]), ['c', 'd'])

    # This creates a dependency cycle, a -> b -> c -> a.
    c = convert.ArgConverter(None).with_arg('c').of('a')
    for converters in itertools.permutations((a, b, c, d)):
        with assert_raises(RuntimeError):
            convert.order_converters(converters)


def test_convert_args():

    @convert.ArgConverter
    def double(x):
        return 2 * x

    @convert.ArgConverter
    def triple(x):
        return 3 * x

    @convert.ArgConverter
    def add_together(x, y, z):
        return x + y + z

    @convert.convert_args(a=add_together.of('b', 'c'),
                          b=double,
                          c=triple)
    def multiply_together(a, b, c):
        return a * b * c

    assert_equal(multiply_together(2, 3, 5), (2 + 6 + 15) * 6 * 15)







