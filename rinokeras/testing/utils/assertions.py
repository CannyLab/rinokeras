"""
Some handy assertions
"""

def assert_not_none(elems):
    if isinstance(elems, (list, tuple)):
        for e in elems:
            if e is None:
                raise AssertionError('Element: {} is None'.format(e))
    else:
       if elems is None:
                raise AssertionError('Element: {} is None'.format(e))

def assert_expected_shapes(elems, shapes):
    if isinstance(elems, (list, tuple)):
        for x,y in zip(elems, shapes):
            if x.shape != y:
                raise AssertionError('Element {} shape ({}) does not match {}'.format(x, x.shape, y))
    else:
        if elems.shape != shapes[0]:
            raise AssertionError('Element {} shape ({}) does not match {}'.format(elems, elems.shape, shapes))