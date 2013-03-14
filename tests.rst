Error reporting should be accurate::

  >>> from voluptuous import *
  >>> schema = Schema(['one', {'two': 'three', 'four': ['five'],
  ...                          'six': {'seven': 'eight'}}])
  >>> res = schema(['one', {'two': 'three', 'four': ['five'],
  ...                          'six': {'seven': 'eight'}}])
  >>> [res[0], sorted(res[1].items())]
  ['one', [('four', ['five']), ('six', {'seven': 'eight'}), ('two', 'three')]]

It should show the exact index and container type, in this case a list value::

  >>> schema(['one', 'two'])
  Traceback (most recent call last):
  ...
  ValidationError: expected a dict @ 1.

It should also be accurate for nested values::

  >>> schema(['one', {'two': 'nine', 'four': ['five'],
  ...                          'six': {'seven': 'eight'}}])
  Traceback (most recent call last):
  ...
  ValidationError: Not a valid value @ 1.two.
  >>> schema(['one', {'two': 'three', 'four': ['six'],
  ...                          'six': {'seven': 'eight'}}])
  Traceback (most recent call last):
  ...
  ValidationError: Not a valid value @ 1.four.0.
  >>> schema(['one', {'two': 'three', 'four': ['five'],
  ...                          'six': {'seven': 'nine'}}])
  Traceback (most recent call last):
  ...
  ValidationError: Not a valid value @ 1.six.seven.

Errors should be reported depth-first::

  >>> validate = Schema({'one': {'two': 'three', 'four': 'five'}})
  >>> validate({'one': {'two': 'three', 'four': 'six'}})
  Traceback (most recent call last):
  ...
  ValidationError: Not a valid value @ one.four.


dict, list, and tuple should be available as type validators::

  >>> Schema(dict)({'a': 1, 'b': 2})
  {'a': 1, 'b': 2}
  >>> Schema(list)([1,2,3])
  [1, 2, 3]
  >>> Schema(tuple)((1,2,3))
  (1, 2, 3)


Validation should return instances of the right types when the types are
subclasses of dict or list::

  >>> class Dict(dict):
  ...   pass
  >>>
  >>> d = Schema(dict)(Dict(a=1, b=2))
  >>> d
  {'a': 1, 'b': 2}
  >>> type(d) is Dict
  True
  >>> class List(list):
  ...   pass
  >>>
  >>> l = Schema(list)(List([1,2,3]))
  >>> l
  [1, 2, 3]
  >>> type(l) is List
  True

Multiple errors are reported::

  >>> schema = Schema({'one': 1, 'two': 2})
  >>> try:
  ...   schema({'one': 2, 'two': 3, 'three': 4})
  ... except ValidationError as e:
  ...   paths = sorted(e.errors.keys())
  ...   print([(path, translate_exception(error)) for path in paths for error in e.errors[path]])  # doctest: +NORMALIZE_WHITESPACE
  [('one', 'Not a valid value'),
  ('three', 'Extra key not allowed'),
  ('two', 'Not a valid value')]
  >>> schema = Schema([[1], [2], [3]])
  >>> try:
  ...   schema([1, 2, 3])
  ... except ValidationError as e:
  ...   paths = sorted(e.errors.keys())
  ...   print([(path, translate_exception(error)) for path in paths for error in e.errors[path]])  # doctest: +NORMALIZE_WHITESPACE
  [('0', 'expected a list'),
   ('1', 'expected a list'),
   ('2', 'expected a list')]

Custom classes validate as schemas::

    >>> class Thing(object):
    ...     pass
    >>> schema = Schema(Thing)
    >>> t = schema(Thing())
    >>> type(t) is Thing
    True

Classes with custom metaclasses should validate as schemas::

    >>> class MyMeta(type):
    ...     pass
    >>> class Thing(object):
    ...     __metaclass__ = MyMeta
    >>> schema = Schema(Thing)
    >>> t = schema(Thing())
    >>> type(t) is Thing
    True

Schemas built with All() should give the same error as the original validator (Issue #26)::

    >>> schema = Schema({
    ...   'items': All([{
    ...     'foo': str
    ...   }])
    ... })

    >>> schema({'items': [{}]})
    Traceback (most recent call last):
    ...
    ValidationError: Missing mandatory value @ items.0.foo.
