# encoding: utf-8
#
# Copyright (C) 2010-2013 Alec Thomas <alec@swapoff.org>
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.
#
# Author: Alec Thomas <alec@swapoff.org>

"""Schema validation for Python data structures.

Given eg. a nested data structure like this:

    {
        'exclude': ['Users', 'Uptime'],
        'include': [],
        'set': {
            'snmp_community': 'public',
            'snmp_timeout': 15,
            'snmp_version': '2c',
        },
        'targets': {
            'localhost': {
                'exclude': ['Ping'],
                'include': ['Uptime']
                'features': {
                    'Uptime': {
                        'retries': 3,
                    },
                    'Users': {
                        'snmp_community': 'monkey',
                        'snmp_port': 15,
                    },
                },
                'include': ['Users'],
                'set': {
                    'snmp_community': 'monkeys',
                },
            },
        },
    }

A schema like this:

    >>> settings = {
    ...   'snmp_community': str,
    ...   'retries': int,
    ...   'snmp_version': All(str, Any('3', '2c', '1')),
    ... }
    >>> features = [Any('Ping', 'Uptime', 'Http')]
    >>> schema = Schema({
    ...    'exclude': features,
    ...    'include': features,
    ...    'set': settings,
    ...    'targets': {
    ...      'exclude': features,
    ...      'include': features,
    ...      'features': {
    ...        'Uptime': settings,
    ...        'Users': settings
    ...      },
    ...    },
    ... })

Validate like so:

    >>> schema({
    ...   'set': {
    ...     'snmp_community': 'public',
    ...     'snmp_version': '2c',
    ...     'retries': 25
    ...   },
    ...   'include': ['Uptime'],
    ...   'exclude': ['Ping'],
    ...   'targets': {
    ...     'exclude': ['Ping'],
    ...     'include': ['Uptime'],
    ...     'features': {
    ...       'Uptime': {'retries': 3,
    ...                  'snmp_community': 'wolf',
    ...                  'snmp_version': '2c'},
    ...       'Users': {'snmp_community': 'monkey',
    ...                 'snmp_version': '3',
    ...                 'retries': 15},
    ...     },
    ...   },
    ... })  # doctest: +NORMALIZE_WHITESPACE
    {'exclude': ['Ping'],
     'include': ['Uptime'],
     'set': {'snmp_version': '2c', 'snmp_community': 'public', 'retries': 25},
     'targets': {'exclude': ['Ping'],
                 'include': ['Uptime'],
                 'features': {'Uptime':
                       {'retries': 3, 'snmp_community': 'wolf', 'snmp_version': '2c'},
                              'Users':
                       {'snmp_version': '3', 'snmp_community': 'monkey', 'retries': 15}}}}

"""
from collections import defaultdict
from functools import wraps
import os
import re
import sys
if sys.version > '3':
    import urllib.parse as urlparse
    long = int
    unicode = str
else:
    import urlparse


__author__ = 'Alec Thomas <alec@swapoff.org>'
__maintainer__ = 'Sebastien Requiem <sebastien.requiem@gmail.com>'
__version__ = '0.6.1'


class Undefined(object):
    def __nonzero__(self):
        return False

    def __repr__(self):
        return '...'


UNDEFINED = Undefined()

import collections

class TransformedDict(collections.MutableMapping):
    """A dictionary which applies an arbitrary key-altering function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs)) # use the free update to set keys

    def __getitem__(self, key):
        try:
            return self.store[key]
        except KeyError:
            return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        try:
            del self.store[self.__keytransform__(key)]
        except KeyError:
            del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __contains__(self, key):
        ret = key in self.store
        if ret is False:
            return self.__keytransform__(key) in self.store

        return ret

    def __keytransform__(self, key):
        return key


class MyTransformedDict(TransformedDict):
  def __keytransform__(self, key):
    return Optional(key)


class ValidationError(Exception):
    def __init__(self, errors=None):
        self.errors = errors or defaultdict(list)

    def add(self, path, error):
        if isinstance(path, list):
            path = "data[%s]" % "][".join(map(repr, path))

        self.errors[path].append(error)

    def add_multi(self, errors):
        for err in errors.errors:
            for exc in errors.errors[err]:
                self.errors[err].append(exc)

    def __str__(self):
        return self.msg

    def __len__(self):
        return len(self.errors.keys())

    @property
    def msg(self):
        first_element = self.errors.keys()[0]
        return "%s @ %s" % (self.errors[first_element][0], first_element)


class Schema(object):
    """A validation schema.

    The schema is a Python tree-like structure where nodes are pattern
    matched against corresponding trees of values.

    Nodes can be values, in which case a direct comparison is used, types,
    in which case an isinstance() check is performed, or callables, which will
    validate and optionally convert the value.
    """

    def __init__(self, schema, extra=False, coerce=False):
        """Create a new Schema.

        :param schema: Validation schema. See :module:`voluptuous` for details.
        :param extra: Keys in the data need not have keys in the schema.
        :param coerce: Whether to coerce values by default.
        """
        self.schema = schema
        self.extra = extra
        self._coerce = coerce

    def __call__(self, data, path=[]):
        """Validate data against this schema."""
        return self.validate(path, self.schema, data)

    def validate(self, path, schema, data):
        """
        According to the schema given, we will try
        to valdate the data. Each method process the data differently
        """
        try:
            if isinstance(schema, dict):
                return self.validate_dict(path, schema, data)
            elif isinstance(schema, list):
                return self.validate_list(path, schema, data)
            elif isinstance(schema, tuple):
                return self.validate_tuple(path, schema, data)
            type_ = type(schema)
            if type_ is type:
                type_ = schema
            if type_ in (int, long, str, unicode, float, complex, object,
                         list, dict, type(None)) or callable(schema):
                return self.validate_scalar(path, schema, data, self._coerce)
        except ValidationError as e:
            raise
        except (ValueError, TypeError) as e:
            errors = ValidationError()
            errors.add(path, e)
            raise errors

        raise ValueError('unsupported schema data type %r')


    def validate_dict(self, path, schema, data):
        """Validate a dictionary.

        A dictionary schema can contain a set of values, or at most one
        validator function/type.

        A dictionary schema will only validate a dictionary:

            >>> validate = Schema({})
            >>> validate([])
            Traceback (most recent call last):
            ...
            ValidationError: expected a dictionary @ data[]

        An invalid dictionary value:

            >>> validate = Schema({'one': 'two', 'three': 'four'})
            >>> validate({'one': 'three'})
            Traceback (most recent call last):
            ...
            ValidationError: not a valid value @ data['one']

        A missing required key:

            >>> validate({'two': 'three'})
            Traceback (most recent call last):
            ...
            ValidationError: Missing key @ data['one']

        An invalid key:

            >>> validate({'one': 'two', 'two': 'three', 'three': 'four'})
            Traceback (most recent call last):
            ...
            ValidationError: extra keys not allowed @ data['two']

        Validation function, in this case the "int" type:

            >>> validate = Schema({'one': 'two', 'three': 'four', int: str})

        """

        # We transform the schema to activate the Key Lookup on Optional classes.
        schema = MyTransformedDict(schema)

        if not isinstance(data, dict):
            raise ValueError('expected a dictionary')

        out = type(data)()
        required_keys = set(key for key in schema if not isinstance(key, Optional))

        errors = ValidationError()

        if not self.extra:
            schema_keys = [key.schema if isinstance(key, Marker) else key for key in schema.keys()]
            extra_keys = set(data.keys()).difference(schema_keys)
            if extra_keys:
                for key in extra_keys:
                    errors.add(path + [key], ValueError('extra keys not allowed'))

        for key, value in data.items():
            key_path = path + [key]

            # Is the key present in the schema {key: <Rule>} or {Optional(key): <Rules>}
            if key in schema:
                # We found the key, we can discard it from the required_keys
                required_keys.discard(key)
                svalue = schema[key]
                try:
                    out[key] = self.validate(key_path, svalue, value)
                except ValidationError as e:
                    for err in e.errors:
                        for exc in e.errors[err]:
                            errors.add(err, exc)
                except (ValueError, TypeError) as e:
                    errors.add(key_path, e)
            elif self.extra:
                out[key] = value

        for key in required_keys:
            msg = key.msg if hasattr(key, 'msg') and key.msg else 'required key not provided'
            errors.add(path + [key], ValueError("Missing key"))

        if len(errors):
            raise errors
        return out

    def _validate_sequence(self, path, schema, data, seq_type):
        """Validate a sequence type.

        This is a sequence of valid values or validators tried in order.

        >>> validator = Schema(['one', 'two', int])
        >>> validator(['one', 'two', 2])
        ['one', 'two', 2]
        >>> validator(['one'])
        Traceback (most recent call last):
        ...
        ValidationError: Missing key @ data[1]
        >>> validator([3.5])
        Traceback (most recent call last):
        ...
        ValidationError: not a valid value @ data[0]
        """
        seq_type_name = seq_type.__name__
        if not isinstance(data, seq_type):
            raise TypeError('expected a %s' % seq_type_name)

        # Empty seq schema, allow any data.
        if not schema:
            return data

        out = []
        invalid = None
        errors = ValidationError()

        index_path = UNDEFINED
        for i, s in enumerate(schema):
            index_path = path + [i]
            try:
                value = data[i]
                try:
                    out.append(self.validate(index_path, s, value))
                except (ValueError, TypeError) as e:
                    errors.add(index_path, e)
                except ValidationError as e:
                    errors.add_multi(e)
            except IndexError as e:
                errors.add(index_path, ValueError("Missing key"))

        if len(errors):
            raise errors

        return type(data)(out)

    def validate_tuple(self, path, schema, data):
        """Validate a tuple.

        A tuple is a sequence of valid values or validators tried in order.

        >>> validator = Schema(('one', 'two', int))
        >>> validator(('one', 'two', 3))
        ('one', 'two', 3)
        >>> validator((3.5,))
        Traceback (most recent call last):
        ...
        ValidationError: not a valid value @ data[0]
        """
        return self._validate_sequence(path, schema, data, seq_type=tuple)

    def validate_list(self, path, schema, data):
        """Validate a list.

        A list is a sequence of valid values or validators tried per index.

        >>> validator = Schema(['one', 'two', int])
        >>> validator(['one'])
        Traceback (most recent call last):
        ...
        ValidationError: Missing key @ data[1]
        >>> validator([3.5])
        Traceback (most recent call last):
        ...
        ValidationError: not a valid value @ data[0]
        """
        return self._validate_sequence(path, schema, data, seq_type=list)

    @staticmethod
    def validate_scalar(path, schema, data, coerce=False):
        """A scalar value.

        The schema can either be a value or a type.

        >>> Schema.validate_scalar([], int, 1)
        1
        >>> Schema.validate_scalar([], float, '1')
        Traceback (most recent call last):
        ...
        TypeError: expected float

        Callables have
        >>> Schema.validate_scalar([], lambda v, p: float(v), '1')
        1.0

        As a convenience, ValueError's are trapped:

        >>> Schema.validate_scalar([], lambda v, p: float(v), 'a')
        Traceback (most recent call last):
        ...
        ValueError: could not convert string to float: a
        """
        if isinstance(schema, type):
            if isinstance(data, schema):
                return data
            elif not coerce:
                raise TypeError('expected %s' % schema.__name__)
        if callable(schema):
            return schema(data, path)
        else:
            if data != schema:
                raise ValueError('not a valid value')
        return data

class Marker(object):
    """Mark nodes for special treatment."""

    def __init__(self, schema, msg=None):
        self.schema = schema
        self._schema = Schema(schema)
        self.msg = msg

    def __eq__(self, v):
        if isinstance(v, self.__class__):
            return self.schema == v.schema
        return self.schema == v

    def __hash__(self):
        return hash(self.schema)

    def __ne__(self, v):
        return self.schema != v

    def __call__(self, v):
        try:
            return self._schema(v)
        except Invalid as e:
            if not self.msg or len(e.path) > 1:
                raise
            raise Invalid(self.msg)

    def __str__(self):
        return str(self.schema)

    def __repr__(self):
        return repr(self.schema)

class Optional(Marker):
    """Mark a node in the schema as optional."""


def Extra(_):
    """Allow keys in the data that are not present in the schema."""
    raise SchemaError('"Extra" should never be called')

# As extra() is never called there's no way to catch references to the
# deprecated object, so we just leave an alias here instead.
extra = Extra

def IsTrue():
    """Assert that a value is true, in the Python sense.

    >>> validate = Schema(IsTrue())

    "In the Python sense" means that implicitly false values, such as empty
    lists, dictionaries, etc. are treated as "false":

    >>> validate([])
    Traceback (most recent call last):
    ...
    ValidationError: value was not true @ data[]
    >>> validate([1])
    [1]
    >>> validate(False)
    Traceback (most recent call last):
    ...
    ValidationError: value was not true @ data[]

    ...and so on.
    """
    def f(v, path):
        if v:
            return v
        raise ValueError('value was not true')
    return f


def IsFalse():
    """Assert that a value is false, in the Python sense.

    (see :func:`IsTrue` for more detail)

    >>> validate = Schema(IsFalse())
    >>> validate([])
    []
    """
    def f(v, path):
        if v:
            raise ValueError('value was not false')
        return v
    return f


def Boolean():
    """Convert human-readable boolean values to a bool.

    Accepted values are 1, true, yes, on, enable, and their negatives.
    Non-string values are cast to bool.

    >>> validate = Schema(Boolean())
    >>> validate(True)
    True
    >>> validate('moo')
    Traceback (most recent call last):
    ...
    ValidationError: expected boolean @ data[]
    """
    def f(v, path):
        if isinstance(v, basestring):
            v = v.lower()
            if v in ('1', 'true', 'yes', 'on', 'enable'):
                return True
            if v in ('0', 'false', 'no', 'off', 'disable'):
                return False
            raise ValueError('expected boolean')
        return bool(v)
    return f


def Any(*validators, **kwargs):
    """Use the first validated value.

    :param msg: Message to deliver to user if validation fails.
    :returns: Return value of the first validator that passes.

    >>> validate = Schema(Any('true', 'false',
    ...                       All(Any(int, bool))))
    >>> validate('true')
    'true'
    >>> validate(1)
    1
    >>> validate('moo')
    Traceback (most recent call last):
    ...
    ValidationError: no rules are matching @ data[]
    """
    msg = kwargs.pop('msg', None)
    schemas = [Schema(val) for val in validators]

    @wraps(Any)
    def f(v, path):

        for schema in schemas:
            try:
                return schema(v, path)
            except (ValueError, TypeError, ValidationError) as e:
                pass
        else:
            raise ValueError("no rules are matching")

    return f


def All(*validators, **kwargs):
    """Value must pass all validators.

    The output of each validator is passed as input to the next.

    :param msg: Message to deliver to user if validation fails.

    >>> validate = Schema(All('10', str))
    >>> validate('10')
    '10'
    """
    msg = kwargs.pop('msg', None)
    schemas = [Schema(val) for val in validators]

    @wraps(All)
    def f(v, path):
        errors = ValidationError()
        for schema in schemas:
            try:
                v = schema(v, path)
            except (ValueError, TypeError) as e:
                errors.add(path, e)
            except ValidationError as e:
                errors.add_multi(e)

        if len(errors):
            raise errors

        return v
    return f


def Match(pattern, msg=None):
    """Value must match the regular expression.

    >>> validate = Schema(Match(r'^0x[A-F0-9]+$'))
    >>> validate('0x123EF4')
    '0x123EF4'
    >>> validate('123EF4')
    Traceback (most recent call last):
    ...
    ValidationError: does not match regular expression @ data[]

    Pattern may also be a compiled regular expression:

    >>> validate = Schema(Match(re.compile(r'0x[A-F0-9]+', re.I)))
    >>> validate('0x123ef4')
    '0x123ef4'
    """
    if isinstance(pattern, basestring):
        pattern = re.compile(pattern)

    @wraps(Match)
    def f(v, path):
        if not pattern.match(v):
            raise ValueError(msg or 'does not match regular expression')
        return v
    return f


def Replace(pattern, substitution, msg=None):
    """Regex substitution.

    >>> validate = Schema(All(Replace('you', 'I'),
    ...                       Replace('hello', 'goodbye')))
    >>> validate('you say hello')
    'I say goodbye'
    """
    if isinstance(pattern, basestring):
        pattern = re.compile(pattern)

    @wraps(Replace)
    def f(v, path):
        return pattern.sub(substitution, v)
    return f


def Url():
    """Verify that the value is a URL."""
    @wraps(Url)
    def f(v, path):
        try:
            urlparse.urlparse(v)
            return v
        except:
            raise ValueError('expected a URL')
    return f


def IsFile():
    """Verify the file exists."""
    @wraps(IsFile)
    def f(v, path):
        ret = os.path.isfile(v)
        if not ret:
            raise TypeError("not a file")
        return ret
    return f


def IsDir():
    """Verify the directory exists.

    >>> IsDir()('/', [])
    '/'
    """
    @wraps(IsDir)
    def f(v, path):
        ret = os.path.isdir(v)
        if not ret:
            raise TypeError("not a directory")
        return v
    return f


def PathExists():
    """Verify the path exists, regardless of its type."""
    @wraps(PathExists)
    def f(v, path):
        ret = os.path.exists(v)
        if not ret:
            raise TypeError("path does not exist")
        return v
    return f


def Range(min=None, max=None, msg=None):
    """Limit a value to a range.

    Either min or max may be omitted.

    :raises Invalid: If the value is outside the range.
    """
    @wraps(Range)
    def f(v, path):
        if min is not None and v < min:
            raise ValueError(msg or 'value must be at least %s' % min)
        if max is not None and v > max:
            raise ValueError(msg or 'value must be at most %s' % max)
        return v
    return f


def Clamp(min=None, max=None, msg=None):
    """Clamp a value to a range.

    Either min or max may be omitted.
    """
    @wraps(Clamp)
    def f(v, path):
        if min is not None and v < min:
            v = min
        if max is not None and v > max:
            v = max
        return v
    return f


def Length(min=None, max=None, msg=None):
    """The length of a value must be in a certain range."""
    @wraps(Length)
    def f(v, path):
        if min is not None and len(v) < min:
            raise ValueError(msg or 'length of value must be at least %s' % min)
        if max is not None and len(v) > max:
            raise ValueError(msg or 'length of value must be at most %s' % max)
        return v
    return f


def DefaultTo(default_value, msg=None):
    """Sets a value to default_value if none provided.

    >>> s = Schema(DefaultTo(42, []))
    >>> s(None)
    42
    """
    @wraps(DefaultTo)
    def f(v, path):
        if v is None:
            v = default_value
        return v
    return f


if __name__ == '__main__':
    import doctest
    doctest.testmod()
