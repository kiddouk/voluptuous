Voluptuous is a Python data validation library
==============================================

.. image:: https://secure.travis-ci.org/alecthomas/voluptuous.png?branch=master
  :target: https://travis-ci.org/alecthomas/voluptuous

Voluptuous, *despite* the name, is a Python data validation library. It is
primarily intended for validating data coming into Python as JSON, YAML,
etc.

It has three goals:

1. Simplicity.
2. Support for complex data structures.
3. Provide useful error messages.

.. contents:: Table of Contents

Show me an example
------------------
Twitter's `user search API
<http://apiwiki.twitter.com/Twitter-REST-API-Method:-users-search>`_ accepts
query URLs like::

  $ curl 'http://api.twitter.com/1/users/search.json?q=python&per_page=20&page=1

To validate this we might use a schema like::

  >>> from voluptuous import Schema
  >>> schema = Schema({
  ...   'q': str,
  ...   'per_page': int,
  ...   'page': int,
  ... })

This schema very succinctly and roughly describes the data required by the API,
and will work fine. But it has a few problems. Firstly, it doesn't fully
express the constraints of the API. According to the API, ``per_page`` should
be restricted to at most 20, for example. To describe the semantics of the API
more accurately, our schema will need to be more thoroughly defined::

  >>> from voluptuous import All, Length, Range, Optional
  >>> schema = Schema({
  ...   'q': All(str, Length(min=1)),
  ...   Optional('per_page'): All(int, Range(min=1, max=20)),
  ...   Optional('page'): All(int, Range(min=0)),
  ... })

This schema fully enforces the interface defined in Twitter's documentation,
and goes a little further for completeness.

"q" is required::

  >>> schema({})
  Traceback (most recent call last):
  ...
  ValidationError: Missing mandatory value @ q.

...must be a string::

  >>> schema({'q': 123})
  Traceback (most recent call last):
  ...
  ValidationError: expected a str @ q.

...and must be at least one character in length::

  >>> schema({'q': ''})
  Traceback (most recent call last):
  ...
  ValidationError: length must be at least 1 @ q.
  >>> schema({'q': '#topic'})
  {'q': '#topic'}

"per_page" is a positive integer no greater than 20::

  >>> schema({'q': '#topic', 'per_page': 900})
  Traceback (most recent call last):
  ...
  ValidationError: value (900) must be at most 20 @ per_page.
  >>> schema({'q': '#topic', 'per_page': -10})
  Traceback (most recent call last):
  ...
  ValidationError: value (-10) must be at least 1 @ per_page.

"page" is an integer >= 0::

  >>> schema({'q': '#topic', 'page': 'one'})
  Traceback (most recent call last):
  ...
  ValidationError: expected a int @ page.
  >>> schema({'q': '#topic', 'page': 1})
  {'q': '#topic', 'page': 1}


Defining schemas
----------------
Schemas are nested data structures consisting of dictionaries, lists,
scalars and *validators*. Each node in the input schema is pattern matched
against corresponding nodes in the input data.

Literals
~~~~~~~~
Literals in the schema are matched using normal equality checks::

  >>> schema = Schema(1)
  >>> schema(1)
  1
  >>> schema = Schema('a string')
  >>> schema('a string')
  'a string'

Types
~~~~~
Types in the schema are matched by checking if the corresponding value is an
instance of the type::

  >>> schema = Schema(int)
  >>> schema(1)
  1
  >>> schema('one')
  Traceback (most recent call last):
  ...
  ValidationError: expected a int @ .


Lists
~~~~~
Lists are treated as a list of strict elements. Everything should match:


  >>> schema = Schema([1, 'a', 'string'])
  >>> schema([1])
  Traceback (most recent call last):
  ...
  ValidationError: Missing mandatory value @ 1.
  >>> schema([1, 1, 1])
  Traceback (most recent call last):
  ...
  ValidationError: Not a valid value @ 1.
  >>> schema(['a', 1, 'string', 1, 'string'])
  Traceback (most recent call last):
  ...
  ValidationError: Not a valid value @ 1.


Validation functions
~~~~~~~~~~~~~~~~~~~~
Validators are simple callables that raise an ``Invalid`` exception when they
encounter invalid data. The criteria for determining validity is entirely up to
the implementation; it may check that a value is a valid username with
``pwd.getpwnam()``, it may check that a value is of a specific type, and so on.

The simplest kind of validator is a Python function that raises `ValueError`
when its argument is invalid. Conveniently, many builtin Python functions have
this property. Here's an example of a date validator::

..note:: TypeError can also be used for type checking

  >>> from datetime import datetime
  >>> def Date(fmt='%Y-%m-%d'):
  ...   def f(v, p):
  ...      try:
  ...          return datetime.strptime(v, fmt)
  ...      except ValueError:
  ...          raise ValueError("DOESNOTMATCH", v, fmt)
  ...   return f

  >>> schema = Schema(Date())
  >>> schema('2013-03-03')
  datetime.datetime(2013, 3, 3, 0, 0)
  >>> schema('2013-03')
  Traceback (most recent call last):
  ...
  ValidationError: value 2013-03 does not match the regexp %Y-%m-%d @ .

.. _extra:

Dictionaries
~~~~~~~~~~~~
Each key-value pair in a schema dictionary is validated against the corresponding key in the data dictionary::

  >>> schema = Schema({1: 'one', Optional(2): 'two'})
  >>> schema({1: 'one'})
  {1: 'one'}

Extra dictionary keys
`````````````````````
By default any additional keys in the data, not in the schema will trigger
exceptions::

  >>> schema = Schema({2: 3})
  >>> schema({1: 2, 2: 3})
  Traceback (most recent call last):
  ...
  ValidationError: Extra key not allowed @ 1.

This behaviour can be altered on a per-schema basis with ``Schema(..., extra=True)``::

  >>> schema = Schema({2: 3}, extra=True)
  >>> schema({1: 2, 2: 3})
  {1: 2, 2: 3}


Required dictionary keys
````````````````````````
By default, keys in the schema are required to be in the data::

  >>> schema = Schema({1: 2, 3: 4})
  >>> schema({3: 4})
  Traceback (most recent call last):
  ...
  ValidationError: Missing mandatory value @ 1.


Optional dictionary keys
````````````````````````

Per default, all keys are required. Some keys may be individually marked as optional using the marker token ``Optional(key)``::

  >>> from voluptuous import Optional, ValidationError, translate_exception
  >>> schema = Schema({1: 2, Optional(3): 4})
  >>> schema({})
  Traceback (most recent call last):
  ...
  ValidationError: Missing mandatory value @ 1.
  >>> schema({1: 2})
  {1: 2}
  >>> schema({1: 2, 4: 5})
  Traceback (most recent call last):
  ...
  ValidationError: Extra key not allowed @ 4.
  >>> schema({1: 2, 3: 4})
  {1: 2, 3: 4}

Error reporting
---------------
Validators must throw an ``Invalid`` exception if invalid data is passed to
them. All other exceptions are treated as errors in the validator and will not
be caught.

Each ``Invalid`` exception has an associated ``path`` attribute representing
the path in the data structure to our currently validating value. This is used
during error reporting, but also during matching to determine whether an error
should be reported to the user or if the next match should be attempted. This
is determined by comparing the depth of the path where the check is, to the
depth of the path where the error occurred. If the error is more than one level
deeper, it is reported.

The upshot of this is that *matching is depth-first and fail-last*.

To illustrate this, here is an example schema::

  >>> schema = Schema([[2, 3], 6])

Each value in the top-level list is matched depth-first in-order. Given input
data of ``[[6]]``, the inner list will match the first element of the schema,
but the literal ``6`` will not match any of the elements of that list. This
error will be reported back to the user immediately. No backtracking is
attempted::

  >>> try:
  ...     schema([[6]])
  ... except ValidationError as e:
  ...     print translate_exception(e.errors['0.0'][0])
  Not a valid value

If we pass the data ``[6]``, the ``6`` is not a list type and so will not
recurse into the first element of the schema. This will create a type error::

  >>> try:
  ...     schema([6])
  ... except ValidationError as e:
  ...     print translate_exception(e.errors['0'][0])
  expected a list


Why use Voluptuous over another validation library?
---------------------------------------------------
**Validators are simple callables**
  No need to subclass anything, just use a function.

**Errors are simple exceptions.**
  A validator can just ``raise VALUEERROR(msg)`` and expect the user to get useful
  messages.

**Schemas are basic Python data structures.**
  Should your data be a dictionary of integer keys to strings?  ``{int: str}``
  does what you expect. List of integers, floats or strings? ``[int, float, str]``.

**Designed from the ground up for validating more than just forms.**
  Nested data structures are treated in the same way as any other type. Need a
  list of dictionaries? ``[{}]``

**Consistency.**
  Types in the schema are checked as types. Values are compared as values.
  Callables are called to validate. Simple.

Other libraries and inspirations
--------------------------------
Voluptuous is heavily inspired by `Validino
<http://code.google.com/p/validino/>`_, and to a lesser extent, `jsonvalidator
<http://code.google.com/p/jsonvalidator/>`_ and `json_schema
<http://blog.sendapatch.se/category/json_schema.html>`_.

I greatly prefer the light-weight style promoted by these libraries to the
complexity of libraries like FormEncode.
