from .pathnode import _PathNode

import inspect
NOTHING = inspect._empty
from inspect import (
    _POSITIONAL_ONLY,           # 0
    _POSITIONAL_OR_KEYWORD,     # 1
    _VAR_POSITIONAL,            # 2
    _KEYWORD_ONLY,              # 3
    _VAR_KEYWORD,               # 4
)

from itertools import zip_longest, islice, chain
from collections.abc import Iterable
import pandas as pd
import numpy as np
import bisect

from attrs import define, field
import attrs

# cli-app with: loop._run_live()
from rich.progress import Progress, SpinnerColumn
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.live import Live
from rich.theme import Theme

SOLARIZED = {
    'base03':   '#002b36',
    'base02':   '#073642',
    'base01':   '#586e75',
    'base00':   '#657b83',
    'base0':    '#839496',
    'base1':    '#93a1a1',
    'base2':    '#eee8d5',
    'base3':    '#fdf6e3',
    'yellow':   '#b58900',
    'orange':   '#cb4b16',
    'red':      '#dc322f',
    'magenta':  '#d33682',
    'violet':   '#6c71c4',
    'blue':     '#268bd2',
    'cyan':     '#2aa198',
    'green':    '#859900',
}

mytheme = Theme(SOLARIZED | {
    'repr.ipv6':            f'none',
    'repr.attrib_name':     f'not italic {SOLARIZED["orange"]}',
    'repr.attrib_value':    f'none',
    'repr.number':          f'none',  # SOLARIZED['magenta'],  # magenta
    'repr.call':            SOLARIZED['blue'],
    #~ 'repr.bool_false':     'italic bright_red',
    #~ 'repr.bool_true':      'italic bright_green',
    'repr.bool_false':      f'italic bold {SOLARIZED["red"]}',
    'repr.bool_true':       f'italic bold {SOLARIZED["green"]}',
    'repr.str':             f'not bold not italic {SOLARIZED["green"]}',
    'prompt.default':       f'bold {SOLARIZED["cyan"]}',
    'current':              f'bold {SOLARIZED["blue"]}',
})

console = Console(theme=mytheme, log_path=False)


"""
Design decissions:

[1] Every named path-attribute is an instance of a separate (sub-) class!

[2] This subclass contains its name and node-configuration-data within
    the sub-class!

[3] The instance (of the class-attribute) only holds the path-parent instance!

        ValueNode
          |
          | is sub-classed by
          |
          V
        MyValueNode._value = 321
          |
          | is instanciated for every new path-node
          |
          V
        my_value._parent = base_node_in_path

"""

class _ValueNode(_PathNode):
    _value = NOTHING

    def __init__(self, value=NOTHING):
        # save arguments in order to be used for sub-class creation
        #   then `_value` become attribute of the sub-class
        if value is not NOTHING:
            self._value = value

    _lazy = True
    def _eval_value(self, dm=None):
        return self._value

    def _get_value(self, dm=None, idx=-1):
        return self._value

    def _eval(self, dm):
        return self._value

    def _get_attr(self):
        return self

    def _iter_children(self, dm=None):
        yield from ()

    def _new_subcls_obj(self, fdct, parent_name, name, level):
        """Convert instance configuration into attributes of a new sub-class"""
        ### the attribute name is saved in the subcls-name
        subcls = type(name, (self.__class__,), {})
        subcls.__qualname__ = f'{parent_name}.{name}'
        ### specific to type of node
        if '_value' in self.__dict__:
            subcls._value = self._value
        ### create new sub-classed object
        new_obj = subcls.__new__(subcls)
        return name, new_obj

    def _repr(self, as_string=True):
        if as_string:
            return repr(self._value)
        else:
            return self._value


class RefNode(_PathNode):
    _other = NOTHING
    _level = 2
    _attr_cached = None
    # todo: track chained pointers for end-user transparency
    # todo optimization: shorten pointer-name, enlarge root-name
    _idx = -1

    def __init__(self, obj, level=None, idx=-1):
        self._other = obj
        if level is not None:
            self._level = level
        if idx != -1:
            self._idx = idx

    def _get_value(self, dm=None, idx=None):
        attr = self._get_attr()
        if idx is None:
            _idx = self._idx
        else:
            _idx = idx
            if self._idx != _idx:
                msg = f'WARNING: {self._idx = } but {idx = } requested'
                print(msg)
        return  attr._get_value(dm, _idx)

    def _eval(self, dm=None):
        attr = self._get_attr()
        value = attr._eval(dm)
        return value

    def _get_attr(self):
        if attr := self._attr_cached:
            return attr
        obj = self._root_of_other()
        for name in self._other:
            obj = getattr(obj, name)
        attr = self._attr_cached = obj._get_attr()
        return attr

    def _iter_children(self, dm=None):
        attr = self._get_attr()
        yield from attr._iter_children(dm)

    def _root_of_other(self):
        return self._get_parent(self._level)

    def _new_subcls_obj(self, fdct, parent_name, name, level):
        """Convert instance configuration into attributes of a new sub-class"""
        ### the attribute name is saved in the subcls-name
        subcls = type(name, (self.__class__,), {})
        subcls.__qualname__ = f'{parent_name}.{name}'
        ### specific to type of node
        # replace the `_other` object
        if isinstance(self._other, _PathNode):
            # logging
            #~ print(f'{" " * 4*level }RefNode found:  '
                  #~ f'_other = {self._other._pathname!r}  '
                  #~ f'_level = {level}'
            #~ )
            subcls._other = tuple(node._name for node in self._other._path)
        elif isinstance(self._other, str):
            subcls._other = tuple(self._other.split('.'))
        else:
            subcls._other = self._other
        _level = _level = self.__dict__.get('_level', level)
        if _level != RefNode._level:
            subcls._level = _level
        _idx = self.__dict__.get('_idx', RefNode._idx)
        if _idx != RefNode._idx:
            subcls._idx = _idx
        ### create new sub-classed object
        new_obj = subcls.__new__(subcls)
        return name, new_obj

    def _repr(self, as_string=True):
        cls = self.__class__
        msg = self._get_attr()._pathname
        idx = self._idx
        if idx == slice(None):
            msg += f'[:]'
        elif idx != -1:
            msg += f'[{repr(idx)}]'
        return msg


class SelfRefNode(RefNode):
    def __init__(self, level):
        self._level = level
        self._other = ()

    def _get_value(self, dm=None, idx=None):
        value = self._root_of_other()
        return value

    def _eval(self, dm=None):
        value = self._root_of_other()
        return value

    def _get_attr(self):
        # return that object which has the right _get_value() result
        return self


class _TupleNode(_PathNode):
    _values = NOTHING

    _lazy = True

    def __init__(self, *args):
        # save arguments in order to be used for sub-class creation
        #   then `_value` become attribute of the sub-class
        self._values = args

    def _new_subcls_obj(self, fdct, parent_name, name, level):
        """Convert instance configuration into attributes of a new sub-class"""
        ### the attribute name is saved in the subcls-name
        subcls = type(name, (self.__class__,), {})
        subcls.__qualname__ = f'{parent_name}.{name}'
        new_parent_name = subcls.__qualname__
        ### specific to type of node
        for idx, value in enumerate(self._values):
            key = f'_{idx}'
            _, new_attr = fdct._filter_node(parent_name, key, value, level + 1)
            setattr(subcls, key, new_attr)
        subcls._num = len(self._values)
        ### create new sub-classed object
        new_obj = subcls.__new__(subcls)
        return name, new_obj

    def _get_attr(self):
        # needed for RefNode
        return self

    def _get_value(self, dm=None, idx=-1):
        return tuple(item._get_value(dm, idx) for item in self)

    def _eval(self, dm):
        return tuple(item._eval(dm) for item in self)

    def _eval_value(self, dm):
        # it's never called, just a dummy for the '_eval_value'-interface
        # the tuple-value is build in _iter_children
        assert 0
        return dm.read(self)

    def _iter_children(self, dm=None):
        values = []
        for idx in range(self._num):
            attr = getattr(self, f'_{idx}')
            value = yield attr._get_attr()
            values.append(value)
        return tuple(values)

    def __iter__(self):
        for idx in range(self._num):
            yield getattr(self, f'_{idx}')._get_attr()

    def __len__(self):
        return self._num

    def __getitem__(self, idx):
        return getattr(self, f'_{idx}')

    @classmethod
    def _node_attrs(cls, skip_cls=None, filter_func=None):
        return {(k := f'_{idx}'): getattr(cls, k) for idx in range(cls._num)}

    def _repr(self, as_string=True):
        items = [n._repr(as_string) for n in self]
        return f'({", ".join(items)})'


class StateNode(_PathNode):
    """ Same like _ValueNode with addition to:
d
            * get/set via dm
            * next-node as ConditionalNode
                - next-function w/ condition => ._cond-dict
                - next-function w/o condition => ._default
                    ._default-function _defaults_ to lambda state: state

                    Preventing dummy-state-updates is the resposibility
                    of the higher-level Loop-Manager (Nested, Concat, ...)

        Update-logic in separate (LooPy-) managers, like Nested, Concat, Zip, ...
    """
    #~ init = Input(0)  # but I dont want all the inputs meta-data
    #~ next = ConditionalNode()

    def __init__(self, init=0):
        self._init = init

    def _new_subcls_obj(self, fdct, parent_name, name, level):
        """Convert instance configuration into attributes of a new sub-class"""
        ### the attribute name is saved in the subcls-name
        subcls = type(name, (self.__class__,), {})
        subcls.__qualname__ = f'{parent_name}.{name}'
        ### specific to type of node
        if not fdct.get('_statename', ''):
            fdct['_statename'] = name
        else:
            msg = (f'one state with name {fdct["_statename"]!r} already exists, '
                   f'new state with name {name!r} can not be accepted'
                   f' (for multiple states use tuple or subsystems)')
            raise ValueError(msg)
        key = 'init'
        _, init_node = fdct._filter_node(parent_name, key, self._init, level + 1)
        subcls.init = init_node
        ### create new sub-classed object
        new_obj = subcls.__new__(subcls)
        return name, new_obj

    def _get_attr(self):
        return self

    def _eval(self, dm):
        # todo: StateNode feels like SystemNode with _init = __return__
        #   and custom _iter_children()
        if self not in dm:
            value = self.init._eval(dm)
            dm.write(self, value)
        return dm.read(self)

    def _get_value(self, dm, idx=-1):
        if self not in dm:
            value = self.init._eval(dm)
            dm.write(self, value)
        return dm.read(self, idx=idx)

    def _update(self):
        """Update dm-value of this state with the next-node value"""
        raise NotImplementedError

    def _iter_children(self, dm=None):
        yield from ()

    _lazy = True  # eval (func-) node only if new inputs available
    def _eval_value(self, dm):
        """Writes init-value into dm only once because of empty children."""
        # logging
        #~ print(f'StateNode._eval_value:  return _init before writing into dm')
        return self.init._eval(dm)

    def _repr(self, as_string=True):
        cls = self.__class__
        try:
            cname = cls.__bases__[0].__name__
        except KeyError:
            cname = cls.__name__
        return f'{cname}(init={self.init._repr(as_string)})'


class InputRangeError(Exception):
    def __init__(self, min, value, max, pathname=''):
        self.min = min
        self.max = max
        self.value = value
        self.message = f'{min} <= {value} <= {max} ({pathname})'
        super().__init__(self.message)


@attrs.define
class Input(_PathNode):
    """ Design decision:
            If an input should be behave like all other variables
            which can be referenced.
            Then it can NOT be a classical input-descriptor!

        # test stepwise system configuration
            a = A1(y=555)
            a.x = 444
            assert a._arguments == {'x': 444, 'y': 555}

        This is not possible because __prepare__ transforms the attrs immediately.
        Then __init_subclasses__ must be used again. But here the transformation
        is a bit more complex... I would like to postpone it.

        Some prototype ideas for that:

        * Disable attr-setting except for inputs

        * For Input-Class use:

            def __set__(self, obj, value):
                obj._arguments[self.name] = value

            def __get__(self, obj, cls=None):
                try:
                    return obj._arguments[self.name]
                except KeyError:
                    return self.default
    """
    default = field(default=NOTHING)
    min = field(default=None)
    max = field(default=None)
    unit = field(default='')
    fmt = field(default='')
    kind = field(default=_POSITIONAL_OR_KEYWORD)

    def _validate(self, value, pathname=''):
        _min, _max = self.min, self.max
        if _min is not None:
            if not (_min <= value):
                raise InputRangeError(_min, value, _max, pathname)
        if _max is not None and value is not None:
            if not (value <= _max):
                raise InputRangeError(_min, value, _max, pathname)

    def _new_subcls_obj(self, fdct, parent_name, name, level):
        """Convert instance configuration into attributes of a new sub-class"""
        ### the attribute name is saved in the subcls-name
        #       an extra new subcls is not necessary because
        #       the default value is return as a named node (new_obj)
        ### specific to type of node
        # validate unique input names
        if name in fdct._inputs:
            msg = f'duplicate input name {name!r}'
            raise TypeError(msg)
        # validate correct kind of input
        if fdct._inputs:
            last_inp_name = tuple(fdct._inputs)[-1]
            last_inp = fdct._inputs[last_inp_name]
            last_inp_kind = last_inp.kind
        else:
            last_inp_kind = -1
        if last_inp_kind in (_VAR_POSITIONAL, _VAR_KEYWORD):
            if not (self.kind > last_inp_kind):
                msg = (f'kind of input {name!r} must be '
                       f'greater than {self.kind}')
                raise TypeError(msg)
        else:
            if not (self.kind >= last_inp_kind):
                msg = (f'kind of input {name!r} must be '
                       f'greater or equal than {self.kind}')
                raise TypeError(msg)
        # get (and filter) default value
        default = self.default
        if self.kind is _VAR_POSITIONAL:
            if default is NOTHING:
                default = self.default = _TupleNode()
            else:
                raise ValueError(f'{self.kind} input {name!r} '
                                  'can not have a default value')
        elif self.kind is _VAR_KEYWORD:
            if default is NOTHING:
                default = self.default = {}
            else:
                raise ValueError(f'{self.kind} input {name!r} '
                                  'can not have a default value')
        # validate correct order of input defaults
        if fdct._inputs and self.kind == _POSITIONAL_OR_KEYWORD:
            # self is at least the second Input
            if default is NOTHING and last_inp.default is not NOTHING:
                msg = f'non-default input follows default input'
                raise TypeError(msg)
        # count min/max number of positional input arguments
        nmin, nmax = fdct.get('_inp_num_pos', (0, 0))
        if self.kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD):
            nmax += 1
            if default is NOTHING:
                nmin += 1
        elif self.kind == _VAR_POSITIONAL:
            nmax = np.inf
        fdct['_inp_num_pos'] = (nmin, nmax)
        # todo/release: test subsystems if _inp_num_pos is set correctly!
        #   it is important but nice-to-have because
        #   this bug concerns 'inlined'-functions in TaskSequence
        #   which is not a focus for the next release!
        # use default-value as class-attribute
        name, new_obj = fdct._filter_node(parent_name, name, default, level)
        if hasattr(new_obj, '_auto_inputs'):
            new_obj._auto_inputs(self, fdct, parent_name, level)
        fdct._inputs[name] = self
        ### create new sub-classed object
        #       not necessary
        return name, new_obj


@attrs.define
class Output(_PathNode):
    func = field(default=lambda nom: nom)
    min  = field(default=-np.inf, kw_only=True)
    ltl  = field(default=None, kw_only=True)
    nom  = field(default=None, kw_only=True)
    utl  = field(default=None, kw_only=True)
    max  = field(default=np.inf, kw_only=True)
    unit = field(default='', kw_only=True)
    fmt  = field(default='', kw_only=True)

    def __call__(self, func):
        self.func = func
        return self

    def __attrs_post_init__(self):
        self._validate()

    def _validate(self):
        names = ('min', 'ltl', 'nom', 'utl', 'max')
        for nlo, nup in zip(names[0:], names[1:]):
            vlo, vup = getattr(self, nlo), getattr(self, nup)
            try:
                if not (vlo <= vup):
                    msg = f'({nlo} = {vlo}) <= ({nup} = {vup})'
                    raise ValueError(msg)
            except TypeError:
                pass

    def _new_subcls_obj(self, fdct, parent_name, name, level):
        """Convert instance configuration into attributes of a new sub-class"""
        ### the attribute name is saved in the subcls-name
        #       an extra new subcls is not necessary because
        #       the default value is return as a named node (new_obj)
        ### specific to type of node
        # validate unique output names
        if name in fdct._outputs:
            new_obj = fdct[name]
            attr = fdct._outputs[name]
            for field in attrs.fields(self.__class__):
                key = field.name
                val = getattr(self, key)
                if val != field.default:
                    setattr(attr, key, val)
            subcls = new_obj.__class__
        else:
            attr = self
            fdct._outputs[name] = self
            # create new SystemNode() with attr-function as __return__
            bases = (OutputNode,)
            subcls = type(name, bases, {})
            subcls.__qualname__ = f'{parent_name}.{name}'
            new_obj = subcls.__new__(subcls)
        parent_name = subcls.__qualname__
        ### overwrite __return__ function ###
        subcls.__return__ = attr.func
        ### convert field into nodes ###
        if 0:
            kwargs = {k: v for k, v in attrs.asdict(attr).items() if k != 'func'}
            new_obj = subcls(**kwargs)
            name, new_obj = new_obj._new_subcls_obj(fdct, parent_name, name, level)
        for key, val in attrs.asdict(attr).items():
            if key == 'func':
                continue
            key, val = fdct._filter_node(parent_name, key, val, level + 1)
            setattr(subcls, key, val)
        ### set return function ###
        # create Input-attributes for all function arguments
        _inputs = subcls._inputs
        _reserved = set(attrs.asdict(attr))
        _return_params = _ReturnFunction()
        for pname, pvalue, param in _return_params.iter_params(attr.func):
            pname, new_attr = fdct._filter_node(parent_name, pname, pvalue, level + 1)
            if not hasattr(subcls, pname):
                setattr(subcls, pname, new_attr)
            _inputs[pname] = Input(pvalue, kind=param.kind)
            if pname not in _reserved and pvalue is NOTHING:
                fdct._auto_connect.setdefault(pname, []).append(subcls)
        _return_params.to_attrs(subcls)
        return name, new_obj


class _ReturnFunction:
    def iter_params(self, func):
        self._arg_names = []
        self._arg_var_name = ''
        self._kw_names = []
        self._kw_var_name = ''
        self.__return__ = func
        sig = inspect.signature(func)
        params = iter(sig.parameters.items())
        for pname, param in params:
            match param.kind:
                case inspect.Parameter.KEYWORD_ONLY:
                    self._kw_names.append(pname)
                case inspect.Parameter.VAR_KEYWORD:
                    self._kw_var_name = pname
                case inspect.Parameter.VAR_POSITIONAL:
                    self._arg_var_name = pname
                case _:
                    self._arg_names.append(pname)
            yield pname, param.default, param

    def to_attrs(self, obj):
        fname = self.__return__.__name__
        obj.__return__ = self.__return__
        obj._arg_names = tuple(self._arg_names)
        obj._arg_var_name = self._arg_var_name
        obj._kw_names = tuple(self._kw_names)
        obj._kw_var_name = self._kw_var_name

    def add_annotations_to_inputs(self, obj):
        input_annotations = {}
        func_annotations = inspect.get_annotations(self.__return__)
        for name in obj._inputs.keys() & func_annotations.keys():
            name_annotations = input_annotations.setdefault(name, {})
            name_annotations[()] = func_annotations[name]
        obj._input_annotations = input_annotations


class MetaFilter(type):
    def __prepare__(cls_name, bases, **kwargs):
        # logging
        #~ print(f'MetaFilter.__prepare__({cls_name!r}, {bases}, {kwargs=})')
        namespace = _FilterDict(cls_name, bases)
        return namespace

    def __new__(mcs, cls_name, bases, dct):
        # logging
        #~ print(f'MetaFilter.__new__({mcs}, {cls_name!r}, {bases}, dct=...)')
        #~ print(f'    {dct = }')
        new_cls = type.__new__(mcs, cls_name, bases, dict(dct))
        #~ new_cls.__module__ = __name__
        # type.__new__() calls bases[0].__init_subclass__(new_cls)
        if isinstance(dct, _FilterDict):
            dct._finalize(new_cls)
        # logging
        #~ print(f'    {new_cls = }')
        return new_cls


class _FilterDict(dict):
    def __init__(self, cls_name, bases=(), level_init=1):
        self._cls_name = cls_name
        self._cls_bases = bases
        self._level_init = level_init
        self._finished = set()
        self._inputs = {}  # {inp_name: Input-obj}
        self._outputs = {}  # {out_name: Output-obj}
        self._input_annotations = {}  # {inp_name: {annotated_subcls: obj} }
        self._auto_connect = {}  # {inp_name: [sub-func, ...]}

    def __repr__(self):
        msg = f'<{self._cls_name}: level={self._level_init}, {super().__repr__()}>'
        return msg

    def __setitem__(self, key, value):
        #~ print(f'{self._cls_name}:  SET {key} = {value!r}')
        level = self._level_init
        if not key.startswith('_'):
            # logging
            #~ print(f'{self._cls_name}:  SET {key} = {value!r}')
            # this is the first level of _filter_node() calls
            # further recursions are done by SystemNode._from_draft()
            parent_name = self['__qualname__']
            key, value = self._filter_node(parent_name, key, value, level)
            # logging
            #~ print(f'{self._cls_name}:  SET {key} = {value!r}')
        dict.__setitem__(self, key, value)

    def _filter_node(self, parent_name, name, attr, level):
        """Return new_obj which is an instance of a new sub-class

        The name of subcls is set to the attributes name
        used for path construction.
        """
        if not (isinstance(attr, _PathNode) or inspect.isfunction(attr)):
            subcls = type(name, (_ValueNode,), {'__module__': __name__})
            subcls.__qualname__ = f'{parent_name}.{subcls.__qualname__}'
            subcls._value = attr
            new_obj = subcls.__new__(subcls)
            # logging
            #~ print(f'{" " * 4*level }no-node-obj => {new_obj!r}')

        elif isinstance(attr, _PathNode) and attr._root.__class__ in self._finished:
            # logging
            #~ msg = (f"RESOLVING namespace CONFLICT with "
                   #~ f"'{name} = RefNode({attr._pathname})'")
            #~ print(f'{" " * 4*level }{msg}')
            name, new_obj = RefNode(attr)._new_subcls_obj(self, parent_name, name, level)
            # this is the same problem like
            #
            #   def myfunc(): return 321
            #   myfunc()  # 321
            #   myfunc.__name__  # 'myfunc'
            #
            #   g = myfunc
            #   g()  # 321
            #   g.__name__  # 'myfunc'

        elif isinstance(attr, (_ValueNode, _TupleNode,
                               StateNode, RefNode, Input, Output, SystemNode)):
            name, new_obj = attr._new_subcls_obj(self, parent_name, name, level)

        elif inspect.isfunction(attr):
            if attr.__name__ == '__return__':
                value = RefNode(())
                name, new_obj = self._filter_node(parent_name, name, value, level)
                return name, new_obj
            # create new SystemNode() with attr-function as __return__
            #   todo: auto-connect together with Input-class
            # logging
            #~ print(f'{" " * 4*level }new SystemNode for {name}-function:')
            if ('Task' in [c.__name__ for c in self._cls_bases]
                and (name.startswith('task') or name.startswith('test'))
            ):
                bases = (Task,)
                subcls = type(name, bases, {})
                #~ if name.startswith('task'):
                    #~ subcls._lazy = False
            else:
                bases = (SystemNode,)
                subcls = type(name, bases, {})
            subcls.__qualname__ = f'{parent_name}.{name}'
            parent_name = subcls.__qualname__
            # create Input-attributes for all function arguments
            _inputs = {}  # todo: is this really necessary or overwritten later?
            _return_params = _ReturnFunction()
            for pname, pvalue, param in _return_params.iter_params(attr):
                pname, new_attr = self._filter_node(parent_name, pname, pvalue, level + 1)
                setattr(subcls, pname, new_attr)
                _inputs[pname] = Input(pvalue, kind=param.kind)
                if pvalue is NOTHING:
                    self._auto_connect.setdefault(pname, []).append(subcls)
            _return_params.to_attrs(subcls)
            subcls._inputs = _inputs
            _return_params.add_annotations_to_inputs(subcls)
            # remember annotations from subcls
            _input_annotations = self._input_annotations
            for a_name, anno_dict in subcls._input_annotations.items():
                for key, anno in anno_dict.items():
                    path = (name,) + key
                    _input_annotations.setdefault(a_name, {})[path] = anno
            # logging
            #~ print(f'    create new subclass-object')
            new_obj = subcls.__new__(subcls)

        self._finished.add(new_obj.__class__)
        return name, new_obj

    def _finalize(self, new_cls):
        if self._inputs:
            # do not overwrite the inputs of base-classes
            # todo: more testing
            _inputs = dict(self._inputs)
            if new_cls._inputs:
                # validate if base-class only has _KEYWORD_ONLY inputs
                # and skip overwritten input-attributes
                for name, inp in new_cls._inputs.items():
                    if name in self:
                        continue # input of base-class is overwritten
                    elif inp.kind is not _KEYWORD_ONLY:
                        msg = f'{name!r} input from base class must be KEYWORD_ONLY'
                        raise ValueError(msg)
                    else:
                        _inputs[name] = inp
            new_cls._inputs = _inputs
        #~ if statename := self.get('_statename', ''):
            #~ new_cls._statename = getattr(new_cls, statename)
        self._apply_auto_connect(new_cls)
        self._extract_return_args(new_cls)
        # collect annotations from all subsystems
        _input_annotations = {}
        for name in self._input_annotations.keys() & new_cls.__dict__.keys():
            anno_dict = self._input_annotations[name]
            _input_annotations.setdefault(name, {}).update(anno_dict)
        if _input_annotations:
            new_cls._input_annotations = _input_annotations
        if self._outputs:
            new_cls._outputs = dict(self._outputs)
            ### check for non-equal configured output-functions ###
            no_func, has_func = [], []
            default_func = attrs.fields(Output).func.default
            for name, op in self._outputs.items():
                if op.func is default_func:
                    no_func.append(name)
                else:
                    has_func.append(name)
            if no_func and has_func:
                has_func = ', '.join(f'{name!r}' for name in has_func)
                no_func = ', '.join(f'{name!r}' for name in no_func)
                print(f'WARNING: mixed outputs in {new_cls.__name__!r}')
                print(f'    outputs with function: {has_func}')
                print(f'    outputs w/o  function: {no_func}')


    def _apply_auto_connect(self, new_cls):
        # logging
        #~ print(f'    AUTO-CONNECT:')
        parent_name = new_cls.__qualname__
        level = self._level_init + 1
        for name, classes in self._auto_connect.items():
            if hasattr(new_cls, name):
                other = getattr(new_cls, name)
                refnode = RefNode(other)
            elif name == 'self':
                refnode = SelfRefNode(level=2)
            else:
                msg = f'{name!r} not in namespace of {self._cls_name}: ' \
                      f'unknown argument values for ' \
                      f'{", ".join([c.__qualname__ for c in classes])}'
                raise ValueError(msg)
            name, new_obj = self._filter_node(parent_name, name, refnode, level)
            for func_cls in classes:
                setattr(func_cls, name, new_obj)
                _other_repr = f'{".".join(new_obj._other)!r}'
                _level_repr = f'level={new_obj._level}'
                refnode_repr = f'RefNode({_other_repr}, {_level_repr})'
                # logging
                #~ print(f'        {func_cls.__qualname__}.{name} = {refnode_repr}')

    def _extract_return_args(self, new_cls):
        if '__return__' in self and self['__return__'] is not None:
            rfunc = self['__return__']
        else:
            return
        # logging
        #~ print(f'    extract arguments from systems __return__ function')
        parent_name = new_cls.__qualname__
        _return_params = _ReturnFunction()
        for pname, pvalue, param in _return_params.iter_params(rfunc):
            if pname == 'self':
                pvalue = SelfRefNode(level=1)
            if pvalue is not NOTHING:
                if not hasattr(new_cls, pname):
                    # logging
                    #~ print(f'Return-Function {rfunc}: set {pname!r} = {pvalue!r}')
                    level = self._level_init
                    pname, new_attr = self._filter_node(parent_name, pname, pvalue, level)
                    setattr(new_cls, pname, new_attr)
                else:
                    msg = f'can not overwrite {pname!r}, it is already an attribute'
                    raise ValueError(msg)
                    # Some api-ideas for this case in ...
                    #
                    #   __return__ could be a pointer to a other (sub-) SystemNode
                    #
                    #   But this would violate the design decission that every
                    #   SystemNode has exactly one return-FUNCTION
                    #
                    #   On th other hand, that is also a function.
                    #   But with a differnt argument-namespace, which is handled
                    #   by the other (sub-) system, isn't it?
                    #
                    #   Maybe somthing like
                    #
                    #       def _get_return_node(self):
                    #           return self
        _return_params.to_attrs(new_cls)
        _return_params.add_annotations_to_inputs(new_cls)
        # remember annotations from new_cls
        _input_annotations = self._input_annotations
        for a_name, anno_dict in new_cls._input_annotations.items():
            for key, anno in anno_dict.items():
                path = key  # (name,) + key
                _input_annotations.setdefault(a_name, {})[path] = anno


class SystemNode(_PathNode, metaclass=MetaFilter):
    _arguments = NOTHING
    _inputs = {}  # {inp_name: Input-obj}
    _input_annotations = {}  # {inp_name: {annotated_subcls: obj} }
    _inp_num_pos = (0, 0)    # min/max number of positional input arguments
    _outputs = {}  # {out_name: Output-obj}
    _statename = ''  # name of state-attribute
    # (for multiple states use tuple or subsystems)

    # argument types of __return__ function
    _arg_var_name = ''  # VAR_POSITIONAL
    _kw_names = []      # KEYWORD_ONLY
    _kw_var_name = ''   # VAR_KEYWORD
    _arg_names = []     # POSITIONAL_OR_KEYWORD and POSITIONAL_ONLY

    _lazy = True  # eval (func-) node only if new inputs available

    def __init__(self, *args, **kwargs):
        self._arguments = {}
        if args:
            zipped = zip(args, self._inputs.items())
            for idx, (value, (name, inp)) in enumerate(zipped):
                if inp.kind == _VAR_POSITIONAL:
                    values = args[idx:]
                    self._arguments[name] = values
                    break
                self._arguments[name] = value
        if kwargs:
            keys = list(kwargs.keys())
            _arguments = {}
            for name in tuple(self._inputs)[len(self._arguments):]:
                kind = self._inputs[name].kind
                if kind in (1, 3) and name in kwargs:
                    _arguments[name] = kwargs.pop(name)
                elif kind == 4 and kwargs:
                    _arguments[name] = kwargs
                    keys.append(name)
                    kwargs = {}
                    break
            if kwargs:
                msg = f'{self.__class__.__name__}() got unexpected ' \
                      f'keyword argument {list(kwargs.keys())[0]!r}'
                raise TypeError(msg)
            self._arguments |= {k: _arguments[k] for k in keys if k in _arguments}
        self._validate_pos_args(len(args), self._arguments)

    @classmethod
    def _validate_pos_args(cls, nargs, arguments):
        nmin, nmax = cls._inp_num_pos
        if nargs > nmax:
            msg = f'{cls.__name__}() takes '
            msg += str(nmin) if nmin == nmax else f'from {nmin} to {nmax}'
            msg += f' positional arguments but {nargs} '
            msg += 'were given' if nargs > 1 else 'was given'
            raise TypeError(msg)
        missing = [n for n in tuple(cls._inputs)[:nmin] if n not in arguments]
        if missing:
            ndiff = len(missing)
            msg = f'{cls.__name__}() missing {ndiff} required positional '
            msg += 'arguments: ' if ndiff > 1 else 'argument: '
            msg += ', '.join(repr(n) for n in missing)
            raise TypeError(msg)
        missing = []
        for name, inp in cls._inputs.items():
            if inp.default is NOTHING and name not in arguments:
                missing.append(name)
        if missing:
            ndiff = len(missing)
            msg = f'{cls.__name__}() missing {ndiff} required keyword-only '
            msg += 'arguments: ' if ndiff > 1 else 'argument: '
            msg += ', '.join(repr(n) for n in missing)
            raise TypeError(msg)

    def _new_subcls_obj(self, fdct, parent_name, name, level):
        """Convert instance configuration into attributes of a new sub-class"""
        ### the attribute name is saved in the subcls-name
        subcls = type(name, (self.__class__,), {})
        subcls.__qualname__ = f'{parent_name}.{subcls.__qualname__}'
        ### specific to type of node
        # logging
        #~ print(f'{" " * 4*level }SystemNode found:')
        #~ print(f'{" " * 4*level }arguments: {self._arguments!r}')
        parent_name = subcls.__qualname__
        for key, val in self._arguments.items():
            inp = self._inputs[key]
            if inp.kind is _VAR_POSITIONAL:
                val = _TupleNode(*val)
            key, new_attr = fdct._filter_node(parent_name, key, val, level + 1)
            if hasattr(new_attr, '_auto_inputs'):
                new_attr._auto_inputs(inp, fdct, parent_name, level + 1)
            setattr(subcls, key, new_attr)
        # remember annotations from subcls
        _input_annotations = fdct._input_annotations
        for a_name, anno_dict in subcls._input_annotations.items():
            for key, anno in anno_dict.items():
                path = (name,) + key
                _input_annotations.setdefault(a_name, {})[path] = anno
        ### create new sub-classed object
        new_obj = subcls.__new__(subcls)
        return name, new_obj

    def _apply_configuration(self, dm=None):
        """ Apply argument configuration to __dict__.

        This tries to solve the problem

            loopy = LooPy()
            a = A(1, 2)
            a._apply_configuration()
            loopy.eval(a)

        by avoiding an extra sub-class which name is unknown at this point.
        """
        root = self._root
        if root._arguments is NOTHING:
            return self
        cls = root.__class__
        dct = _FilterDict(cls.__name__)
        dct['__qualname__'] = cls.__qualname__
        for key, val in root._arguments.items():
            kind = root._inputs[key].kind
            if kind is _VAR_POSITIONAL:
                dct[key] = _TupleNode(*val)
            else:
                dct[key] = val
            new_attr = dct[key]
            new_attr._parent = self
            root.__dict__[key] = new_attr
        return self

    def __return__():
        """Return the output value of the system, defaults to None.

        The __return__ function behaves like staticmethod since this function
        is not called as a bound instance method but instead as an unbound
        class attribute

            self.__class__.__return__(*args, **kwargs)

        If the first argument is named 'self' then the right object node
        is injected in 'args'.

        Multiple outputs can be either tupled into the return value
        or separate sub-systems.

        The default values of keyword arguments are converted to general
        system attributes.

            class MyQuad(SystemNode):
                def __return__(x=10):
                    return x**2

        is equivalent to

            class MyQuad(SystemNode):
                x = 10
                def __return__(x):
                    return x**2
        """
        return None

    def _get_value(self, dm=None, idx=-1):
        if dm:
            value = dm.read(self, idx=idx)
        else:
            value = f'{self._pathname}.__return__'
        return value

    def _get_attr(self):
        return self

    def _iter_return_args(self, dm=None):
        for name in self._arg_names:
            attr = getattr(self, name)
            yield attr._get_value(dm)
        if name := self._arg_var_name:
            attr = getattr(self, name)
            yield from attr._get_value(dm)

    def _iter_return_kwargs(self, dm=None):
        for name in self._kw_names:
            attr = getattr(self, name)
            yield name, attr._get_value(dm)
        if name := self._kw_var_name:
            attr = getattr(self, name)
            yield from attr._get_value(dm).items()

    def _eval_value(self, dm=None):
        #~ if self.__return__ is None: return
        args = tuple(self._iter_return_args(dm))
        kwargs = dict(self._iter_return_kwargs(dm))
        value = self.__class__.__return__(*args, **kwargs)
        return value

    def _eval(self, dm):
        for node, value in self._iter_dfs_nodes(dm):
            if value is None:
                value = node._eval_value(dm)
            dm.write(node, value)
        return dm.read(self)

    def _iter_dfs_nodes(self, dm, respect_lazy=True):
        """Iterate nodes needs to be (re-) evaluated using depth-first-search."""
        # Based on copy from dfs_labeled_edges(G, source=None, depth_limit=None)
        # in networkx/algorithms/traversal/depth_first_search.py
        # Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
        # by D. Eppstein, July 2004.
        start = self
        #~ depth_limit = len(config.__nodes__) + 1
        depth_limit = 100
        needs_eval = set()
        visited = set()
        visited.add(start)
        #~ yield start, start, "forward"
        last_child = [None]
        stack = [(start, depth_limit, last_child, start._iter_children(dm))]
        # logging
        #~ print(f'START node: {start}')
        while stack:
            parent, depth_now, last_child, children = stack[-1]
            # logging
            #~ print(f'{parent = }')
            try:
                node = last_child[0]
                retval = node._get_value(dm) if node else None
                child = children.send(retval)
                last_child[0] = child
                if not hasattr(child, '_eval_value'):
                    continue
                    # this prevents node from appending to stack
                    # because this node has just a constant config-value
                # logging
                #~ print(f'    {child = }')
                #~ print(f'    {dm.last_write_cnt(child), dm.last_write_cnt(parent)}')
                if dm.last_write_cnt(child) > dm.last_write_cnt(parent):
                    # logging
                    #~ print(f'        is newer than parent')
                    needs_eval.add(parent)
                if child not in visited:
                    #~ yield parent, child, "forward"
                    visited.add(child)
                    if depth_now > 1:
                        last_child = [None]
                        item = (child, depth_now - 1, last_child,
                                child._iter_children(dm))
                        stack.append(item)
                        #~ print(f'    append {child} with children: {list(child._iter_children(dm))}')
                #~ else:
                    #~ yield parent, child, "nontree"
                    #~ print(f'    was visited:  {child=}')
            except StopIteration as e:
                parent_retval = e.value
                # logging
                #~ print(f'    has NO further children')
                stack.pop()
                if stack:
                    #~ yield stack[-1][0], parent, "reverse"
                    grandpar = stack[-1][0]
                    if (parent in needs_eval
                        or (parent not in dm)
                        or (not parent._lazy and respect_lazy)
                    ):
                        # logging
                        #~ print(f'    needs EVAL, append to OUTPUT')
                        yield parent, parent_retval
                        needs_eval.add(grandpar)
                        # logging
                        #~ print(f'    needs_eval.add: {grandpar = }')
                    #~ last_child = parent
                    # todo: if optimization is needed,
                    #   then this avoids the last_child[0] memory,
                    #   but explicit is better then implicit
        #~ yield start, start, "reverse"
        # logging
        #~ print(f'{start = }')
        assert start == parent  # it should be!
        if (start in needs_eval
            or (start not in dm)
            or (not start._lazy and respect_lazy)
        ):
            # logging
            #~ print(f'    needs EVAL, append to OUTPUT')
            yield start, parent_retval

    def _iter_arguments(self):
        """Iterate over the argument objects of __return__"""
        for name in self._arg_names:
            attr = getattr(self, name)
            yield name, attr._get_value()
        if name:= self._arg_var_name:
            attr = getattr(self, name)
            yield name, attr._get_value()
        yield from self._iter_return_kwargs()

    def _iter_return_arguments(self):
        for name in self._arg_names:
            yield getattr(self, name)
        if name:= self._arg_var_name:
            yield getattr(self, name)
        for name in self._kw_names:
            yield getattr(self, name)
        if name := self._kw_var_name:
            attr = getattr(self, name)

    def _iter_children(self, dm=None):
        """Iterate over children (= subsystems) of return-function

        Currently `_arg_var_name` and `_kw_var_name` are a _ValueNode,
        e.g. a tuple and a dict.

        todo: is flatten _arg_var_name and _kw_var_name possible?

            This looks like a 1:n reference, isn't it?

            Maybe an extra sub-system collecting all *args and returning
            a tuple of them?
        """
        for name in self._arg_names:
            attr = getattr(self, name)
            attr = attr._get_attr()
            value = yield attr
            # args.append(value)
        if name:= self._arg_var_name:
            attr = getattr(self, name)
            values = yield attr
            # args += values
        for name in self._kw_names:
            attr = getattr(self, name)
            attr = attr._get_attr()
            value = yield attr
            # kwargs[name] = value

    def _iter_return_args(self, dm=None):
        for name in self._arg_names:
            attr = getattr(self, name)
            yield attr._get_value(dm)
        if name := self._arg_var_name:
            attr = getattr(self, name)
            yield from attr._get_value(dm)

    @classmethod
    def _get_loopnames(cls):
        loopnames = {}
        for name, node in cls._node_attrs(skip_cls=Task).items():
            if hasattr(node, '__mainloop__'):
                key = node.loop_level._get_value()
                if key is None:
                    key = name
                loopnames.setdefault(key, []).append(name)
        return tuple(loopnames.values())

    @classmethod
    def _repr(cls, as_string=True):
        _inputs = cls._inputs
        args = []
        for name, node in cls.__dict__.items():
            if name in _inputs:
                value = node._repr(as_string) if hasattr(node, '_repr') else node
                arg = f'{name}={value}'
                args.append(arg)
        try:
            cname = cls.__bases__[0].__name__
        except KeyError:
            cname = cls.__name__
        return f'{cname}({", ".join(args)})'

    def _read_input_config(self):
        cls = self.__class__
        input_names = []
        input_configs = []
        for name in cls._inputs:
            if name in ('loop_level', 'result', 'loops_cached'):
                # todo: remove this somehow
                continue
            input_names.append(f'.{name} = ')
            input_configs.append(getattr(self, name)._repr(as_string=False))
        task_name = f'{self._pathname}'
        data = {
            task_name: input_names,
            '(VALUES)': input_configs,
            '': [''] * len(input_names),
        }
        df = pd.DataFrame(data)
        #~ df = pd.DataFrame(data).set_index(task_name)
        df = df.set_index('')
        return df

    def __getitem__(self, idx):
        return Squeeze(self)


class OutputNode(SystemNode):
    min  = Input()
    ltl  = Input()
    nom  = Input()
    utl  = Input()
    max  = Input()
    unit = Input()
    fmt  = Input()

    def __return__(nom):
        return nom

    def check(min, ltl, utl, max, value=RefNode( () )):
        if value is None:
            return None  # pd.NA
        if (min <= value <= max):
            if ltl is None and utl is None:
                return None  # pd.NA
            if ltl is None:
                return bool(value <= utl)
            elif utl is None:
                return bool(ltl <= value)
            else:
                return bool(ltl <= value  <= utl)
        else:
            return None  # pd.NA


class Sum(SystemNode):
    args = Input(kind=_VAR_POSITIONAL)

    def _iter_children(self, dm=None):
        self._result = 0
        for attr in self.args:
            attr = attr._get_attr()
            value = yield attr
            self._result += value
        return self._result


# in python 3.12: itertools.batched
def batched(iterable, n, tuple_list=tuple):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple_list(islice(it, n)):
        yield batch


def reshape(a, shape):
    """
    >>> a = range(2 * 3 * 4)
    >>> reshape(a, [2, 3, 4]))
    [[[[0, 1, 2, 3],
       [4, 5, 6, 7],
       [8, 9, 10, 11]],

      [[12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]]]]

    for more:
    https://stackoverflow.com/questions/56121903/how-to-reshape-a-list-without-numpy
    """
    for dim in reversed(shape):
        a = list(batched(a, dim, list))
        #~ a = tuple(zip(*[iter(a)] * dim))
        #~ break
        #~ a = list(list(sublist) for sublist in zip(*[iter(a)] * dim))
    return a


def flatten(a, shape):
    for _ in shape:
        a = list(chain.from_iterable(a))
    return a


class ConditionalNode(SystemNode):
    args = Input(kind=_VAR_POSITIONAL)
    default = Input(0, kind=_KEYWORD_ONLY)

    def _iter_children(self, dm=None):
        for cond, conseq in batched(self.args, 2):
            cond = cond._get_attr()
            value = yield cond
            if value:
                conseq = conseq._get_attr()
                result = yield conseq
                return result
        default = self.default._get_attr()
        result = yield default
        return result


class Pointer(SystemNode):
    other = Input()  # path-names relative to PinterNode - level
    level = Input(2)

    def root(self, level):
        # from RefNode._root_of_other
        for idx, node in enumerate(self._path_to_root()):
            if idx == level:
                break
        return node

    def _iter_children(self, dm=None):
        other_path = yield self.other._get_attr()
        if isinstance(other_path, str):
            other_path = tuple(other_path.split('.'))
        # validate type of other: tuple(str, ...)
        try:
            if not all(isinstance(name, str) for name in other_path):
                raise TypeError
        except TypeError:
            msg = f'type of other_path is not tuple(str, ...): {other_path = }'
            raise ValueError(msg)
        root = yield self.root  # of other
        other_node = root._get_subnode(other_path)
        value = yield other_node
        return value


class _LoopSystem(SystemNode):
    loop_level = Input(None, kind=_KEYWORD_ONLY)
    # use only constants, like loop_level=0 or loop_level='inner'

    result = Input(RefNode( () ), kind=_KEYWORD_ONLY)
    # loop-result references to _LoopSystem-obj.__return__()

    def __mainloop__(self):
        return self

    def _is_valid(self, dm=None):
        return self.is_valid._eval(dm)

    def is_valid():
        return True

    def _has_next(self, dm=None):
        return self.has_next._eval(dm)

    def has_next():
        return True

    def _is_interrupted(self, dm=None):
        return self.is_interrupted._eval(dm)

    def is_interrupted():
        return False

    def _next_updates(self, dm=None):
        # Iterating over sets of functions updating the system-states
        # Each function-set is evaluated quasi-parallel in order to
        #   decouple the order of evluation
        yield {self.next}

    def _update(self, dm):
        # todo: It this general enough for moving into SystemNode?
        for nodes in self._next_updates(dm):
            for node in nodes:
                node._eval(dm)
            for node in nodes:
                parent_node = node._parent
                state_node = getattr(parent_node, parent_node._statename)
                value = node._get_value(dm)
                dm.write(state_node, value)

    def _restart_updates(self, dm=None):
        # same like _next_updates but with restart-functions
        yield {self.restart}

    def _iter_result_node(self, dm):
        """yields loop_result-node which must be evaluated"""
        # needs: self._is_interrupted(), _has_next(), _update(), _is_valid()
        loop = self.__mainloop__()
        loop_result = loop.result._get_attr()
        if loop._is_interrupted(dm) and loop._has_next(dm):
            loop._update(dm)
        while loop._is_valid(dm):
            nodes = loop_result._iter_dfs_nodes(dm, respect_lazy=False)
            if not next(nodes, None):
                if not loop._is_interrupted(dm) and loop._has_next(dm):
                    loop._update(dm)
                else:
                    break
            yield loop_result

    def _run(self, dm, **kwargs):
        loop = self.__mainloop__()
        loop_result = loop.result._get_attr()
        _, result_values = dm._data.get(loop_result._key, ([], []))
        pos_start = len(result_values)
        for node in self._iter_result_node(dm):
            node._eval(dm)
        _, result_values = dm._data.get(loop_result._key, ([], []))
        return result_values[pos_start:]


class CountingLoop(_LoopSystem):
    num = Input(0)

    idx = StateNode(0)

    def next(idx):
        return idx + 1

    def restart():
        return 0  # todo/release/nice-to-have:  return idx._init

    def __return__(idx):
        return idx
        # todo: __return__ = idx
        #   this could be an expression-function!!!
        #   maybe that would eliminate RefNode?!

    length = num

    def is_valid(idx, length):
        return 0 <= idx < length

    def has_next(idx, length):
        return idx < (length - 1)

    def min_max(num):
        return (0, num)  # min=idx._init


class NestedLoop(_LoopSystem):
    #~ loops = Input(kind=_VAR_POSITIONAL)
    loops_cached = Input( (()) , kind=_KEYWORD_ONLY)

    # todo: this should be in the result-node!
    #   needs new __return__-generator with dm-protocol
    def _iter_children(self, dm=None):
        loops = yield self.loops_cached._get_attr()
        result = []
        for loops_zipped in loops:
            values_zipped = []
            for loop in loops_zipped:
                value = yield loop.result._get_attr()
                values_zipped.append(value)
            if len(values_zipped) > 1:
                result.append(tuple(values_zipped))
            else:
                result += values_zipped
        return tuple(result)

    def _is_valid(self, dm):
        loops = self.loops_cached._eval(dm)
        values_zipped = []
        for loops_zipped in loops:
            value = all(loop._get_attr()._is_valid(dm) for loop in loops_zipped)
            values_zipped.append(value)
        return any(values_zipped)

    def _has_next(self, dm):
        loops = self.loops_cached._eval(dm)
        values_zipped = []
        for loops_zipped in loops:
            value = all(loop._get_attr()._has_next(dm) for loop in loops_zipped)
            values_zipped.append(value)
        return any(values_zipped)

    def _is_interrupted(self, dm):
        loops = self.loops_cached._eval(dm)
        values_zipped = []
        for loops_zipped in loops:
            value = any(loop._get_attr()._is_interrupted(dm) for loop in loops_zipped)
            values_zipped.append(value)
        return all(values_zipped)

    def _next_updates(self, dm):
        loops = self.loops_cached._eval(dm)
        # logging
        #~ print(f'######## [1.] {loops = }')
        for n in range(len(loops) - 1, -1, -1):
            loops_zipped = [loop._get_attr() for loop in loops[n]]
            # logging
            #~ print(f'######## {[loop._pathname for loop in loops_zipped]}')
            if (not any(loop._is_interrupted(dm) for loop in loops_zipped)
                and all(loop._has_next(dm) for loop in loops_zipped)
            ):
                break
        if all(loop._has_next(dm) for loop in loops_zipped):
            for loop in loops_zipped:
                # logging
                #~ print(f'######## {n=}  next_update:  {loop}')
                yield from loop._next_updates(dm)

        # refresh loops due to possible new task
        loops = self.loops_cached._eval(dm)
        # logging
        #~ print(f'######## [2.] {loops = }')
        #~ print(f'######## {len(loops) = }')
        #~ print(f'######## {     n + 1 = }')
        n += 1
        while n < len(loops):
            loops_zipped = [loop._get_attr() for loop in loops[n]]
            # todo/release/nice-to-have: try to move this
            #   into _LoopSystem._next_updates()
            #   in order to have only one _next_updates()
            #   which handles restarts automatically
            #   and only stops due to loopy.run()
            if (any(loop._is_interrupted(dm) for loop in loops_zipped)
                and all(loop._has_next(dm) for loop in loops_zipped)
            ):
                for loop in loops_zipped:
                    # logging
                    #~ print(f'######## next_update:  {loop}')
                    yield from loop._next_updates(dm)
            else:
                for loop in loops_zipped:
                    # logging
                    #~ print(f'######## RESTART_update:  {loop}')
                    yield from loop._restart_updates(dm)
                # refresh loops due to possible new task
                loops = self.loops_cached._eval(dm)
                # logging
                #~ print(f'######## [3.] {loops = }')
            n += 1
        # logging
        #~ print(f'    {self.result._eval(dm) = }')


    def _restart_updates(self, dm=None):
        loops = self.loops_cached._eval(dm)
        # logging
        #~ print(f'    RESTART: {self=}')
        #~ print(f'    ##### {loops = }')
        #~ print(f'    {self.result._eval(dm) = }')
        n = 0
        while n < len(loops):
            loops_zipped = loops[n]
            for loop in loops_zipped:
                yield from loop._restart_updates(dm)
            loops = self.loops_cached._eval(dm)
            n += 1


class _MinMax(SystemNode):
    def _iter_children(self, dm=None):
        # https://stackoverflow.com/a/8459309/5599281
        node = self._parent
        items = iter(node.items)
        item = next(items, None)
        if item is None:
            return (None, None)
        minimum = maximum = yield item  # todo/idea/new-dm:  None == yield None
        #~ import ipdb; ipdb.set_trace()
        for item in items:
            #~ import ipdb; ipdb.set_trace()
            if hasattr(item, 'min_max'):
                min_value, max_value = yield item.min_max
                if min_value < minimum:
                    minimum = min_value
                if max_value > maximum:
                    maximum = max_value
            else:
                value = yield item
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
        return (minimum, maximum)


class ConcatLoop(NestedLoop):
    items = Input(kind=_VAR_POSITIONAL)

    def num(items):
        return len(items)

    idx = CountingLoop(num)

    def is_valid(num):
        return num > 0

    def loops_cached(self, idx):
        loops = [(self.idx,)]
        item = self.items[idx]
        try:
            loop = item._get_attr().__mainloop__()
            loops.append( (loop,) )  # no zipped loops
        except AttributeError:
            pass
        return tuple(loops)

    def _iter_children(self, dm=None):
        items = yield self.items
        idx = yield self.idx.idx
        item = items[idx]
        if hasattr(item, '_eval_value'):
            result = yield item
        else:
            result = item
        return result

    min_max = _MinMax()


class _VertIndex(_LoopSystem):
    sections = Input()
    sidx = Input()

    def vnums(sections):
        vnums = []
        for sec in sections:
            vnums.append( len(sec['v_paths']) )
        return vnums

    idx = StateNode(init=RefNode('restart'))
    def restart(vnums):
        return [0] * len(vnums)

    def next(idx, sidx):
        idx = list(idx)
        idx[sidx] += 1
        return idx

    def __return__(idx, sidx):
        return idx[sidx]

    def is_valid(idx, sidx, vnums):
        return 0 <= idx[sidx] < vnums[sidx]

    def has_next(idx, sidx, vnums):
        return idx[sidx] < (vnums[sidx] - 1)


class _HLoopsVPath(SystemNode):
    def _iter_children(self, dm):
        node = self._parent
        npath = ()
        hloops = []
        while hasattr(node, 'sidx'):
            #~ hloops.append( (node.sidx,) )
            hloops.append( (npath + ('sidx',),) )
            sections = yield node.sections
            sidx = yield node.sidx
            section = sections[sidx]
            for names_zipped in section.get('h_loops', []):
                loops_zipped = []
                for name in names_zipped:
                    #~ loop = node._get_subnode( (name,) ).__mainloop__()
                    loop = npath + (name,)
                    loops_zipped.append(loop)
                hloops.append( tuple(loops_zipped) )
            #~ hloops.append( (node.vidx,) )
            hloops.append( (npath + ('vidx',),) )
            vidx = yield node.vidx
            vpath = section['v_paths'][vidx]
            if vpath:
                node = node._get_subnode(vpath)
                npath += vpath
            else:
                break
        return hloops, npath


class _Result(SystemNode):
    h_loops = Input()
    vpath = Input()

    def _iter_children(self, dm=None):
        # logging
        #~ print(f'### Result: {self._pathname}  ###')
        # eval all h_loop-results
        h_loops = yield self.h_loops._get_attr()  # = task.mloops
        for loops_zipped in h_loops:
            for loop in loops_zipped:
                node = loop.result._get_attr()
                # logging
                #~ print(f'###     eval: {node._key}')
                yield node
        vpath = yield self.vpath._get_attr()
        node = self._get_parent(level=1)._get_subnode(vpath)._get_attr()
        # logging
        #~ print(f'###     yield {node._key}  ==>  {node._get_attr()._key}')
        _lazy, node._lazy = node._lazy, False
        value = yield node._get_attr()
        node._lazy = _lazy
        # logging
        #~ print(f'###     {value = }')
        return value


class Task(NestedLoop):
    def _check_input_limits(self, dm, skip_cls=None):
        cls = self.__class__
        if skip_cls is None:
            skip_cls = Task
        filter_func = lambda k, v: hasattr(v, '_inputs')
        paths = self._iter_namespace_paths(skip_cls=skip_cls, filter_func=filter_func)
        #~ import ipdb; ipdb.set_trace()
        for spath in reversed(list(paths)):
            snode = cls._get_subnode(cls, spath)
            #~ print(spath)
            inputs = snode._inputs
            names = inputs.keys()
            #~ import ipdb; ipdb.set_trace()
            for name in names:
                inp = inputs[name]
                #~ import ipdb; ipdb.set_trace()
                if inp.min is None and inp.max is None:
                    continue
                #~ print(f'    {name}:  {inp.min = }  {inp.max = }')
                #~ inp_node = getattr(snode, name)
                path = spath + (name,)
                inp_node = self._get_subnode(path)
                pathname = '.'.join((self._name,) + path)
                if hasattr(inp_node, 'min_max'):
                    min_value, max_value = inp_node.min_max._eval(dm)
                    #~ print(f'        {min_value = }')
                    inp._validate(min_value, pathname)
                    #~ print(f'        {max_value = }')
                    inp._validate(max_value, pathname)
                else:
                    value = inp_node._eval(dm)
                    #~ print(f'        {value = }')
                    inp._validate(value, pathname)

    def sections(self):
        section_names = [['setup'], ['task', 'test'], ['teardown'], ['final']]
        cls = self.__class__
        cls_namespace = cls._node_attrs(skip_cls=Task)
        sections = {}
        task_h_loops = {}
        for name, attr in cls_namespace.items():
            for section in section_names:
                if (any(name.startswith(n) for n in section)
                    and isinstance(attr, (SystemNode, Task))
                ):
                    sec = sections.setdefault(section[0], {})
                    sec.setdefault('v_paths', []).append( (name,) )
                    break
            else:
                # for-loop didn't break: (name, attr) is not a section
                if hasattr(attr, '__mainloop__') and 'task' not in sections:
                    key = attr.loop_level._get_value()
                    if key is None:
                        key = name
                    task_h_loops.setdefault(key, []).append(name)

        for name in self._outputs:
            if 'task' in sections:
                sec = sections.setdefault('final', {})
                sec.setdefault('v_paths', []).append( (name,) )
                sec.setdefault('v_paths', []).append( (name, 'check') )
            else:
                sec = sections.setdefault('task', {})
                sec.setdefault('v_paths', []).append( (name,) )
                sec.setdefault('v_paths', []).append( (name, 'check') )

        # if Task.__return__ definded ...
        # ... then decide to which sections it should be appended
        _base_classes = cls.__mro__[-1 - len(Task.__mro__) ::-1]
        cls_has_return = any(('__return__' in c.__dict__) for c in _base_classes)
        if cls_has_return and 'task' in sections:
            sec = sections.setdefault('final', {})
            sec.setdefault('v_paths', []).append( () )
        elif cls_has_return or not sections:
            sec = sections.setdefault('task', {})
            sec.setdefault('v_paths', []).append( () )
        # task_h_loops: convert zipped-names into tuples
        sections['task']['h_loops'] = [tuple(_) for _ in task_h_loops.values()]
        #~ return [sections[name] for name in section_names if name in sections]
        section_list = []
        for name, *_ in section_names:
            if name in sections:
                s = sections[name]
                s['debug_name'] = name
                section_list.append(s)
        return section_list

    def snum(sections):
        return len(sections)

    sidx = CountingLoop(snum)

    vidx = _VertIndex(sections, sidx)

    hloops_vpath = _HLoopsVPath()

    def loops_cached(self, hloops_vpath):
        hloops = []
        for paths_zipped in hloops_vpath[0]:
            loops_zipped = []
            for path in paths_zipped:
                loop = self._get_subnode(path).__mainloop__()
                loops_zipped.append(loop)
            hloops.append( tuple(loops_zipped) )
        return hloops

    def vpath(hloops_vpath):
        _, vpath = hloops_vpath
        return vpath

    result = _Result(loops_cached, vpath)

    def _squeeze(self, *args, **kwargs):
        return Squeeze(self, *args, **kwargs)

    # must overwrite/clear/reset/undo the _iter_children in NestedLoop
    _iter_children = _LoopSystem._iter_children

    def _run(self, dm, logging=False):
        task_result = self._get_attr()
        _, result_values = dm._data.get(task_result._key, ([], []))
        pos_start = len(result_values)
        for node in self._iter_result_node(dm):
            if logging:
                hloop_paths, current_path = dm.read(self.hloops_vpath)
                current = '.'.join(current_path) if current_path else self._pathname
                hloops = []
                for loops_zipped in hloop_paths:
                    for loop_path in loops_zipped:
                        name = loop_path[-1]
                        if any(name.startswith(_) for _ in ('sidx', 'vidx')):
                            continue
                        lpnode = self._get_subnode(loop_path)
                        lpvalue = lpnode._eval(dm)
                        loop_name = '.'.join(loop_path)
                        hloops.append(f'{loop_name}={lpvalue!r}')
                text = f'[current]{current}[/current]'
                if hloops:
                    text += f' | {", ".join(hloops)}'
                now = pd.Timestamp.now().strftime('%H:%M:%S')
                console.print(f'[not bold dim cyan][{now}][/]  {text}')
            retval = node._eval(dm)
            if logging and retval is not None:
                now = pd.Timestamp.now().strftime('%H:%M:%S')
                console.print(f'[dim cyan][{now}][/]      = {retval!r}')
        if not logging:
            _, result_values = dm._data.get(task_result._key, ([], []))
            retvals = result_values[pos_start:]
            if len(retvals) > 1:
                return retvals
            elif len(retvals) == 1:
                return retvals[0]

    def _called_steps(self, dm):
        cnts, vals = dm._data[self.vpath._key]
        steps = []
        for path in vals:
            pathname = '.'.join(path) if path else '__return__'
            steps.append(pathname)
        return steps

    def _run_live(self, dm):
        sections = self.sections._eval(dm)
        tname = self._name
        main_panel, progress_sections = self._create_progresses(sections, tname)
        ### start looping
        sv_idx = (None, None)
        with Live(main_panel, console=console, refresh_per_second=12, auto_refresh=True) as _live:
            for node in self._iter_result_node(dm):
                sidx = self.sidx._eval(dm)
                vidx = self.vidx._eval(dm)
                if (sidx, vidx) != sv_idx:
                    # stop old vnode
                    if sv_idx != (None, None):
                        progress.update(id, running=running, completed=100, passed=passed)
                        progress.stop_task(id)
                    # start new vnode
                    progress, vids = progress_sections[sidx]
                    id = vids[vidx]
                    progress.reset(id, start=False, running='')
                    progress.start_task(id)
                    progress.update(id, running='[bold]')
                    sv_idx = sidx, vidx
                    _live.update(_live.renderable, refresh=True)
                # eval starts
                retval = node._eval(dm)
                # eval ends
                if isinstance(retval, (bool, np.bool_)) and retval == True:
                    passed = True
                    running = '[green]'
                elif isinstance(retval, (bool, np.bool_)) and retval == False:
                    passed = False
                    running = '[red]'
                else:
                    passed = None
                    running = ''
            progress.update(id, running=running, completed=100, passed=passed)
            progress.stop_task(id)
            _live.update(_live.renderable, refresh=True)

    def _create_progresses(self, sections, rootname='MainTask', spinner=0):
        spinner_name = ['point', 'circleQuarters'][spinner]
        progress_sections = []
        lines = []
        for section in sections:
            progress = Progress(
                MySpinnerColumn(spinner_name),
                "{task.fields[running]}{task.description}",
                #~ TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            vids = []
            for vpath in section['v_paths']:
                #~ vname = '.'.join(vpath) if vpath else rootname
                vname = '.'.join(vpath) if vpath else '__return__'
                id = progress.add_task(vname)
                progress.reset(id, start=False, running='')
                vids.append(id)
            progress_sections.append( (progress, vids) )
            lines.append(progress)
            lines.append(Rule(style='blue'))
        group = Group(*lines[:-1])
        main_panel = Panel.fit(group, title=rootname,
            border_style='blue', padding=(0, 1)
        )
        return main_panel, progress_sections


class MySpinnerColumn(SpinnerColumn):
    def render(self, task):
        _frames = self.spinner.frames
        width = len(_frames[0]) if _frames else 1
        if not task.started:
            text = ' ' * width
        elif task.finished:
            passed = task.fields.get('passed', None)
            if passed:
                text = Text.from_markup('[green]:heavy_check_mark:')
            elif passed is None:
                text = Text.from_markup(':heavy_check_mark:')
            else:
                text = Text.from_markup('[red]:heavy_check_mark:')
            gap = width - len(text)
            left = gap // 2
            right = left + (gap % 2)
            text = Text.assemble(' ' * left, text, ' ' * right)
        else:
            text = self.spinner.render(task.get_time())
        return text


def _commonpath(*paths):
    """Return the first sub-items of path which are equal in other

    Example:
        >>> _commonpath(['a', 'b', 'c', 'd'], ['a', 'b', 'c'])
        ['a', 'b', 'c'], (['d'], []))
        >>> _commonpath(['a', 'b', 'c', 'd'], ['a', 'b', 'fail'])
        ['a', 'b'],  (['c', 'd'], ['fail']))
    """
    idx = -1
    for idx, items in enumerate(zip(*paths)):
        if len(set(items)) > 1:
            break
    else:
        idx += 1
    return paths[0][:idx], [path[idx:] for path in paths]


def _is_parentpath(path, other):
    """Test if other is (root-) part of path.

    Example:
        >>> _commonpath(['a', 'b', 'c', 'd'], ['a', 'b', 'c'])
        True
        >>> _commonpath(['a'], ['a', 'b', 'fail'])
        False
        >>> _commonpath(['a', 'b', 'c', 'd'], ['a', 'b', 'fail'])
        False
    """
    if len(path) < len(other):
        return False
    for idx, (s, o) in enumerate(zip(path, other)):
        if s != o:
            return False
    return True


class Squeeze(SystemNode):
    dnode = Input()  # data-node
    as_array = Input(True)
    flatten = Input(True, kind=_KEYWORD_ONLY)

    def _iter_children(self, dm):
        dnode = self.dnode._get_attr()
        yield dnode  # just for needs_eval otherwise dm.write() is not called
        cnt = dm.last_write_cnt(dnode)
        cmd_name = dm.read('__cmd__', cnt).partition('(')[2].rpartition(')')[0]
        cmd_path = tuple(cmd_name.split('.'))[1:]
        cmd_node = self._root._get_subnode(cmd_path)
        hloop_paths, _ = dm.read(cmd_node.hloops_vpath, cnt)

        dnode_path = tuple(n._name for n in dnode._path[1:])
        self_path = tuple(n._name for n in self._path[1:])
        _parent_path, (self_short, dnode_short) = _commonpath(self_path, dnode_path)
        _, (__, parent_path) = _commonpath(cmd_path, _parent_path)

        #~ import ipdb; ipdb.set_trace()
        # get lowest outer-hloop
        # test, if self_short and dnode_short are in the same section
        sections = yield cmd_node._get_subnode(parent_path).sections
        for section in sections:
            v_paths = section['v_paths']
            if self_short[:1] in v_paths and dnode_short[:1] in v_paths:
                _inner_path = dnode_path
                _inner_path = parent_path + dnode_short[:1]

                # nice idea to get the outer_loop from 'static' sections
                # but again recursive nesting is needed!
                #~ outer_loop = section.get('h_loops', [])
                #~ outer_loop = [parent_path + loop for loop in outer_loop]
                break
        else:  # loop was not broken
            _inner_path = parent_path
        # get (relevant) hloops as path-tuples
        hloops = []
        idx_inner = None
        for loops_zipped in hloop_paths:
            # todo: respect zipped loops!
            for loop_path in loops_zipped:
                name = loop_path[-1]
                if any(name.startswith(_) for _ in ('sidx', 'vidx')):
                    continue
                #~ import ipdb; ipdb.set_trace()
                if idx_inner is None and _is_parentpath(loop_path, _inner_path):
                    idx_inner = len(hloops)
                hloops.append(loop_path)
        if idx_inner is None:
            idx_inner = len(hloops)
        inner_hloops = hloops[idx_inner:]
        outer_hloops = hloops[:idx_inner]
        # read all values of dnode[last_cnt_of inner_-_outer_loop:]
        if outer_hloops:
            oloop = cmd_node._get_subnode(outer_hloops[-1])
            last_cnt = dm.last_write_cnt(oloop)
        else:
            last_cnt = -1
        cnts, values = dm._data[dnode._key]
        # search smallest idx with cnts[idx] >= last_cnt
        idx_right = bisect.bisect_left(cnts, last_cnt)
        inner_values = values[idx_right:]
        #~ import ipdb; ipdb.set_trace()
        if (yield self.as_array):
            return np.asarray(inner_values)
        else:
            return inner_values
        # ...
        shape = ()
        for loop in inner_hloops:
            key = cmd_node._get_subnode(loop)._key
            cnts, _ = dm._data[key]
            num = len(cnts) // np.prod(shape, dtype=int)
            shape += (num,)
        cnts = np.array(cnts).reshape(shape)
        dcnts, dvals = dm._data[dnode._key]
        _values = []
        for cnt in cnts[(-1,) * (idx_inner - 1)].flatten():
            idx = bisect.bisect_left(dcnts, cnt)
            _values.append(dvals[idx])
        _values = np.array(_values)
        _values.resize(shape[idx_inner:] + _values.shape[1:])
        #~ import ipdb; ipdb.set_trace()
        return _values

    def _repr(self, as_string=True):
        cls = self.__class__
        try:
            cname = cls.__bases__[0].__name__
        except KeyError:
            cname = cls.__name__
        return f'{cname}({self.dnode._repr(as_string)})'


def _squeeze(dnode, dm):
    # a copy of Squeeze class needed for dm.run()
    self = dnode._root
    cnt = dm.last_write_cnt(dnode)
    cmd_name = dm.read('__cmd__', cnt).partition('(')[2].rpartition(')')[0]
    cmd_path = tuple(cmd_name.split('.'))[1:]
    cmd_node = self._root._get_subnode(cmd_path)
    hloop_paths, _ = dm.read(cmd_node.hloops_vpath, cnt)

    dnode_path = tuple(n._name for n in dnode._path[1:])
    self_path = tuple(n._name for n in self._path[1:])
    _parent_path, (self_short, dnode_short) = _commonpath(self_path, dnode_path)
    _, (__, parent_path) = _commonpath(cmd_path, _parent_path)

    #~ import ipdb; ipdb.set_trace()
    # get lowest outer-hloop
    # test, if self_short and dnode_short are in the same section
    sections = dm.read(cmd_node._get_subnode(parent_path).sections)
    for section in sections:
        v_paths = section['v_paths']
        if self_short[:1] in v_paths and dnode_short[:1] in v_paths:
            _inner_path = dnode_path
            # nice idea to get the outer_loop from 'static' sections
            # but again recursive nesting is needed!
            #~ outer_loop = section.get('h_loops', [])
            #~ outer_loop = [parent_path + loop for loop in outer_loop]
            break
    else:  # loop was not broken
        _inner_path = parent_path
    # get (relevant) hloops as path-tuples
    hloops = []
    idx_inner = None
    for loops_zipped in hloop_paths:
        # todo: respect zipped loops!
        for loop_path in loops_zipped:
            name = loop_path[-1]
            if any(name.startswith(_) for _ in ('sidx', 'vidx')):
                continue
            #~ import ipdb; ipdb.set_trace()
            if idx_inner is None and _is_parentpath(loop_path, _inner_path):
                idx_inner = len(hloops)
            hloops.append(loop_path)
    if idx_inner is None:
        idx_inner = len(hloops)
    inner_hloops = hloops[idx_inner:]
    outer_hloops = hloops[:idx_inner]
    # read all values of dnode[last_cnt_of inner_-_outer_loop:]
    if outer_hloops:
        oloop = cmd_node._get_subnode(outer_hloops[-1])
        last_cnt = dm.last_write_cnt(oloop)
    else:
        last_cnt = -1
    cnts, values = dm._data[dnode._key]
    # search smallest idx with cnts[idx] >= last_cnt
    idx_right = bisect.bisect_left(cnts, last_cnt)
    inner_values = values[idx_right:]
    #~ import ipdb; ipdb.set_trace()
    return inner_values


class LoopLin(CountingLoop):
    start = Input(None)
    stop  = Input(None)
    step  = Input(1)
    num   = Input(None)

    def num_step(start, stop, step, num):
        delta = stop - start
        if num is None:
            _step = abs(step) if delta > 0 else -abs(step)
            _num = int(delta / _step) + 1
        else:
            div = num - 1
            div = div if div > 1 else 1
            _step = float(delta) / div
            _num = num
        return _num, _step

    def __return__(idx, start, num_step):
        num, step = num_step
        return start + step * idx

    def length(num_step):
        num, _ = num_step
        return num

    def min_max(start, stop):
        return min(start, stop), max(start, stop)

    @classmethod
    def _auto_inputs(cls, inp, fdct, pname, level):
        try:
            if cls.start._value in {NOTHING, None}:
                _, cls.start = fdct._filter_node(pname, 'start', inp.min, level)
            if cls.stop._value in {NOTHING, None}:
                _, cls.stop = fdct._filter_node(pname, 'stop', inp.max, level)
        except AttributeError:
            pass


class LoopLog(CountingLoop):
    """Sweeps logarithmically from start to stop within num steps.

    >>> SweepLog(1, 1000, num=4).as_list()
    [1.0, 10.0, 100.0, 1000.0]

    >>> SweepLog(1, 64, num=7, base=2).as_list()
    [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    """
    start = Input(None)
    stop  = Input(None)
    num   = Input(None)
    base  = Input(10)

    def log_start(start, base):
        if base == 10:
            return np.log10(start)
        elif base == 2:
            return np.log2(start)
        else:
            # this general term is numerically less precise
            return np.log(start) / np.log(base)

    def log_stop(stop, base):
        if base == 10:
            return np.log10(stop)
        elif base == 2:
            return np.log2(stop)
        else:
            # this general term is numerically less precise
            return np.log(stop) / np.log(base)

    exponent = LoopLin(log_start, log_stop, num=num, result=RefNode( () ))

    def __return__(base, exponent):
        return np.power(base, exponent)

    def length(num_step=exponent.num_step):
        num, _ = num_step
        return num

    def min_max(base, start, stop):
        minimum = min(start, stop)
        maximum = max(start, stop)
        return minimum, maximum

    def __mainloop__(self):
        return self.exponent

    @classmethod
    def _auto_inputs(cls, inp, fdct, pname, level):
        try:
            if cls.start._value in {NOTHING, None}:
                _, cls.start = fdct._filter_node(pname, 'start', inp.min, level)
            if cls.stop._value in {NOTHING, None}:
                _, cls.stop = fdct._filter_node(pname, 'stop', inp.max, level)
        except AttributeError:
            pass


class LoopBisect(NestedLoop):
    start = Input(None)
    stop = Input(None)
    cycles = Input(3)

    n = CountingLoop(cycles)

    def start_n(start, stop, n):
        delta = abs(stop - start)
        offs = 0 if n in {0} else delta / 2**n
        return start + offs

    def step_n(start, stop, n):
        delta = abs(stop - start)
        step = delta if n in {0, 1} else delta / 2**(n-1)
        return step

    result = LoopLin(start_n, stop, step_n)

    def loops_cached(self):
        return ((self.n,), (self.result,))

    def min_max(start, stop):
        return min(start, stop), max(start, stop)

    @classmethod
    def _auto_inputs(cls, inp, fdct, pname, level):
        try:
            if cls.start._value in {NOTHING, None}:
                _, cls.start = fdct._filter_node(pname, 'start', inp.min, level)
            if cls.stop._value in {NOTHING, None}:
                _, cls.stop = fdct._filter_node(pname, 'stop', inp.max, level)
        except AttributeError:
            pass
