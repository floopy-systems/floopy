class _PathNode:
    _parent = None

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        name = self.__class__.__name__
        try:
            inst = obj.__dict__[name]
        except KeyError:
            _cls = self.__class__
            inst = _cls.__new__(_cls)
            inst._parent = obj
            obj.__dict__[name] = inst
            # logging
            #~ print(f'*** {obj}  creates {name!r} from {_cls}')
        return inst

    def __set__(self, obj, value):
        name = self.__class__.__name__
        print(f'attribute {name!r} '
              f'from object {obj!r} '
              f'should be set to {value!r}')

    def _path_to_root(self):
        """Iterate path from self back to root."""
        obj = self
        while obj is not None:
            yield obj
            obj = obj._parent

    @property
    def _root(self):
        for obj in self._path_to_root(): pass
        return obj

    @property
    def _path(self):
        # should be cached?
        return tuple(self._path_to_root())[::-1]

    @property
    def _pathname(self):
        return '.'.join(node._name for node in self._path)

    # return key for data-manager (this method is not needed/tested here)
    _key = _pathname

    @property
    def _name(self):
        name = self.__class__.__name__
        return name

    def _get_parent(self, level=1):
        for idx, node in enumerate(self._path_to_root()):
            if idx == level:
                break
        return node

    def _commonpath(self, other):
        common = ()
        for s, o in zip(self._path, other._path):
            if s is o:
                common = s
        return common

    def _get_subnode(self, path):
        # validate type of path: tuple(str, ...)
        try:
            if not (isinstance(path, tuple)
                    and all(isinstance(name, str) for name in path)
            ):
                raise TypeError
        except TypeError:
            msg = f'type of other_path is not tuple(str, ...): {path = }'
            raise ValueError(msg)
        node = self
        for name in path:
            node = getattr(node, name)
        return node

    def _get_namedpath(self, parent=None):
        # todo: needs testing
        if parent is None:
            return tuple(obj._name for obj in self._path)
        common = ()
        for idx, (s, p) in enumerate(zip(self._path, parent._path)):
            if s is p:
                common = s
            elif common:
                break
            else:
                msg = f'{parent._pathname} is not related with self={self._pathname}'
                raise ValueError(msg)
        return tuple(obj._name for obj in self._path[idx:])

    @property
    def _namedpath(self):
        return self._get_namedpath()

    def _iter_pathnames(self):
        for name, attr in self._node_attrs().items():
            attr = getattr(self, name)
            yield attr._pathname
            yield from iter(attr._iter_pathnames())

    def _iter_namespace(self, basename=None, skip_cls=None, filter_func=None):
        """Iterate over all subsequent names in namespace of this node.

        equal to _iter_pathnames() without creation of instances
        """
        if basename is None:
            basename = self._pathname
        for name, attr in self._node_attrs(skip_cls, filter_func).items():
            qualname = f'{basename}.{name}' if basename else name
            yield qualname
            try:
                yield from attr._iter_namespace(qualname, skip_cls, filter_func)
            except AttributeError:
                pass

    def _namespace(self, basename=None, skip_cls=None, filter_func=None):
        return list(self._iter_namespace(basename, skip_cls, filter_func))

    @classmethod
    def _iter_namespace_paths(self, basepath=(), skip_cls=None, filter_func=None):
        yield basepath
        for name, attr in self._node_attrs(skip_cls, filter_func).items():
            qualpath = basepath + (name,)
            yield qualpath
            try:
                yield from attr._iter_namespace_paths(qualpath, skip_cls, filter_func)
            except AttributeError:
                pass

    @classmethod
    def _node_attrs(cls, skip_cls=None, filter_func=None):
        if skip_cls is None:
            skip_cls = _PathNode
        if filter_func is None:
            #~ filter_func = lambda k, v: isinstance(v, _PathNode)
            filter_func = lambda k, v: hasattr(v, '_parent')
        attrs = {}  # attributes can be overwritten by subclasses
        # skip all base classes of cls_ref
        # and reverse the remaining base classes
        for base_cls in cls.__mro__[-1 - len(skip_cls.__mro__) ::-1]:
            #~ if base_cls.__name__.startswith('_'): continue  # skip baseclasses
            attrs = _update_reorder(attrs, base_cls.__dict__, filter_func)
        return attrs

    def __repr__(self):
        cls = self.__class__
        msg = (f'<{cls.__module__}.{cls.__name__} object '
               f'at {hex(id(self))} named {self._pathname!r}>')
        return msg


def _update_reorder(d1, d2, d2_filter=lambda k, v: True):
    """ Update dictionary d1 with d2 with the order of d2.

        >>> d1 = {'start': 1, 'x': 2, 'middle': 3, 'y': 3, 'end': 4}
        >>> d2 = {'a': 111, 'y': 222, 'x': 333,}
        >>> _update_reorder(d1, d2)
        {'start': 1, 'a': 111, 'y': 222, 'middle': 3, 'x': 333, 'end': 4}
        #            ^^^       ^^^                    ^^^
    """
    results = {}
    replaced = set()
    d2 = dict(d2)
    while d1 or d2:
        # step 1: consume from d1 until key appears in d2
        _visited_s1 = set()
        for k1, v1 in d1.items():
            _visited_s1.add(k1)
            if (k1 in d2) or (k1 in replaced):
                break
            else:
                results[k1] = v1
        # step 2: consume from d2 until key appears in d1
        _visited_s2 = set()
        for k2, v2 in d2.items():
            if d2_filter(k2, v2):
                results[k2] = v2
            _visited_s2.add(k2)
            if k2 in d1:
                replaced.add(k2)
                break
        for key in _visited_s2:
            d2.pop(key)
        for key in _visited_s1:
            d1.pop(key)
    return results
