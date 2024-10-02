import os
import sys
import bisect
import inspect
import pathlib
import jsonpickle
import numpy as np
import pandas as pd
import tzlocal
import platform
from getmac import getmac
from subprocess import check_output, CalledProcessError
from itertools import zip_longest

from .basesystem import (
    SystemNode, Input, RefNode,
    Sum, ConditionalNode, StateNode,
    CountingLoop, NestedLoop,
    LoopLin, LoopLog, ConcatLoop, LoopBisect,
    LoopBisect as LoopBisectSystem,
    Pointer,
    Task,
    Squeeze,
    _squeeze,
    console,
    _commonpath,
)


class DataManager:
    _autosave = True
    _logging = True
    _basedir = pathlib.Path('dm')

    def __init__(self):
        self._data = {}
        self._cnt = 0  # count every write-operation == max(of all ixds) + 1
        self._times = []
        self._fnames = {}  # {root_key: [(node,) | (filename, ln_no)] }

    def __eq__(self, other):
        # check_data
        s, o = self._data, other._data
        sd = dict((k, v) for k, v in s.items() if self._is_valid_key(k))
        so = dict((k, v) for k, v in o.items() if self._is_valid_key(k))
        check_data = (sd == so)
        # check_cnt
        s, o = self._cnt, other._cnt
        check_cnt = (s == o)
        # check_fnames
        s, o = self._fnames, other._fnames
        check_fnames = (s.keys() == o.keys())
        # check_times
        s, o = self._times, other._times
        check_times = (s == o)
        return (check_data, check_cnt, check_fnames, check_times)

    @staticmethod
    def _is_key_valid(key):
        return not (  'cached' in key
                    or key.split('.')[-1].startswith('dut'))

    @staticmethod
    def _iter_node_functions(node):
        cls = node.__class__
        if func := cls.__dict__.get('__return__', None):
            yield func
        for name, attr in cls.__dict__.items():
            acls = attr.__class__
            if (not name.startswith('_')
                and len(acls.__mro__) > 1
                and acls.__mro__[1] in (Task, SystemNode)
            ):
                yield acls.__return__

    def _iter_filenames(self):
        fnames = {}
        for key, value in self._fnames.items():
            if len(value) > 1:
                yield key, *value
            elif func := next(self._iter_node_functions(value[0]), None):
                fname = func.__code__.co_filename
                if 'ipykernel' in fname:
                    continue
                ln_no = func.__code__.co_firstlineno
                yield key, fname, ln_no
                value = fname, ln_no
            else:
                node = value[0]
                msg = f"WARNING: can't extract filename from {node._key!r}: "
                msg += "add 'def __return__(): return' to class"
                #~ print(msg)
                yield key, msg, None
                value = msg, None
            fnames[key] = value
        self._fnames = fnames

    def _get_info(self):
        info = {}
        info['timezone'] = tzlocal.get_localzone().key
        info['network'] = getmac.get_mac_address()
        uname = platform.uname()
        info['arch'] = uname.machine
        info['hostname'] = uname.node
        info['system'] = uname.system
        info['version'] = uname.version
        info['release'] = uname.release
        info['python'] = sys.version
        # get filenames
        filenames = {}
        for (rkey, fname, ln_no) in self._iter_filenames():
            if fname not in filenames:
                finfo = {}
                try:
                    if not os.path.isfile(fname):
                        raise FileNotFoundError
                    fdir = os.path.dirname(fname)
                    hg_repo = check_output(['hg', 'root', '--cwd', fdir])
                    hg_repo = hg_repo.decode().rstrip('\n')
                    finfo['hg_repo'] = hg_repo
                    #~ hg_path = check_output(['hg', 'path', '-R', hg_repo])
                    #~ hg_path = hg_path.decode().rstrip('\n')
                    #~ finfo['hg_path'] = hg_path
                    hg_id = check_output(['hg', 'id', '-i', '-n', '-R', hg_repo])
                    id, nid = hg_id.decode().rstrip('\n').split()
                    finfo['hg_id'] = '{}:{}'.format(nid.strip('+'), id)
                except (CalledProcessError, FileNotFoundError):
                    pass
                filenames[fname] = finfo
            if ln_no is not None:
                node_ln = filenames[fname].setdefault('node_ln', {})
                node_ln[f'{rkey}'] = ln_no
        if filenames:
            filenames = [{'filename': fn} | finf for fn, finf in filenames.items()]
            info['filenames'] = filenames
        return info

    @staticmethod
    def _get_key(node_path_key):
        try:
            return node_path_key._key           # assume node-object
        except AttributeError as e:
            if isinstance(node_path_key, tuple):
                return '.'.join(node_path_key)  # assume tuple-path
            elif isinstance(node_path_key, str):
                return node_path_key            # assume dotted path-str
            else:
                msg = f'needs either _PathNode, tuple-path or dot-str: {node_path_key = }'
                raise ValueError(msg)

    def write(self, node_path_key, value, overwrite=False):
        key = self._get_key(node_path_key)
        cnts, values = self._data.setdefault(key, ([], []))
        if overwrite:
            cnts[-1:] = [self._cnt]
            values[-1:] = [value]
        else:
            cnts.append(self._cnt)
            values.append(value)
        self._times.append(pd.Timestamp.now())
        self._cnt += 1

    def read(self, node, cnt=None, idx=-1):
        key = self._get_key(node)
        cnts, values = self._data[key]
        if cnt is None:
            # return the last value by default: idx=-1
            if idx is None:
                idx = slice(None)
            return values[idx]
        # piece-wise constant functions: search greatest idx with cnts[idx] <= cnt
        idx_left = bisect.bisect_right(cnts, cnt) - 1
        if idx_left < 0:
            return np.nan
        else:
            return values[idx_left]

    def __contains__(self, node):
        key = self._get_key(node)
        return key in self._data

    def last_write_cnt(self, node):
        key = self._get_key(node)
        cnts, _ = self._data.get(key, ([-1], None))
        return cnts[-1]

    def values_after(self, node, ref_node):
        if not ref_node:
            _, vals = self.read_raw(node)
            if len(vals) > 1:
                vals = [vals]
            return vals
        cnts, _ = self._data[ref_node._key]
        #~ print(f'{cnts = }')
        node_cnts, node_vals = self._data.get(node._get_attr()._key, ([], []))
        # search smallest idx with node_cnts[idx] >= cnt
        idxs = [bisect.bisect_left(node_cnts, cnt) for cnt in cnts]
        values = []
        for idx, idx_next in zip_longest(idxs, idxs[1:], fillvalue=None):
            val = node_vals[idx:idx_next]
            if len(val) == 1:
                val = val[0]
            elif len(val) == 0:
                #~ val = pd.NA
                val = ''
            values.append(val)
        #~ import ipdb; ipdb.set_trace()
        return values

    def values_before(self, node, ref_node):
        cnts, _ = self._data[ref_node._get_attr()._key]
        values = [self.read(node, cnt=cnt) for cnt in cnts]
        return values

    def _to_dict(self, pickler=None):
        d = {}
        d['_cnt'] = self._cnt
        info = self._get_info()
        d['_info'] = info
        for k in sorted(self._data):
            v = self._data[k]
            if not self._is_key_valid(k):
                continue
            cnts, vals = v
            d[k] = {'cnts': cnts, 'vals': vals}
        d['_times'] = [str(ts) for ts in self._times]
        return d

    def _from_dict(self, d):
        self._cnt = d['_cnt']
        fnames = {};
        for finfo in d['_info']['filenames']:
            for node_key, ln_no in finfo['node_ln'].items():
                fnames[node_key] = finfo['filename'], ln_no
        self._fnames = fnames
        self._times = [pd.Timestamp(ts) for ts in d['_times']]
        self._data = {}
        for k in sorted(d):
            if k in ('_cnt', '_times', '_fnames', '_info'):
                continue
            item = d[k]
            self._data[k] = (item['cnts'], item['vals'])
        return self

    def to_json(self, fname=''):
        if not self._times:
            return
        dct = self._to_dict()
        jtw = JsonTabWriter()
        jstr = jtw.to_json(dct)
        keys = ''.join(f'_{key}' for key in self._fnames)
        tstamp = str(self._times[0]).replace(' ', '_').partition('.')[0]
        fname = fname if fname != '' else f'dm_{tstamp}{keys}.json'
        os.makedirs(self._basedir, exist_ok=True)
        fname = self._basedir / fname
        with open(fname, 'w') as file:
            file.writelines(jstr)
        return fname

    @classmethod
    def from_json(cls, fname=''):
        if not fname:
            files = []
            for fn in os.listdir(os.getcwd() / cls._basedir):
                if fn.startswith('dm_') and fn.endswith('.json'):
                    files.append(fn)
            if files:
                fname = cls._basedir / sorted(files)[-1]
                print(fname)
            else:
                msg = f"no 'dm/dm_*.json' file found"
                raise ValueError(msg)
        with open(fname, 'r') as file:
            jstr = file.read()
        dm = cls()
        dct = jsonpickle.loads(jstr)
        dm._from_dict(dct)
        return dm

    def to_yaml(self, fname=''):
        pass

    @classmethod
    def from_yaml(self, fname=''):
        pass

    def eval(self, node, lazy=None, auto_save=False):
        dm = self
        if inspect.isclass(node) and issubclass(node, SystemNode):
            node = node()
        elif isinstance(node, SystemNode):
            node._root._apply_configuration(dm)
        self.write('__cmd__', f'eval({node._key})')
        node_root = node._root
        self._fnames[node_root._key] = (node_root,)  # for _iter_filenames()
        if lazy is None:
            value = node._eval(dm)
        else:
            _lazy, node._lazy = node._lazy, lazy
            value = node._eval(dm)
            node._lazy = _lazy
        if auto_save:
            fname = self.to_json()
            if self._logging:
                #~ console.print('─' * len(str(fname)))
                console.print()
                console.print(f'{fname}')
        return value

    def run(self, node, auto_save=True, logging=False, live=False):
        logging = logging and self._logging
        # todo: add node._reset() or just delete data-dict?!?
        if auto_save and self._autosave:
            self.__init__()
        dm = self
        if inspect.isclass(node) and issubclass(node, SystemNode):
            node = node()
        elif isinstance(node, SystemNode):
            node._root._apply_configuration(dm)
        if hasattr(node, '_check_input_limits'):
            node._check_input_limits(dm)
        node_root = node._root
        self.write('__cmd__', f'run({node._key})')
        self._fnames[node_root._key] = (node_root,)  # for _iter_filenames()
        node = node.__mainloop__()
        if live:
            values = node._run_live(dm)
        else:
            values = node._run(dm, logging=logging)
        lines = []
        if logging:
            passed = []
            cls = node.__class__
            cls_namespace = cls._node_attrs(skip_cls=Task)
            for name, attr in cls_namespace.items():
                if name.startswith('test'):
                    test_node = getattr(node, name)
                    test_passed = []
                    for op_name in test_node._outputs:
                        check_node = test_node._get_subnode( (op_name, 'check') )
                        check_result = _squeeze(check_node, dm)
                        test_passed += check_result
                    passed.append(all(test_passed))
            msg = []
            if num_failed := passed.count(False):
                msg.append( f'[red]{num_failed} failed[/]' )
            if num_passed := passed.count(True):
                msg.append( f'[green]{num_passed} passed[/]' )
            if msg:
                lines.append( f'Summery: {", ".join(msg)}' )
        if auto_save and self._autosave:
            fname = self.to_json()
            lines.append( f'File: {fname}' )
        if logging:
            #~ console.print('─' * len(str(fname)))
            if not live:
                console.print()
        for line in lines:
            console.print(line)
        #~ return self.read_task(node)
        return values

    def run_live(self, node, auto_save=True):
        return self.run(node, auto_save=auto_save, live=True)

    def run_log(self, node, auto_save=True):
        return self.run(node, auto_save=auto_save, logging=True)

    def read_namespace(self, node, basename=None):
        if inspect.isclass(node) and issubclass(node, SystemNode):
            node = node()
        names = ['' if basename == '' else node._pathname]
        names += node._iter_namespace(basename=basename, skip_cls=Task)
        return names

    def read_raw(self, node):
        """Return either saved values from data-manager or input config"""
        if inspect.isclass(node) and issubclass(node, SystemNode):
            node = node()
        key = self._get_key(node)
        if node in self:
            cnts, values = self._data[key]
            return cnts, values
        else:
            msg = f'no data available for node: {key}'
            raise ValueError(msg)

    def read_function(self, node, names=['__return__'], set_index=True):
        if inspect.isclass(node) and issubclass(node, SystemNode):
            node = node()
        vpaths = [() if n in ('__return__', '') else (n,) for n in names]
        data, loop_colomns = self._read_node_data(node, vpaths)
        df = pd.DataFrame(data)
        #~ import ipdb; ipdb.set_trace()
        #~ df.columns = list(nodes.keys())
        if set_index:
            if loop_colomns:
                df = df.set_index(loop_colomns)
            else:
                df.index = [''] * len(df)
        return df

    def _read_node_data(self, node, vpaths=[()], exclude=['dut']):
        node_path = node._get_namedpath()
        vn_short = vpaths[-1]
        vnode = node._get_subnode(vn_short)
        vn_path = vnode._get_namedpath()
        data = {}
        # extract hloops from: vnode, vnode._parent, ..., vnode._root
        n = vnode
        npath_short = ()
        h_loops = []
        while True:
            try:
                _sections = self.read(n.sections)
            except AttributeError:
                _sections = []
            #~ import ipdb; ipdb.set_trace()
            for section in _sections:
                #~ import ipdb; ipdb.set_trace()
                if npath_short in section.get('v_paths', []):
                    hloop_names = section.get('h_loops', [])
                    for names_zipped in hloop_names[::-1]:
                        h_loops += [getattr(n, name) for name in names_zipped[::-1]]
                    break
            #~ import ipdb; ipdb.set_trace()
            if n._parent is None:
                break
            else:
                npath_short = (n._name,)
                n = n._parent
        loop_colomns = []
        for loop in reversed(h_loops):
            path = loop._get_namedpath()
            common, (_, short_path) = _commonpath(node_path, path)
            if common == node_path:
                node_name = '.'.join(short_path) if short_path else path[-1]
            else:
                node_name = loop._pathname
            data[node_name] = self.values_before(loop._get_attr(), vnode)
            loop_colomns.append(node_name)
            #~ import ipdb; ipdb.set_trace()
        # extract function input arguments
        if (len(vpaths) == 1 and vpaths[0] == () and
            hasattr(vnode, '_iter_return_arguments')
        ):
            for _node in vnode._iter_return_arguments():
                path = _node._get_namedpath()
                common, (_, short_path) = _commonpath(vn_path, path)
                if common == vn_path:  # and short_path:
                    node_name = '.'.join(short_path) if short_path else '__return__'
                else:
                    node_name = _node._pathname
                if [s for s in exclude if s in node_name]:
                    continue
                data[node_name] = self.values_before(_node._get_attr(), vnode)
        # extract node data
        for path in vpaths:
            col_name = '.'.join(path)
            col_name = col_name or '__return__'
            _node = node._get_subnode(path)
            data[col_name] = self.values_before(_node._get_attr(), vnode)
        #~ import ipdb; ipdb.set_trace()
        return data, loop_colomns

    def read_task(self, node,
                       section=None,
                       set_index=True,
                       show_output_value=True,
                       show_output_check=True,
                       exclude=['dut'],
                       #~ hide_const=False,
                       #~ hide_nondefaults=False,
    ):
        """Read all vnodes from the final or task/test section"""
        if inspect.isclass(node) and issubclass(node, SystemNode):
            node = node()
        vpaths = []
        col_names = []
        if hasattr(node, 'sections'):
            _sections = self.read(node.sections)
            _sections = {s['debug_name']: s for s in _sections}
            if section is None and node._outputs:
                for name in node._outputs:
                    if show_output_value:
                        vpaths.append( (name,) )
                        col_names.append( (node._name, name) )
                    if show_output_check:
                        vpaths.append( (name, 'check') )
                        col_names.append( (node._name, 'check') )
            elif (section is None and not node._outputs
                  and () in _sections.get('final', {}).get('v_paths', [])
            ):
                vpaths = _sections['final']['v_paths']
            elif section is None and not node._outputs:
                _section = _sections['task']
                for vpath in _section['v_paths']:
                    vnode = node._get_subnode(vpath)
                    for name in vnode._outputs:
                        if show_output_value:
                            vpaths.append( vpath + (name,) )
                            col_names.append( (vpath[-1], name) )
                        if show_output_check:
                            vpaths.append( vpath + (name, 'check') )
                            col_names.append( (vpath[-1], 'check') )
                    if not vnode._outputs:
                        # vnode has no outputs, just use default __return__
                        vpaths.append( vpath )
                        #~ col_names.append( (vpath[-1],) )
            else:
                section = 'task' if section == 'test' else section
                try:
                    _section = _sections[section]
                    vpaths = _section['v_paths']
                except KeyError:
                    pass
        if not vpaths:
            vpaths = [ () ]  # default is same like read_function(node)
        data, loop_colomns = self._read_node_data(node, vpaths, exclude)
        df = pd.DataFrame(data)
        #~ import ipdb; ipdb.set_trace()
        #~ df.columns = list(nodes.keys())
        if set_index:
            if loop_colomns:
                df = df.set_index(loop_colomns)
            else:
                df.index = [''] * len(df)
        if (len(df.columns) == len(col_names)
            and not all(len(names) == 1 for names in col_names)
        ):
            df.columns = pd.MultiIndex.from_tuples(col_names)
        #~ df = df.loc[:, ~df.columns.isin(exclude)]
        #~ import ipdb; ipdb.set_trace()
        return df

    def read_input_config(self, node):
        return node._read_input_config()


class JsonTabWriter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.indents = [0]
        self.s = ''
        self.llen = 0
        self.p = jsonpickle.Pickler()

    def write(self, s):
        self.s += s
        self.llen += len(s)
        #~ self.indents[-1] += len(s)

    def newline(self):
        yield f'{self.s}\n'
        self.s =  ' ' * self.indents[-1]
        self.llen = 0

    def tab(self):
        self.indents.append(self.indents[-1] + self.llen)
        self.llen = 0

    def untab(self):
        self.indents.pop()

    def lines(self, obj):
        return self.to_json(obj).split('\n')

    def to_json(self, o, level=0):
        """code from ...
        https://stackoverflow.com/questions/10097477/python-json-array-newlines
        """
        if isinstance(o, (str, bool, int, float, type(None))):
            self.write(jsonpickle.dumps(o))
        elif isinstance(o, dict):
            self.write('{')
            if len(self.indents) == 2:
                yield from self.newline()
                self.write('    ')
            self.tab()
            for n, (k, v) in enumerate(o.items()):
                self.write(f'"{k}": ')
                yield from self.to_json(v, level+1)
                if n < len(o) - 1:
                    self.write(',')
                    yield from self.newline()
            self.write('}')
            self.untab()
        elif isinstance(o, list):
            self.write('[')
            self.tab()
            for n, e in enumerate(o):
                yield from self.to_json(e, level+1)
                if n < len(o) - 1:
                    self.write(',')
                    if isinstance(e, (dict, tuple, set)) or hasattr(e, '_to_dict'):
                        yield from self.newline()
                    else:
                        self.write(' ')
            self.write(']')
            self.untab()
        elif isinstance(o, tuple):
            yield from self.to_json({'py/tuple': list(o)}, level+1)
        elif isinstance(o, (np.ndarray, np.generic, pd.Timestamp)):
            #~ yield from self.to_json(o.flatten().tolist())
            yield from self.to_json(self.p.flatten(o), level+1)
        else:
            #~ dct = o._to_dict() if hasattr(o, '_to_dict') else self.p.flatten(o)
            dct = o._to_dict(self.p)
            yield from self.to_json(dct, level+1)
        #~ self.s
        if level == 0:
            yield from self.newline()
