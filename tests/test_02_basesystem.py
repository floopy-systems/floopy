import pytest
from floopy.basesystem import (
    SystemNode, Input, RefNode, _TupleNode, _ValueNode, NOTHING,
    LoopLin,
    _POSITIONAL_OR_KEYWORD, _VAR_POSITIONAL, _KEYWORD_ONLY, _VAR_KEYWORD,
)


def test_RefNode_with_str():
    class MyA(SystemNode):
        x = 100

    class MyB(SystemNode):
        a1 = MyA()
        y1 = RefNode('a1.x')
        y2 = RefNode('y1')

    b = MyB()
    assert b.y1._other == ('a1', 'x')
    assert b.y2._other == ('y1',)


def test_empty_ValueNode():
    class Quad_N(SystemNode):
        x1 = _ValueNode()
        x2 = _ValueNode(2)

    qn = Quad_N()
    assert qn.x1._value == NOTHING
    assert '_value' not in qn.x1.__class__.__dict__
    assert qn.x2._value == 2
    assert '_value' in qn.x2.__class__.__dict__


def test_repr_of_node_values():
    class Quad(SystemNode):
        x = Input()
        gain = Input(1)
        offs = Input(0)

    class MyTasks(SystemNode):
        task_quad = Quad(3, offs=LoopLin(0, 10, num=3))

    tasks = MyTasks()
    value = tasks.task_quad._repr()
    assert value == 'Quad(x=3, offs=LoopLin(start=0, stop=10, num=3))'


def test_input_annotations():  # = {inp_name: {(sub_path,): anno_obj}}
    class MySqrt(SystemNode):
        x = Input()
        def pre_abs(x: '>>> is_positive'):
            return abs(x)
        def post(x: 'is_float >>> is_integer'):
            return int(x)
        def __return__(x: 'is_positive >>> is_float'):
            return x**0.5

    class TaskProgram(SystemNode):
        x = Input()
        s1 = MySqrt(x)
        s2 = MySqrt(x)
        def __return__(x: 'is_integer >>> '):
            return x

    tp = TaskProgram(None)
    assert tp._input_annotations == {
        'x': {  ():                'is_integer >>> ',

                ('s1',):           'is_positive >>> is_float',
                ('s1', 'pre_abs'): '>>> is_positive',
                ('s1', 'post'):    'is_float >>> is_integer',

                ('s2',):           'is_positive >>> is_float',
                ('s2', 'pre_abs'): '>>> is_positive',
                ('s2', 'post'):    'is_float >>> is_integer'}}


def test_TupleNode():
    class MySystem(SystemNode):
        n = _TupleNode(1, 2)
        m = 3
        p = _TupleNode()

    my = MySystem()
    assert my.n[0]._pathname == 'MySystem.n._0'
    assert my.n[0] == my.n._0
    assert my.n[0]._value == 1
    assert my.n._get_value() == (1, 2)
    assert list(my._namespace()) == [
        'MySystem.n',
        'MySystem.n._0',
        'MySystem.n._1',
        'MySystem.m',
        'MySystem.p',
    ]

    assert len(my.n) == 2
    assert list(my.n) == [my.n[0], my.n[1]]
    assert list(reversed(my.n)) == [my.n[1], my.n[0]]
    assert len(my.p) == 0
    assert list(my.p) == []


def test_non_default_var_inputs():
    try:
        class MyArgs(SystemNode):
            a = Input(default=404, kind=_VAR_POSITIONAL)
    except ValueError as e:
        assert str(e) == "VAR_POSITIONAL input 'a' can not have a default value"

    try:
        class MyArgs(SystemNode):
            b = Input(default=404, kind=_VAR_KEYWORD)
    except ValueError as e:
        assert str(e) == "VAR_KEYWORD input 'b' can not have a default value"


def test_unique_input_names():
    try:
        class MyArgs(SystemNode):
            a = Input()
            a = Input()
    except TypeError as e:
        assert str(e) == "duplicate input name 'a'"


def test_order_of_non_default_inputs():
    try:
        class MyArgs(SystemNode):
            a = Input(0)
            b = Input()
    except TypeError as e:
        assert str(e) == 'non-default input follows default input'


def test_correct_kind_of_input_order():
    try:
        class MyArgs(SystemNode):
            args = Input(kind=_VAR_POSITIONAL)
            debug = Input(False)
    except TypeError as e:
        assert str(e) == ("kind of input 'debug' must be greater than "
                          "POSITIONAL_OR_KEYWORD")


def test_missing_non_default_inputs():
    class MyArgs(SystemNode):
        a = Input()
        b = Input(kind=_KEYWORD_ONLY)

    try:
        a = MyArgs()
    except TypeError as e:
        assert str(e) == "MyArgs() missing 1 required positional argument: 'a'"

    try:
        a = MyArgs(1)
    except TypeError as e:
        assert str(e) == "MyArgs() missing 1 required keyword-only argument: 'b'"


def test_VAR_POSITIONAL_input():
    class MyArgs(SystemNode):
        x = Input()
        y = Input(kind=_VAR_POSITIONAL)
        z = Input(kind=_KEYWORD_ONLY)

    a = MyArgs(1, 2, 3, z=4)
    assert a._arguments == {'x': 1, 'y': (2, 3), 'z': 4}

    a = MyArgs(1, z=2)
    assert a._arguments == {'x': 1, 'z': 2}


def test_input_arguments():
    class MyArgs(SystemNode):
        a = Input()
        b = Input()
        c = Input(0)
        d = Input(0)
        e = Input(0)

    a = MyArgs(1, 2)
    assert a._arguments == {'a': 1, 'b': 2}

    a = MyArgs(1, 2, 3)
    assert a._arguments == {'a': 1, 'b': 2, 'c': 3}

    a = MyArgs(1, 2, e=3)
    assert a._arguments == {'a': 1, 'b': 2, 'e': 3}

    a = MyArgs(1, b=2)
    assert a._arguments == {'a': 1, 'b': 2}

    a = MyArgs(a=1, b=2)
    assert a._arguments == {'a': 1, 'b': 2}

    try:
        a = MyArgs(1, 2, 3, 4, 5, 6, 7)
    except TypeError as e:
        msg = 'MyArgs() takes from 2 to 5 positional arguments but 7 were given'
        assert str(e) == msg

    try:
        a = MyArgs()
    except TypeError as e:
        msg = "MyArgs() missing 2 required positional arguments: 'a', 'b'"
        assert str(e) == msg

    try:
        a = MyArgs(b=2)
    except TypeError as e:
        msg = "MyArgs() missing 1 required positional argument: 'a'"
        assert str(e) == msg

    # test input argument order
    a = MyArgs(1, 2, d=3, c=4)
    assert a._arguments == {'a': 1, 'b': 2, 'd': 3, 'c': 4}

    a = MyArgs(1, d=3, c=4, b=2)
    assert a._arguments == {'a': 1, 'd': 3, 'c': 4, 'b': 2}


@pytest.fixture
def A():
    class A(SystemNode):
        x = Input(111)
        y = Input(222)
        def __return__(x, gain=1):
            return gain * x**2
    return A


@pytest.fixture
def B(A):
    class B(SystemNode):
        x = Input(10)
        y = Input(20)
        a1 = A(x=1000)
        a2 = A(x, y)
        a1_x_ref = a1.x
        def __return__(x, y, a1, a2):
            return x, a2
    return B


@pytest.fixture
def C(A, B):
    class C(SystemNode):
        x1 = Input(11)
        x2 = Input(222)
        b1 = B(y=RefNode(x2))
        b2 = B(x=b1.a2.y, y=A())
        def myfunc(x=1, y=b1.y):
            return x + y
        def yourfunc(x1, x2):
            return a1 + a2
    return C


def test_sub_systems(C):
    # new sub-sub-instance for every root-system C()
    c1 = C()
    c2 = C()
    assert c1.b2.y != c2.b2.y


def test_class_level_of_chained_reference(C):
    # the _same_ RefNode-subclass (B.a2.x) is instantiated several times
    c = C()
    assert c.b1.a2.x._get_attr() == c.b1.x
    assert c.b2.a2.x._get_attr() == c.x2  # ref -> ref -> ref


def test_systemnode_input(C):
    # system-input with sub-system extening the path-namespace
    c = C()
    assert c.b2.y.y._pathname == 'C.b2.y.y'


def test_namespace(C):
    # test complete namespace of path-nodes
    c = C()
    assert list(c.b1._namespace()) == [
        'C.b1.x',
        'C.b1.y',
        'C.b1.a1',
        'C.b1.a1.x',
        'C.b1.a1.y',
        'C.b1.a1.gain',
        'C.b1.a2',
        'C.b1.a2.x',
        'C.b1.a2.y',
        'C.b1.a2.gain',
        'C.b1.a1_x_ref',
    ]


def test_iter_children_and_iter_arguments(C):
    c = C()
    assert list(c.b1._iter_children()) == [c.b1.x, c.x2, c.b1.a1, c.b1.a2]
    assert dict(c.b1._iter_arguments()) == {
            'x': 10,
            'y': 222,
            'a1': 'C.b1.a1.__return__',
            'a2': 'C.b1.a2.__return__'}
    assert list(c.b1.a1._iter_children()) == [c.b1.a1.x, c.b1.a1.gain]
    assert dict(c.b1.a1._iter_arguments()) == {'x': 1000, 'gain': 1}

    assert list(c.b2._iter_children()) == [c.x2, c.b2.y, c.b2.a1, c.b2.a2]
    assert dict(c.b2._iter_arguments()) == {
            'x': 222,
            'y': 'C.b2.y.__return__',
            'a1': 'C.b2.a1.__return__',
            'a2': 'C.b2.a2.__return__'}
    assert list(c.b2.a2._iter_children())  == [c.x2, c.b2.a2.gain]
    assert dict(c.b2.a2._iter_arguments()) == {'x': 222, 'gain': 1}


def test_default_return_function(C):
    c = C()
    assert C._arg_names == ()
    assert c.__class__.__return__() == None


def test_staticmethod_for_return_function(C):
    c = C()
    assert c.b1.a1.__class__.__return__(5) == 5**2


def test_extraction_of_return_arguments(A, B):
    assert A._arg_names == ('x', 'gain')
    assert B._arg_names == ('x', 'y', 'a1', 'a2')


def test_input_configuration(A):
    A_inputs = [(n, inp.default, inp.kind) for n, inp in A._inputs.items()]
    assert A_inputs == [
        ('x', 111, _POSITIONAL_OR_KEYWORD),
        ('y', 222, _POSITIONAL_OR_KEYWORD),
    ]


def test_function_subsystem_for_default_inputs(C):
    c = C()
    assert dict(c.myfunc._iter_arguments()) == {'x': 1, 'y': 222}
    assert dict(c.yourfunc._iter_arguments()) == {'x1': 11, 'x2': 222}


def test_subsystem_for_return_function(C):
    c = C()
    assert c.myfunc._arg_names == ('x', 'y')
    assert c.myfunc.__class__.__return__(1, 2) == 3
    assert c.myfunc._eval_value() == 223


def test_unnamed_one_time_configuration(A):
    a = A(3)._apply_configuration()
    assert a._eval_value() == 9
    assert a.x._pathname == 'A.x'



