import pytest
import numpy as np
import floopy as fly
from floopy.basesystem import (
    _POSITIONAL_OR_KEYWORD, _VAR_POSITIONAL, _KEYWORD_ONLY, _VAR_KEYWORD,
    SystemNode, Input, RefNode, _TupleNode, _ValueNode, NOTHING,
    StateNode, ConditionalNode, Sum, Pointer,
    InputRangeError,
    CountingLoop,
    LoopBisect,
    ConcatLoop,
    NestedLoop,
    LoopLin,
    LoopLog,
    Squeeze,
    Task,
)
from floopy import loop, loop_lin, loop_log, loop_bisect
from floopy.datamanager import DataManager

DataManager._autosave = False
DataManager._logging = False


TODO = 0


def test_plan_input_limits():
    class TestNumberIsEven(fly.Task):
        x = fly.Input(20, min=2, max=20)

    class TestSquareRoot(fly.Task):
        x = fly.Input(min=0)

        def task_square(x):
            return x**2

        def task_root(task_square):
            return task_square**(1/2)

        def __return__(x1=fly.Squeeze(x), x2=fly.Squeeze(task_root)):
            return (x1 == x2).all()
            return x == task_root


    class TestMean(fly.Task):
        start = fly.Input(min=0, max=10)
        stop = fly.Input(min=0, max=10)

        def task_square_numbers(x=fly.loop_lin(start, stop)):
            return x**2

        def final_mean(x=fly.Squeeze(task_square_numbers)):
            return x.mean()

        def __return__(final_mean, start, stop):
            return final_mean == (start + stop) / 2

    with pytest.raises(InputRangeError,
                       match='2 <= 1 <= 20 \\(TestPlan.test_even.x\\)',
    ):
        dm = DataManager()
        class TestPlan(fly.Task):
            x = fly.Input(321)
            y = 123
            z = x

            number = fly.loop(1, 2, 3)

            test_even = TestNumberIsEven(number)
            test_sqroot = TestSquareRoot(x=fly.loop(3, fly.loop_lin(1, 2, num=3), 5))
            test_mean = TestMean(start=7, stop=10)

        tp = TestPlan()
        tp._check_input_limits(dm)


    with pytest.raises(InputRangeError,
                       match='0 <= -2 <= None \\(TestPlan.test_sqroot.x\\)',
    ):
        dm = DataManager()
        class TestPlan(fly.Task):
            x = fly.Input(321)
            y = 123
            z = x

            number = fly.loop(1, 2, 3)

            test_even = TestNumberIsEven(number)
            test_sqroot = TestSquareRoot(x=fly.loop(3, fly.loop_lin(-2, 2, num=3), 5))
            test_mean = TestMean(start=7, stop=10)

        tp = TestPlan()
        tp._check_input_limits(dm)

    with pytest.raises(InputRangeError,
                       match='0 <= 17 <= 10 \\(TestPlan.test_mean.stop\\)',
    ):
        dm = DataManager()
        class TestPlan(fly.Task):
            x = fly.Input(321)
            y = 123
            z = x

            number = fly.loop(1, 2, 3)

            test_even = TestNumberIsEven(number)
            test_sqroot = TestSquareRoot(x=fly.loop(3, fly.loop_lin(1, 2, num=3), 5))
            test_mean = TestMean(start=7, stop=17)

        tp = TestPlan()
        tp._check_input_limits(dm)


def test_dynamic_input_limits():
    class Quad(SystemNode):
        x = Input()
        def __return__(x):
            return x**2

    with pytest.raises(InputRangeError, match='1 <= 9 <= 5'):
        dm = DataManager()
        class MyArgs(Task):
            a = Input(Quad(3), min=1, max=5)
        task = MyArgs()
        task._check_input_limits(dm)

    class Quad(Task):
        x = Input(3, min=5, max=10)
        def __return__(x):
            return x**2

    with pytest.raises(InputRangeError, match='5 <= 3 <= 10'):
        dm = DataManager()
        class MyArgs(Task):
            a = Input(Quad(), min=1, max=5)
        task = MyArgs()
        task._check_input_limits(dm)

    with pytest.raises(InputRangeError, match='1 <= 25 <= 5'):
        dm = DataManager()
        class MyArgs(Task):
            a = Input(Quad(5), min=1, max=5)
        task = MyArgs()
        task._check_input_limits(dm)


def test_static_input_limits():
    dm = DataManager()
    class MyArgs(Task):
        a = Input(5, min=1, max=5)
    task = MyArgs()
    task._check_input_limits(dm)

    with pytest.raises(InputRangeError, match='1 <= 0 <= 5'):
        dm = DataManager()
        class MyArgs(Task):
            a = Input(0, min=1, max=5)
        task = MyArgs()
        task._check_input_limits(dm)

    with pytest.raises(InputRangeError, match='1 <= 7 <= 5'):
        dm = DataManager()
        class MyArgs(Task):
            a = Input(7, min=1, max=5)
        task = MyArgs()
        task._check_input_limits(dm)


def test_default_auto_input():
    class MyTest(Task):
        x = Input(min=3, max=5, default=loop_lin(num=3))

    assert MyTest.x.start._value == 3
    assert MyTest.x.stop._value == 5


def test_dm_auto_input_min_max():
    class MyTest(Task):
        g = Input(min=1, default=2, max=3)
        x = Input(min=3, default=4, max=5)
        y = Input(min=6, default=7, max=8)
        z = Input(min=9, default=10, max=11)

    class TestPlan(Task):
        test_my = MyTest(
            g=loop_lin(1.5, 2.5, num=3),
            x=loop_lin(num=5),
            y=loop_log(num=8),
            z=loop_bisect(cycles=4),
        )

    assert TestPlan.test_my.g.start._value == 1.5
    assert TestPlan.test_my.g.stop._value == 2.5

    assert TestPlan.test_my.x.start._value == 3
    assert TestPlan.test_my.x.stop._value == 5

    assert TestPlan.test_my.y.start._value == 6
    assert TestPlan.test_my.y.stop._value == 8

    assert TestPlan.test_my.z.start._value == 9
    assert TestPlan.test_my.z.stop._value == 11


def test_dm_read_task():
    # test complete namespace of TestPlan
    class TestNumberIsEven(Task):
        x = Input(20)

        def __return__(x):
            return not (x % 2)


    class TestSquareRoot(Task):
        x = Input()

        def task_square(x):
            return x**2

        def task_root(task_square):
            return task_square**(1/2)

        def __return__(x1=Squeeze(x), x2=Squeeze(task_root)):
            return (x1 == x2).all()


    class TestMean(Task):
        start = Input()
        stop = Input()

        def task_square_numbers(x=loop_lin(start, stop)):
            return x**2

        def final_mean(x=Squeeze(task_square_numbers)):
            return x.mean()

        def __return__(final_mean, start, stop):
            return final_mean == (start + stop) / 2


    class TestPlan(Task):
        number = loop(1, 2, 3)

        test_even = TestNumberIsEven(number)
        test_sqroot = TestSquareRoot(x=loop(3, 4, 5))
        test_mean = TestMean(start=7, stop=12)


    dm = DataManager()
    tp = TestPlan()
    dm.run(tp)

    # tp
    df = dm.read_task(tp)
    assert repr(df).split('\n') == [
        '        test_even  test_sqroot  test_mean',
        'number                                   ',
        '1           False         True      False',
        '2            True         True      False',
        '3           False         True      False']

    df = dm.read_task(tp.number)
    assert repr(df).split('\n') == [
        '  __return__',
        '           1',
        '           2',
        '           3']

    # tp.test_even
    df = dm.read_task(tp.test_even)
    assert repr(df).split('\n') == [
        '                 x  __return__',
        'TestPlan.number               ',
        '1                1       False',
        '2                2        True',
        '3                3       False']
    df = dm.read_task(tp.test_even.x)
    assert repr(df).split('\n') == [
        '                 __return__',
        'TestPlan.number            ',
        '1                         1',
        '2                         2',
        '3                         3']

    # tp.test_sqroot
    df = dm.read_task(tp.test_sqroot)
    assert repr(df).split('\n') == [
        '                        x1               x2  __return__',
        'TestPlan.number                                        ',
        '1                [3, 4, 5]  [3.0, 4.0, 5.0]        True',
        '2                [3, 4, 5]  [3.0, 4.0, 5.0]        True',
        '3                [3, 4, 5]  [3.0, 4.0, 5.0]        True']
    df = dm.read_task(tp.test_sqroot, 'task')
    assert repr(df).split('\n') == [
        '                   task_square  task_root',
        'TestPlan.number x                        ',
        '1               3            9        3.0',
        '                4           16        4.0',
        '                5           25        5.0',
        '2               3            9        3.0',
        '                4           16        4.0',
        '                5           25        5.0',
        '3               3            9        3.0',
        '                4           16        4.0',
        '                5           25        5.0']
    df = dm.read_task(tp.test_sqroot.task_square)
    assert repr(df).split('\n') == [
        '                                        x  __return__',
        'TestPlan.number TestPlan.test_sqroot.x               ',
        '1               3                       3           9',
        '                4                       4          16',
        '                5                       5          25',
        '2               3                       3           9',
        '                4                       4          16',
        '                5                       5          25',
        '3               3                       3           9',
        '                4                       4          16',
        '                5                       5          25']
    df = dm.read_task(tp.test_sqroot.task_square.x)
    assert repr(df).split('\n') == [
        '                                        __return__',
        'TestPlan.number TestPlan.test_sqroot.x            ',
        '1               3                                3',
        '                4                                4',
        '                5                                5',
        '2               3                                3',
        '                4                                4',
        '                5                                5',
        '3               3                                3',
        '                4                                4',
        '                5                                5']
    df = dm.read_task(tp.test_sqroot.task_root)
    assert repr(df).split('\n') == [
        '                                        task_square  __return__',
        'TestPlan.number TestPlan.test_sqroot.x                         ',
        '1               3                                 9         3.0',
        '                4                                16         4.0',
        '                5                                25         5.0',
        '2               3                                 9         3.0',
        '                4                                16         4.0',
        '                5                                25         5.0',
        '3               3                                 9         3.0',
        '                4                                16         4.0',
        '                5                                25         5.0']
    df = dm.read_task(tp.test_sqroot.task_root.task_square)
    assert repr(df).split('\n') == [
        '                                        __return__',
        'TestPlan.number TestPlan.test_sqroot.x            ',
        '1               3                                9',
        '                4                               16',
        '                5                               25',
        '2               3                                9',
        '                4                               16',
        '                5                               25',
        '3               3                                9',
        '                4                               16',
        '                5                               25']
    df = dm.read_task(tp.test_sqroot.x)
    assert repr(df).split('\n') == [
        '                 __return__',
        'TestPlan.number            ',
        '1                         3',
        '1                         4',
        '1                         5',
        '2                         3',
        '2                         4',
        '2                         5',
        '3                         3',
        '3                         4',
        '3                         5']

    # tp.test_mean
    df = dm.read_task(tp.test_mean)
    assert repr(df).split('\n') == [
        '                 final_mean  __return__',
        'TestPlan.number                        ',
        '1                 93.166667       False',
        '2                 93.166667       False',
        '3                 93.166667       False']
    df = dm.read_task(tp.test_mean, 'task')
    assert repr(df).split('\n') == [
        '                                       task_square_numbers',
        'TestPlan.number task_square_numbers.x                     ',
        '1               7                                       49',
        '                8                                       64',
        '                9                                       81',
        '                10                                     100',
        '                11                                     121',
        '                12                                     144',
        '2               7                                       49',
        '                8                                       64',
        '                9                                       81',
        '                10                                     100',
        '                11                                     121',
        '                12                                     144',
        '3               7                                       49',
        '                8                                       64',
        '                9                                       81',
        '                10                                     100',
        '                11                                     121',
        '                12                                     144']
    df = dm.read_task(tp.test_mean.task_square_numbers)
    assert repr(df).split('\n') == [
        '                    __return__',
        'TestPlan.number x             ',
        '1               7           49',
        '                8           64',
        '                9           81',
        '                10         100',
        '                11         121',
        '                12         144',
        '2               7           49',
        '                8           64',
        '                9           81',
        '                10         100',
        '                11         121',
        '                12         144',
        '3               7           49',
        '                8           64',
        '                9           81',
        '                10         100',
        '                11         121',
        '                12         144']
    df = dm.read_task(tp.test_mean.task_square_numbers.x)
    assert repr(df).split('\n') == [
        '                 idx  start num_step  __return__',
        'TestPlan.number                                 ',
        '1                  0      7   (6, 1)           7',
        '1                  1      7   (6, 1)           8',
        '1                  2      7   (6, 1)           9',
        '1                  3      7   (6, 1)          10',
        '1                  4      7   (6, 1)          11',
        '1                  5      7   (6, 1)          12',
        '2                  0      7   (6, 1)           7',
        '2                  1      7   (6, 1)           8',
        '2                  2      7   (6, 1)           9',
        '2                  3      7   (6, 1)          10',
        '2                  4      7   (6, 1)          11',
        '2                  5      7   (6, 1)          12',
        '3                  0      7   (6, 1)           7',
        '3                  1      7   (6, 1)           8',
        '3                  2      7   (6, 1)           9',
        '3                  3      7   (6, 1)          10',
        '3                  4      7   (6, 1)          11',
        '3                  5      7   (6, 1)          12']
    df = dm.read_task(tp.test_mean.start)
    assert repr(df).split('\n') == [
        '                 __return__',
        'TestPlan.number            ',
        '1                         7']
    df = dm.read_task(tp.test_mean.stop)
    assert repr(df).split('\n') == [
        '                 __return__',
        'TestPlan.number            ',
        '1                        12']


def test_task_namespace_refnode():
    class MyTask(Task):
        def task_xin():
            return 10

        def task_quad(task_xin):
            return task_xin**2

    t = MyTask()
    dm = DataManager()
    dm.run(t)

    cnts, vals = dm.read_raw(t.task_xin)
    assert vals == [10]
    cnts, vals = dm.read_raw(t.task_quad)
    assert vals == [100]


def test_concat_with_loops_and_values():
    class MyTask(Task):
        x = ConcatLoop(404, CountingLoop(4), 404, LoopLin(20, 30, num=3), 404)

    task = MyTask()
    dm = DataManager()
    data = dm.run(task.x)
    assert data == [404, 0, 1, 2, 3, 404, 20.0, 25.0, 30.0, 404]


def test_dm_iter_filenames():
    class A(Task):
        def __return__():
            return 123

    class B(Task):
        def __return__():
            return 321

    class C(Task):
        a = A()
        b = B()
        def __return__(a, b):
            return a + b

    dm = DataManager()
    a = A()
    dm.run(a)
    b = B()
    dm.run(b)
    c = C()
    dm.run(c)

    fnames = dm._fnames
    assert fnames == {
        'A': (a,),
        'B': (b,),
        'C': (c,),
    }
    fnames = list(dm._iter_filenames())
    fnames = list(dm._iter_filenames())
    assert [isinstance(fn, str) for (k, fn, _) in fnames] == [
        True, True, True]

    fnames = dm._fnames
    assert {k: isinstance(v[0], str) for k, v in fnames.items()} == {
        'A': True,
        'B': True,
        'C': True,
    }


def test_dm_iter_node_functions():
    class MyQuad(Task):
        temp = Input()
        def __return__():
            return


    class MyTask(Task):
        temp = ConcatLoop(20, 50, 100)

        def setup():
            return

        def task_1(temp):
            return f'task_1({temp=})'

        task_quad = MyQuad(temp)

        def final_fit():
            return

        def __return__():
            return


    task = MyTask()
    dm = DataManager()
    funcs = list(dm._iter_node_functions(task))
    assert funcs == [
        MyTask.__return__,
        # temp skipped
        MyTask.setup.__class__.__return__,
        MyTask.task_1.__class__.__return__,
        # task_quad skipped
        MyTask.final_fit.__class__.__return__,
    ]


def test_double_Squeeze():
    class MyQuad(Task):
        temp = Input()
        n = Input()

        def task_1(temp, n, offs=ConcatLoop(0, 500)):
            return n + temp / 100 + offs

        def task_2(s2=Squeeze(task_1, as_array=False)):
            return s2

        def __return__(s1=Squeeze(task_1, as_array=False)):
            return s1


    class MyPlan(Task):
        temp = ConcatLoop(20, 50, 100)

        task_quad = MyQuad(temp, n=LoopLin(30, 40, step=10))
        task_n404 = MyQuad(temp, n=404)

        def __return__(s3=Squeeze(task_quad.task_1, as_array=False)):
            return s3


    plan = MyPlan()

    # MyPlan.task_quad.task_1
    #   inner_hloops (squeezed):
    #       task_quad.task_1.offs = ConcatLoop(0, 500)
    dm = DataManager()
    dm.run(plan.task_quad.task_1)
    values = dm.read(plan.task_quad.task_1, idx=None)
    assert values == [30.2, 530.2]

    # MyPlan.task_quad
    #   inner_hloops (squeezed):
    #       task_quad.task_1.offs = ConcatLoop(0, 500)
    #           task_quad.n = LoopLin(30, 40, step=10)
    dm = DataManager()
    dm.run(plan.task_quad)
    values = dm.read(plan.task_quad.task_1, idx=None)
    assert values == [30.2, 530.2, 40.2, 540.2]

    # MyPlan
    #   outer_hloops (non-squeezed):
    #       temp = ConcatLoop(20, 50, 100)
    dm = DataManager()
    dm.run(plan)

    values = dm.read(plan.task_quad.s1, idx=None)
    assert values == [
        [30.2, 530.2, 40.2, 540.2],
        [30.5, 530.5, 40.5, 540.5],
        [31.0, 531.0, 41.0, 541.0]]

    values = dm.read(plan.task_quad.task_2.s2, idx=None)
    assert values == [
        [30.2, 530.2], [40.2, 540.2],
        [30.5, 530.5], [40.5, 540.5],
        [31.0, 531.0], [41.0, 541.0]]

    values = dm.read(plan.s3, idx=None)
    assert values == [[
        30.2, 530.2, 40.2, 540.2,
        30.5, 530.5, 40.5, 540.5,
        31.0, 531.0, 41.0, 541.0]]

    values = dm.read(plan.task_n404.task_2.s2, idx=None)
    assert values == [
        [404.2, 904.2],
        [404.5, 904.5],
        [405.0, 905.0]]

    values = dm.read(plan.task_n404.s1, idx=None)
    assert values == [
        [404.2, 904.2],
        [404.5, 904.5],
        [405.0, 905.0]]


def test_Squeeze():
    class MyQuad(Task):
        temp = Input()
        n = Input()
        #~ x = LoopLin(25, 35, num=3)

        def task(n, temp):
            return n + temp / 100

        def __return__(d=Squeeze(task, as_array=False)):
            return d


    class MyTask(Task):
        temp = ConcatLoop(20, 50, 100)
        #~ x = LoopLin(0.01, 0.03, num=2)

        def task_1(temp):
            return f'task_1({temp=})'

        task_quad = MyQuad(temp, n=LoopLin(30, 40, step=10))


    task = MyTask()
    dm = DataManager()
    dm.run(task)

    values = dm.read(task.task_quad, idx=None)
    assert values == [[30.2, 40.2], [30.5, 40.5], [31.0, 41.0]]


def test_task__zipped_hv_loops():
    class MyTask(Task):
        x = LoopLin(15, 25, num=3, loop_level=2)
        y = ConcatLoop(20, 50, 100, loop_level=2)

        def task_add(x, y, offs=ConcatLoop(0, 0.05)):
            return x + y + offs

        def task_mul(x, y):
            return x * y

        def __return__(a=Squeeze(task_add), b=Squeeze(task_mul)):
            #~ a = array([ 35.  ,  35.05,  70.  ,  70.05, 125.  , 125.05])
            #~ b = array([ 300., 1000., 2500.])
            return np.hstack([a, b])


    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert (data == np.array(
        [  35.  ,   35.05,   70.  ,   70.05,  125.  ,  125.05,
          300.  , 1000.  , 2500.  ]
    )).all()

    values = dm.read(task.task_add, idx=None)
    assert values == [35.0, 35.05, 70.0, 70.05, 125.0, 125.05]

    values = dm.read(task.task_mul, idx=None)
    assert values == [300.0, 1000.0, 2500.0]


def test_task__empty():
    class MyTask(Task):
        pass

    task = MyTask()
    dm = DataManager()

    data = dm.eval(task.sections)
    assert data == [{'v_paths': [()], 'h_loops': [], 'debug_name': 'task'}]

    data = dm.run(task)
    assert data == None

    steps = task._called_steps(dm)
    assert steps == ['__return__']


def test_task__only_return():
    class MyTask(Task):
        def __return__():
            return 321

    task = MyTask()
    dm = DataManager()

    data = dm.eval(task.sections)
    assert data == [{'v_paths': [()], 'h_loops': [], 'debug_name': 'task'}]

    data = dm.run(task)
    assert data == 321

    steps = task._called_steps(dm)
    assert steps == ['__return__']


def test_task__eval_env_h_loops():
    class MyTask(Task):
        env_x = LoopLin(15, 25, num=2)
        env_x.__class__._lazy = False

        #~ def env_x(x = LoopLin(15, 25, num=2)):
            #~ return x

        y = ConcatLoop(20, 100)

        def __return__():
            return 321

    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert data == [321, 321, 321, 321]
    values = dm.read(task.env_x, idx=None)
    assert values == [
        15.0, 15.0, 25.0, 25.0,  # if _lazy = False
        #~ 15.0, 25.0,              # if _lazy = True
    ]
    values = dm.read(task.y, idx=None)
    assert values == [20, 100, 20, 100]

    steps = task._called_steps(dm)
    assert steps == [
        '__return__', '__return__', '__return__', '__return__']


def test_task__nested_loops():
    class MyTask(Task):
        x = LoopLin(15, 25, num=2)
        y = ConcatLoop(20, 100)

        def __return__(x, y):
            return 321, x, y

    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert data == [
        (321, 15.0, 20),
        (321, 15.0, 100),
        (321, 25.0, 20),
        (321, 25.0, 100)]


def test_task__zipped_loops():
    class MyTask(Task):
        x = LoopLin(15, 25, num=2, loop_level=2)
        y = ConcatLoop(20, 100, loop_level=2)

        def __return__(x, y):
            return 321, x, y

    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert data == [
        (321, 15.0, 20),
        (321, 25.0, 100),
    ]
    steps = task._called_steps(dm)
    assert steps == ['__return__', '__return__']


def test_task__nested_loops_with_setup_teardown():
    class MyTask(Task):
        x = LoopLin(15, 25, num=2)
        y = ConcatLoop(20, 100)

        def setup():
            return 'setup()'

        def __return__(x, y):
            return 321, x, y

        def teardown():
            return 'teardown()'

    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert data == [
        (321, 15.0, 20),
        (321, 15.0, 100),
        (321, 25.0, 20),
        (321, 25.0, 100),
    ]

    steps = task._called_steps(dm)
    assert steps == [
        'setup',
        '__return__', '__return__', '__return__', '__return__',
        'teardown']


def test_task__nested_loops_with_setup_task_teardown():
    class MyTask(Task):
        x = LoopLin(15, 25, num=2)
        y = ConcatLoop(20, 100)

        def setup():
            return 'setup()'

        def task(x, y):
            return 321, x, y

        def teardown():
            return 'teardown()'

    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert data == None

    steps = task._called_steps(dm)
    assert steps == ['setup', 'task', 'task', 'task', 'task', 'teardown']

    values = dm.read(task.task, idx=None)
    assert values == [
        (321, 15.0, 20),
        (321, 15.0, 100),
        (321, 25.0, 20),
        (321, 25.0, 100),
    ]


def test_task__collect_hloops_only_before_task_section():
    class MyTask(Task):
        x = LoopLin(15, 25, num=2)

        def task(x, y):
            return 321, x, y

        y = ConcatLoop([20, 100])


    task = MyTask()
    dm = DataManager()
    sections = task.sections._eval(dm)
    s = sections[0]
    assert s['debug_name'] == 'task'
    assert s['h_loops'] == [('x',)]


def test_task__nested_loops_with_task_teardown_return():
    class MyTask(Task):
        x = LoopLin(15, 25, num=2)
        y = ConcatLoop(20, 100)

        def task(x, y):
            return 321, x, y

        def teardown():
            return 'teardown()'

        # todo/release: resolve namespace conflict
        #~ def __return__(result=Squeeze(task)):
        def __return__(val=Squeeze(task)):
            return val

    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert (data == np.array(
        [[321.,  15.,  20.],
         [321.,  15., 100.],
         [321.,  25.,  20.],
         [321.,  25., 100.]])
    ).all()

    steps = task._called_steps(dm)
    assert steps == ['task', 'task', 'task', 'task', 'teardown', '__return__']

    _, vals = dm.read_raw(task.task)
    assert vals == [
        (321, 15.0, 20),
        (321, 15.0, 100),
        (321, 25.0, 20),
        (321, 25.0, 100),
    ]


def test_Task_of_Task_with_nested_hloops():
    class MyQuad(Task):
        n = Input(-404)

        def setup():
            return f'task_quad.SETUP()'

        def task(n):
            return f'task_quad.task({n=})'

        def __return__(y=Squeeze(n, as_array=False)):
            return f'task_quad.__return__({y=})'

        def teardown():
            return f'task_quad.TEARDOWN'


    class MyTask(Task):
        dut = Input(123)

        def setup_1():
            return f'SETUP_1()'

        def setup_2():
            return f'SETUP_2()'

        env_temp = ConcatLoop(20, 100)
        x = LoopLin(0.01, 0.03, num=2)

        def task_1(dut, env_temp, x=CountingLoop(2)):
            return f'task_1({env_temp=}, {x=})'

        task_quad = MyQuad(n=LoopLin(30, 40, 10))

        def teardown():
            return f'TEARDOWN()'

        def __return__(x):
            return 321 + x


    task = MyTask()
    dm = DataManager()

    data = dm.run(task)
    assert data == 321.03

    steps = task._called_steps(dm)  # todo: add argument values from dm!
    assert steps == [
        'setup_1', 'setup_2',
        # env_temp=20, x=0.01
        'task_1',
        'task_1',
        'task_quad.setup',
        'task_quad.task', 'task_quad.task',
        'task_quad.teardown',
        'task_quad',
        # env_temp=20, x=0.03
        'task_1',
        'task_1',
        'task_quad.setup',
        'task_quad.task', 'task_quad.task',
        'task_quad.teardown',
        'task_quad',
        # env_temp=100, x=0.01
        'task_1',
        'task_1',
        'task_quad.setup',
        'task_quad.task', 'task_quad.task',
        'task_quad.teardown',
        'task_quad',
        # env_temp=100, x=0.03
        'task_1',
        'task_1',
        'task_quad.setup',
        'task_quad.task', 'task_quad.task',
        'task_quad.teardown',
        'task_quad',
        #
        'teardown',
        '__return__',
   ]


def test_Pointer():
    class A(SystemNode):
        names = ConcatLoop('x', 'double', 'quad', result=RefNode( ('p',) ))
        p = Pointer(names)


    class B(SystemNode):
        a = A()

        x = 3

        def double(x):
            return 2*x

        def quad(x):
            return x*x

    b = B()
    dm = DataManager()
    data = dm.run(b.a.names)
    assert data == [3, 6, 9]


def test_StateNode_with_variable_init_value():
    class A(SystemNode):
        a = 3
        b = 4
        def zinit(a, b):
            return a**2 + b**2

        z = StateNode(init=zinit)

        def __return__(z):
            return 2*z

    a = A()
    dm = DataManager()
    data = a._eval(dm)
    assert data == 50


def test_single_loop():
    c1 = CountingLoop(4)
    c2 = LoopLin(10, 20, num=3)

    dm = DataManager()
    data = dm.run(c1)
    assert data == [0, 1, 2, 3]

    data = dm.run(c2)
    assert data == [10.0, 15.0, 20.0]


def test_task_with_simple_loop():
    class MyTask(Task):
        x = CountingLoop(4)

        def __return__(x):
            return x**2

    dm = DataManager()
    task = MyTask()
    data = dm.run(task)
    assert data == [0, 1, 4, 9]


def test_sequence_of_task_evaluations():
    class MyTask(Task):
        #~ temp = LoopLin(20, 100, num=2)
        temp = ConcatLoop(20, 100)
        temp.__class__._lazy = False

        def task_1():
            return 'task_1'

        def task_2():
            return 'task_2'

    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert data == None

    _, values = dm.read_raw(task.temp)
    assert values == [20, 20, 100, 100]

    steps = task._called_steps(dm)
    assert steps == ['task_1', 'task_2', 'task_1', 'task_2']


def test_loop_level_for_zipped_loops():
    class ZippedLoops(Task):
        temp = Input(0)
        n = CountingLoop(3, loop_level=0)
        x = LoopLin(150, 170, 10, loop_level=0)
        z = Input(0)
        def __return__(temp, n, x, z):
            return temp, (n, x), z


    class MyTasks_1(Task):
        task = ZippedLoops()

    dm = DataManager()
    tasks_1 = MyTasks_1()
    data = dm.run(tasks_1)
    assert data == None
    # todo/release: if Task has only one task-node then automatically do
    #                    __return__(task): return task

    _, vals = dm.read_raw(tasks_1.task)
    assert vals == [
        (0, (0, 150), 0),
        (0, (1, 160), 0),
        (0, (2, 170), 0)]


    class MyTasks_2(Task):
        task = ZippedLoops(temp=LoopLin(20, 100, num=2), z=CountingLoop(2))

    dm = DataManager()
    tasks_2 = MyTasks_2()
    data = dm.run(tasks_2)
    assert data == None

    _, vals = dm.read_raw(tasks_2.task)
    assert vals == [
        (20.0, (0, 150), 0),
        (20.0, (0, 150), 1),
        (20.0, (1, 160), 0),
        (20.0, (1, 160), 1),
        (20.0, (2, 170), 0),
        (20.0, (2, 170), 1),
        (100.0, (0, 150), 0),
        (100.0, (0, 150), 1),
        (100.0, (1, 160), 0),
        (100.0, (1, 160), 1),
        (100.0, (2, 170), 0),
        (100.0, (2, 170), 1)]


def test_NestedLoop_with_zipped_loops():
    class MyTask(SystemNode):
        c1 = CountingLoop(2)
        c2 = CountingLoop(3)
        c22 = LoopLin(100, 300, step=100)
        c3 = CountingLoop(4)
        mainloop = NestedLoop(loops_cached = ((c1,), (c2, c22), (c3,)) )
        def __return__(mainloop):
            return mainloop

    dm = DataManager()
    t = MyTask()
    data = dm.run(t.mainloop)
    assert data == [
        (0, (0, 100), 0),
        (0, (0, 100), 1),
        (0, (0, 100), 2),
        (0, (0, 100), 3),
        #
        (0, (1, 200), 0),
        (0, (1, 200), 1),
        (0, (1, 200), 2),
        (0, (1, 200), 3),
        #
        (0, (2, 300), 0),
        (0, (2, 300), 1),
        (0, (2, 300), 2),
        (0, (2, 300), 3),
        ###
        (1, (0, 100), 0),
        (1, (0, 100), 1),
        (1, (0, 100), 2),
        (1, (0, 100), 3),
        (1, (1, 200), 0),
        (1, (1, 200), 1),
        (1, (1, 200), 2),
        (1, (1, 200), 3),
        (1, (2, 300), 0),
        (1, (2, 300), 1),
        (1, (2, 300), 2),
        (1, (2, 300), 3),
    ]


def test_read_task_from_LoopBisect():
    class MyLoopBisect(Task):
        start = Input()
        stop = Input()
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

        x = LoopLin(start_n, stop, step_n)

        def __return__(x):
            return x

    class MyTasks(Task):
        task_bisect = MyLoopBisect(20, 30)


    dm = DataManager()
    tasks = MyTasks()

    data = dm.run(tasks)
    assert data == None

    _, vals = dm.read_raw(tasks.task_bisect)
    assert vals == [20, 30, 25.0, 22.5, 27.5]

    df = dm.read_function(tasks.task_bisect, ['start_n', 'step_n', ''])
    assert repr(df).split('\n') == [
        '        start_n  step_n  __return__',
        'n x                                ',
        '0 20.0     20.0    10.0        20.0',
        '  30.0     20.0    10.0        30.0',
        '1 25.0     25.0    10.0        25.0',
        '2 22.5     22.5     5.0        22.5',
        '  27.5     22.5     5.0        27.5']


def test_LoopBisect():
    s = LoopBisect(20, 30)._apply_configuration()
    dm = DataManager()
    data = dm.run(s)
    assert data == [20, 30, 25.0, 22.5, 27.5]


def test_RefNode_to_self_return():
    # test RefNode( () ) for self.__return__
    class MyQuad(SystemNode):
        x = Input(3)

        def __return__(x):
            return x**2

        y = __return__  # = RefNode( () )

        def scaled(y=__return__, scale=1.5):
            return scale * y

    myquad = MyQuad()
    assert myquad.y._level == 1
    assert myquad.scaled.y._level == 2

    dm = DataManager()
    assert dm.eval(myquad.y) == 9
    assert dm.eval(myquad) == 9
    assert dm.eval(myquad.scaled) == 13.5
    data =  dm._data
    assert data == {
        '__cmd__': ([0, 3, 4],
                    ['eval(MyQuad.y)', 'eval(MyQuad)', 'eval(MyQuad.scaled)']),
        'MyQuad.x':            ([1], [3]),
        'MyQuad':              ([2], [9]),
        'MyQuad.scaled.scale': ([5], [1.5]),
        'MyQuad.scaled':       ([6], [13.5])}


def test_SelfRefNode():
    class MyQuad(SystemNode):
        x = Input(3)
        _offs = 100
        def __return__(self, x):
            return x**2 + self._offs

    myquad = MyQuad()
    assert myquad.self._pathname == 'MyQuad.self'

    dm = DataManager()
    assert dm.eval(myquad) == 109


def test_if_dm_eval_instantiate_a_SystemNode_class():
    class MyQuad(SystemNode):
        x = Input(5)
        def __return__(x):
            return x**2

    dm = DataManager()
    result = dm.eval(MyQuad)
    assert result == 25


def test_ConcatLoop_with_variable_loop_structure():
    class MyTask(SystemNode):
        c1 = LoopLin(15, 25, num=3)
        def hello():
            return 'hello'
        c2 = LoopLin(3, 0)

        c = ConcatLoop(c1, hello, c2)

        id = CountingLoop(2)

        def loops(self):
            return [(self.id.__mainloop__(),), (self.c.__mainloop__(),)]
        mainloop = NestedLoop(loops_cached=loops, result=c)
        def __mainloop__(self):
            return self.mainloop

    task = MyTask()
    dm = DataManager()
    data = dm.run(task)
    assert data == [
        15.0, 20.0, 25.0, 'hello', 3, 2, 1, 0,
        15.0, 20.0, 25.0, 'hello', 3, 2, 1, 0]


def test_ConcatLoop():
    c1 = CountingLoop(3)._apply_configuration()
    c2 = LoopLin(10, 20, num=3)._apply_configuration()
    c = ConcatLoop(200, c1, 400, 600, c2, 800)._apply_configuration()
    dm = DataManager()
    data = dm.run(c)
    assert data == [200, 0, 1, 2, 400, 600, 10.0, 15.0, 20.0, 800]


def test_LoopLin():
    dm = DataManager()
    s = LoopLin(5, 10)._apply_configuration()
    data = dm.run(s)
    assert data == [5, 6, 7, 8, 9, 10]

    dm = DataManager()
    s = LoopLin(10, 5)._apply_configuration()
    data = dm.run(s)
    assert data == [10, 9, 8, 7, 6, 5]

    dm = DataManager()
    s = LoopLin(5, 5)._apply_configuration()
    data = dm.run(s)
    assert data == [5]

    dm = DataManager()
    s = LoopLin(5, 5, num=0)._apply_configuration()
    data = dm.run(s)
    assert data == []

    dm = DataManager()
    s = LoopLin(5, 10, step=0.5)._apply_configuration()
    assert dm.eval(s.num_step) == (11, 0.5)
    data = dm.run(s)
    assert data == [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    dm = DataManager()
    s = LoopLin(5, 10, num=3)._apply_configuration()
    assert dm.eval(s.num_step) == (3, 2.5)
    data = dm.run(s)
    assert data == [5.0, 7.5, 10.0]


def test_LoopLog():
    dm = DataManager()
    s = LoopLog(1, 1000, num=4)._apply_configuration()
    data = dm.run(s)
    assert data == [1.0, 10.0, 100.0, 1000.0]

    dm = DataManager()
    s = LoopLog(1, 64, num=7, base=2)._apply_configuration()
    data = dm.run(s)
    assert data == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def test_task_evaluation_order():
    class One(Task):
        x = Input(0)
        def setup():
            return 'Quad.setup'
        def __return__(x):
            return x
        def teardown():
            return 'Quad.teardown'

    class Two(Task):
        def __return__():
            return 222

    class MyTaskSequence(Task):
        num = CountingLoop(2)

        task_one = One(x=CountingLoop(3))
        task_two = Two()

    tasks = MyTaskSequence()
    dm = DataManager()
    data = dm.run(tasks)
    assert data == None

    steps = tasks._called_steps(dm)
    assert steps == [
        # num = 0
        'task_one.setup',
        'task_one', 'task_one', 'task_one',
        'task_one.teardown',
        'task_two',
        # num = 1
        'task_one.setup',
        'task_one', 'task_one', 'task_one',
        'task_one.teardown',
        'task_two',
    ]
    if TODO:
        # todo: extend _eval() for offline-evaluation from dm
        #       * add cnt=None to _eval() for dm.read
        #       * automatically detect the relevant cnts where
        #         dependency-nodes are changed
        assert test_plan == [
            (0, 0, ('task_one', 'setup')),
            (0, 1, 0, ('task_one',)),
            (0, 1, 1, ('task_one',)),
            (0, 1, 2, ('task_one',)),
            (0, 2, ('task_one', 'teardown')),
            (0, 3, ('task_two',)),
            (1, 0, ('task_one', 'setup')),
            (1, 1, 0, ('task_one',)),
            (1, 1, 1, ('task_one',)),
            (1, 1, 2, ('task_one',)),
            (1, 2, ('task_one', 'teardown')),
            (1, 3, ('task_two',)),
        ]


@pytest.fixture
def Quad():
    class Quad(Task):
        offs = Input()
        x = Input(0)
        numbers = CountingLoop(3)
        y = Input(0)
        def __return__(x, offs, y, numbers):
            #~ print(f'CALL:  Quad({x=}, {offs=})')
            return (y + 1)*(x**2 + 2*offs) + 100*numbers
    return Quad

@pytest.fixture
def Double():
    class Double(Task):
        offs = Input()
        x = Input(0)
        quad = Input(0)
        quad_all = Input([])
        def __return__(x, offs, quad, quad_all):
            # logging
            #~ print(f'CALL:  Double({x=}, {offs=})')
            return 2*x + offs
    return Double


@pytest.fixture
def MyTasks(Quad, Double):
    class MyTasks(Task):
        offs = Input(1000)
        num = CountingLoop(2)
        #~ env_socket = ConcatLoop([101])
        #~ def env_socket(num):
            #~ msg = f'socket: dut.{num = }'
            #~ return msg

        task_quad = Quad(offs, x=num)
        task_double = Double(
            offs,
            x = CountingLoop(2),
            quad = task_quad,
            quad_all = Squeeze(task_quad, as_array=False),
        )
    return MyTasks


def test_dm_read_namespace(MyTasks):
    tasks = MyTasks()
    dm = DataManager()
    names = dm.read_namespace(tasks.task_quad)
    assert names == [
        'MyTasks.task_quad',
        'MyTasks.task_quad.offs',
        'MyTasks.task_quad.x',
        'MyTasks.task_quad.numbers',
        'MyTasks.task_quad.y',
    ]


def test_hv_sections(MyTasks):
    tasks = MyTasks()
    dm = DataManager()
    sections = tasks.sections._eval(dm)
    assert sections == [
        {'v_paths': [('task_quad',), ('task_double',)],
         'h_loops': [('num',)],
         'debug_name': 'task'},
    ]


def test_dm_readouts(MyTasks):
    tasks = MyTasks()
    dm = DataManager()
    data = dm.run(tasks)
    assert data == None

    steps = tasks._called_steps(dm)
    assert steps == [
        # num = 0
        'task_quad', 'task_quad', 'task_quad',
        'task_double',
        'task_double',
        # num = 1
        'task_quad', 'task_quad', 'task_quad',
        'task_double',
        'task_double',
    ]
    if TODO:
        assert test_plan_result == [
            # num = 0
            (0, 0, 0, ('task_quad',)),
            (0, 0, 1, ('task_quad',)),
            (0, 0, 2, ('task_quad',)),
            (0, 1, 0, ('task_double',)),
            (0, 1, 1, ('task_double',)),
            # num = 1
            (1, 0, 0, ('task_quad',)),
            (1, 0, 1, ('task_quad',)),
            (1, 0, 2, ('task_quad',)),
            (1, 1, 0, ('task_double',)),
            (1, 1, 1, ('task_double',))
        ]

    cnts, vals = dm.read_raw(tasks.task_double.quad_all)
    assert vals == [[2000, 2100, 2200], [2001, 2101, 2201]]

    # test read_task()
    df = dm.read_task(tasks.task_quad)
    assert repr(df).split('\n') == [
        '                     x  offs  y  __return__',
        'MyTasks.num numbers                        ',
        '0           0        0  1000  0        2000',
        '            1        0  1000  0        2100',
        '            2        0  1000  0        2200',
        '1           0        1  1000  0        2001',
        '            1        1  1000  0        2101',
        '            2        1  1000  0        2201']

    df = dm.read_task(tasks.task_double)
    #~ _df = df[ df.columns.drop('MyTasks.env_socket') ]
    assert repr(df).split('\n') == [
        '               offs  quad            quad_all  __return__',
        'MyTasks.num x                                            ',
        '0           0  1000  2200  [2000, 2100, 2200]        1000',
        '            1  1000  2200  [2000, 2100, 2200]        1002',
        '1           0  1000  2201  [2001, 2101, 2201]        1000',
        '            1  1000  2201  [2001, 2101, 2201]        1002']

    # test delayed reset of inner loops (NestedLoop._next_updates)
    data = dm.read_raw(tasks.task_quad.numbers.idx)
    assert data == (
        [  31,   49,   63,  146,  167,  181],
        [   0,    1,    2,    0,    1,    2])
    data = dm.read_raw(tasks.task_quad.numbers)
    assert data == (
        [  36,   54,   68,  152,  172,  186],
        [   0,    1,    2,    0,    1,    2])
    data = dm.read_raw(tasks.task_quad)
    assert data == (
        [  40,   56,   70,  154,  174,  188],
        [2000, 2100, 2200, 2001, 2101, 2201])
    """
    >>> from matplotlib import pylab as plt
    >>> _, axs = plt.subplots(3, sharex=True)

    >>> cnts, vals = dm.read_raw(tasks.task_quad.numbers.idx)
    >>> axs[0].step(cnts, vals, '.-', where='post')

    >>> cnts, vals = dm.read_raw(tasks.task_quad.numbers)
    >>> axs[1].step(cnts, vals, '.-', where='post')

    >>> cnts, vals = dm.read_raw(tasks.task_quad)
    >>> axs[2].step(cnts, vals, '.-', where='post')

    >>> plt.show(block=False)
    """

    # test read_input_config()
    df = dm.read_input_config(tasks.task_quad)
    assert repr(df).split('\n') == [
        ' MyTasks.task_quad      (VALUES)',
        '                                ',
        '          .offs =   MyTasks.offs',
        '             .x =    MyTasks.num',
        '             .y =              0']

    df = dm.read_input_config(tasks.task_double)
    assert repr(df).split('\n') == [
        ' MyTasks.task_double                    (VALUES)',
        '                                                ',
        '            .offs =                 MyTasks.offs',
        '               .x =          CountingLoop(num=2)',
        '            .quad =            MyTasks.task_quad',
        '        .quad_all =   Squeeze(MyTasks.task_quad)']


def test_loops_of_current_task_and_glp_state_resolving(Quad, Double):
    class MyTasks(Task):
        offs = Input()

        task_quad = Quad(offs, y=CountingLoop(2), x=CountingLoop(4))

        task_double = Double(x=CountingLoop(3), offs=offs)

        def task_post(offs):
            # todo/release/nice-to-have: this should be a Task
            #   then move _get_loopnames() back to Task
            # logging
            #~ print(f'CALL:  task_post({offs=})')
            return 'return task_post'

        task_my_dbl = Double(offs, x=CountingLoop(2))

        def task_final(cnt=CountingLoop(2)):  # todo: no _get_loopnames()
            # would be fixed by todo above (this func must be Task)
        #~ def task_final():
            # logging
            #~ print('CALL:  task_final()')
            return 'return task_final'

    tasks = MyTasks(offs=1000)._apply_configuration()
    dm = DataManager()

    data = dm.run(tasks)
    assert data == None

    steps = tasks._called_steps(dm)
    assert steps == 24 * ['task_quad'] + [
        'task_double',
        'task_double',
        'task_double',
        'task_post',
        'task_my_dbl',
        'task_my_dbl',
        'task_final', 'task_final',
    ]
    if TODO:  # test loops of current task, glp state-resolving
        assert test_plan_result == [
            (0, 0, 0, 0, ('task_quad',)),
            (0, 0, 0, 1, ('task_quad',)),
            (0, 0, 0, 2, ('task_quad',)),
            (0, 0, 0, 3, ('task_quad',)),
            (0, 0, 1, 0, ('task_quad',)),
            (0, 0, 1, 1, ('task_quad',)),
            (0, 0, 1, 2, ('task_quad',)),
            (0, 0, 1, 3, ('task_quad',)),
            (0, 0, 2, 0, ('task_quad',)),
            (0, 0, 2, 1, ('task_quad',)),
            (0, 0, 2, 2, ('task_quad',)),
            (0, 0, 2, 3, ('task_quad',)),
            (0, 1, 0, 0, ('task_quad',)),
            (0, 1, 0, 1, ('task_quad',)),
            (0, 1, 0, 2, ('task_quad',)),
            (0, 1, 0, 3, ('task_quad',)),
            (0, 1, 1, 0, ('task_quad',)),
            (0, 1, 1, 1, ('task_quad',)),
            (0, 1, 1, 2, ('task_quad',)),
            (0, 1, 1, 3, ('task_quad',)),
            (0, 1, 2, 0, ('task_quad',)),
            (0, 1, 2, 1, ('task_quad',)),
            (0, 1, 2, 2, ('task_quad',)),
            (0, 1, 2, 3, ('task_quad',)),
            (1, 0, ('task_double',)),
            (1, 1, ('task_double',)),
            (1, 2, ('task_double',)),
            (2, ('task_post',)),
            (3, 0, ('task_my_dbl',)),
            (3, 1, ('task_my_dbl',)),
            (4, ('task_final',))
        ]


def test_double_nested_loops():
    # important for interfaces!
    class MyDoubleNestedLoop(SystemNode):
        c1 = CountingLoop(3)
        c2 = CountingLoop(2)
        n1 = NestedLoop(loops_cached = ((c1,), (c2,)) )
        c3 = CountingLoop(2)
        c4 = CountingLoop(2)
        n2 = NestedLoop(loops_cached = ((c3,), (n1,), (c4,)) )
        def __return__(n2):
            return n2

    task = MyDoubleNestedLoop()
    dm = DataManager()
    result = dm.run(task.n2)
    assert result == [
        (0, (0, 0), 0),
        (0, (0, 0), 1),
        (0, (0, 1), 0),
        (0, (0, 1), 1),
        (0, (1, 0), 0),
        (0, (1, 0), 1),
        (0, (1, 1), 0),
        (0, (1, 1), 1),
        (0, (2, 0), 0),
        (0, (2, 0), 1),
        (0, (2, 1), 0),
        (0, (2, 1), 1),
        # the second part triggers n1._restart_updates(dm)
        (1, (0, 0), 0),
        (1, (0, 0), 1),
        (1, (0, 1), 0),
        (1, (0, 1), 1),
        (1, (1, 0), 0),
        (1, (1, 0), 1),
        (1, (1, 1), 0),
        (1, (1, 1), 1),
        (1, (2, 0), 0),
        (1, (2, 0), 1),
        (1, (2, 1), 0),
        (1, (2, 1), 1)]


def test_interface_between_CountingLoop_and_dm_run():
    dm = DataManager()
    c = CountingLoop(3)
    result = dm.run(c)
    assert result == [0, 1, 2]

    ival = dm.eval(c.restart)
    dm.write(c.idx, ival)
    result = dm.run(c)
    assert result == [0, 1, 2]


if TODO:  # test Task: run mainloop in background
    class MyTask(Task):
        c1 = Input(0)
        c2 = Input(0)
        c3 = Input(0)
        def __return__(c1, c2, c3):
            return c1, c2, c3

    task = MyTask( c1 = CountingLoop(2),
                   c2 = CountingLoop(3),
                   c3 = CountingLoop(4) )._apply_configuration()

    # todo/relese/nice-to-have:
    #   task._get_loopnames() should respect task.__dict__
    #   from task._apply_configuration()
    #
    #   (very) important but it didnt touch TasSequence

    dm = DataManager()
    dm.run(task)


def test_mainloop_of_Task_in_background():
    class MyTask(Task):
        c1 = Input(0)
        c2 = Input(0)
        c3 = Input(0)
        def __return__(c1, c2, c3):
            return c1, c2, c3

    class TaskProgram(SystemNode):
        mytask = MyTask(c1 = CountingLoop(2),
                        c2 = CountingLoop(3),
                        c3 = CountingLoop(4))

    tp = TaskProgram()

    dm = DataManager()
    result = dm.run(tp.mytask)
    assert result == [
        (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
        (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
        (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3),

        (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3),
        (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3),
        (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3)]


def test_subclassing():
    # todo: idx is from namespace of base-class CountingLoop
    class MyCounter(CountingLoop):
        def is_interrupted(idx):
            return idx == 1
    assert True


if TODO:  # test _has_next()
    class MyTask(SystemNode):
        c1 = CountingLoop(2)
        c2 = CountingLoop(3)
        c3 = CountingLoop(4)
        mainloop = NestedLoop(loops_cached = ((c1,), (c2,), (c3,)) )
        def __return__(c1, c2, c3):
            return c1, c2, c3

    dm = DataManager()
    t = MyTask()

    while t.mainloop._has_next(dm):
        t.mainloop._update(dm)
    assert dm._data['MyTask.c3.idx'][1] == [
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    assert dm._data['MyTask.c2.idx'][1] == [0, 1, 2, 0, 1, 2]
    assert dm._data['MyTask.c1.idx'][1] == [0, 1]


def test_NestedLoop_tripple_loop():
    class MyTask(SystemNode):
        c1 = CountingLoop(2)
        c2 = CountingLoop(3)
        c3 = CountingLoop(4)
        mainloop = NestedLoop(loops_cached = ((c1,), (c2,), (c3,)) )
        def __return__(c1, c2, c3):
            return c1, c2, c3

    t = MyTask()
    dm = DataManager()
    data = dm.run(t.mainloop)
    assert data == [
        (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
        (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
        (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3),

        (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3),
        (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3),
        (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3),
    ]


def test_is_interrupted_with_single_loop():
    class MyCounter(CountingLoop):
        def _is_interrupted(self, dm):
            idx = self.idx._eval(dm)
            return idx == 2

    class MyTaskInterrupt(SystemNode):
        c1 = MyCounter(6)
        mainloop = NestedLoop(loops_cached = ((c1,),) )
        def __return__(c1):
            return c1

    dm = DataManager()
    t = MyTaskInterrupt()

    a1 = dm.run(t.mainloop)
    assert a1 == [(0,), (1,), (2,)]

    a2 = dm.run(t.mainloop)
    assert a2 == [(3,), (4,), (5,)]


def test_is_interrupted_with_double_loop():
    class MyCounter(CountingLoop):
        def _is_interrupted(self, dm):
            idx = self.idx._eval(dm)
            return idx == 2

    class MyTaskInterrupt(SystemNode):
        c1 = CountingLoop(3)
        c2 = MyCounter(6)
        mainloop = NestedLoop(loops_cached = ((c1,), (c2,)) )
        def __return__(c1, c2):
            return c1, c2

    t = MyTaskInterrupt()
    dm = DataManager()
    values = dm.run(t.mainloop)

    assert values == [
        (0, 0), (0, 1), (0, 2),
        (1, 3), (1, 4), (1, 5),

        (2, 0), (2, 1), (2, 2),
        (2, 3), (2, 4), (2, 5),
    ]


def test_is_interrupted_with_tripple_loop():
    class MyCounter(CountingLoop):
        def _is_interrupted(self, dm):
            idx = self.idx._eval(dm)
            return idx == 1

    class MyTaskInterrupt(SystemNode):
        c1 = CountingLoop(2)
        c2 = CountingLoop(3)
        c3 = MyCounter(4)
        mainloop = NestedLoop(loops_cached = ((c1,), (c2,), (c3,)) )
        def __return__(c1, c2, c3):
            return c1, c2, c3

    t = MyTaskInterrupt()
    dm = DataManager()
    values = dm.run(t.mainloop)

    assert values == [
        (0, 0, 0), (0, 0, 1),
        (0, 1, 2), (0, 1, 3),

        (0, 2, 0), (0, 2, 1),
        (1, 0, 2), (1, 0, 3),

        (1, 1, 0), (1, 1, 1),
        (1, 2, 2), (1, 2, 3),
    ]


def test_CountingLoop_next():
    c = CountingLoop()._apply_configuration()
    dm = DataManager()

    dm.eval(c.next)
    dm.write(c.idx, dm.read(c.next))

    dm.eval(c.next)
    dm.write(c.idx, dm.read(c.next))

    assert dm._data == {
        '__cmd__': ([0, 4], ['eval(CountingLoop.next)', 'eval(CountingLoop.next)']),
        'CountingLoop.idx':      ([1, 3, 6], [0, 1, 2]),
        'CountingLoop.next':     ([2, 5],    [1, 2]),
    }


def test_VAR_POSITIONAL():
    class MyArgs(SystemNode):
        args = Input(kind=_VAR_POSITIONAL)
        def __return__(*args):
            return args

    z1 = StateNode(11)
    c2 = 22
    n = MyArgs(z1, c2, 33)._apply_configuration()

    dm = DataManager()
    dm.eval(n)
    assert dm._data == {
        '__cmd__':        ([0], ['eval(MyArgs)']),
        'MyArgs.args._0': ([1], [11]),
        'MyArgs.args._1': ([2], [22]),
        'MyArgs.args._2': ([3], [33]),
        'MyArgs.args':    ([4], [(11, 22, 33)]),
        'MyArgs':         ([5], [(11, 22, 33)]),
    }


def test_VAR_POSITIONAL_as_a_sub_system():
    class MyArgs(SystemNode):
        args = Input(kind=_VAR_POSITIONAL)
        def __return__(*args):
            return args

    class MySystem(SystemNode):
        z1 = StateNode(11)
        c2 = 22
        n = MyArgs(z1, c2, 33)

    s = MySystem()
    dm = DataManager()
    dm.eval(s.n)
    assert dm._data == {
        '__cmd__':            ([0], ['eval(MySystem.n)']),
        'MySystem.z1':        ([1], [11]),
        'MySystem.c2':        ([2], [22]),
        'MySystem.n.args._2': ([3], [33]),
        'MySystem.n.args':    ([4], [(11, 22, 33)]),
        'MySystem.n':         ([5], [(11, 22, 33)]),
    }


def test_StateNode():
    class MyState(SystemNode):
        z = StateNode(4)
        def __return__(z):
            return z

    state = MyState()
    dm = DataManager()
    dm.eval(state)
    dm.write(state.z, 8)
    dm.eval(state)
    assert dm._data == {
        '__cmd__':   ([0, 4], ['eval(MyState)', 'eval(MyState)']),
        'MyState.z': ([1, 3], [4, 8]),
        'MyState':   ([2, 5], [4, 8]),
    }


def test_ConditionalNode_with_numbers():
    n = ConditionalNode(0, 10, 0, 20, 1, 30, 1, 40, 0, 50)
    dm = DataManager()
    dm.eval(n)
    assert dm._data == {
        '__cmd__':                 ([0], ['eval(ConditionalNode)']),
        'ConditionalNode.args._0': ([1], [0]),
        'ConditionalNode.args._2': ([2], [0]),
        'ConditionalNode.args._4': ([3], [1]),
        'ConditionalNode.args._5': ([4], [30]),
        'ConditionalNode':         ([5], [30]),
    }


def test_ConditionalNode_default_value():
    n = ConditionalNode(0, 10, 0, 20, 0, 30)
    dm = DataManager()
    dm.eval(n)
    assert dm._data == {
        '__cmd__':                 ([0], ['eval(ConditionalNode)']),
        'ConditionalNode.args._0': ([1], [0]),
        'ConditionalNode.args._2': ([2], [0]),
        'ConditionalNode.args._4': ([3], [0]),
        'ConditionalNode.default': ([4], [0]),
        'ConditionalNode':         ([5], [0]),
    }


def test_ConditionalNode_with_RefNodes():
    class MyConditional(SystemNode):
        def cond_1(): return 0
        def one(): return 10

        def cond_2(): return 0
        def two(): return 20

        def cond_3(): return 1
        def three(): return 30

        def cond_4(): return 0
        def four(): return 40

        consequence = ConditionalNode(
                        cond_1, one,
                        cond_2, two,
                        cond_3, three,
                        cond_4, four,
        )

        # todo: '__return__ = consequence' should be possible
        def __return__(consequence):
            return consequence

    n = MyConditional()
    dm = DataManager()
    dm.eval(n)
    assert dm._data == {
            '__cmd__':              ([0], ['eval(MyConditional)']),
            'MyConditional.cond_1': ([1], [0]),
            'MyConditional.cond_2': ([2], [0]),
            'MyConditional.cond_3': ([3], [1]),
            'MyConditional.three': ([4], [30]),
            'MyConditional.consequence': ([5], [30]),
            'MyConditional': ([6], [30]),
    }


def test_Sum_basics():
    dm = DataManager()
    dm.eval(Sum(4, 6, 8))
    assert dm._data == {
        '__cmd__':     ([0], ['eval(Sum)']),
        'Sum.args._0': ([1], [4]),
        'Sum.args._1': ([2], [6]),
        'Sum.args._2': ([3], [8]),
        'Sum':         ([4], [4 + 6 + 8]),
    }


def test_Sum_wo_arguments():
    dm = DataManager()
    dm.eval(Sum())
    assert dm._data == {
        '__cmd__': ([0], ['eval(Sum)']),
        'Sum':     ([1], [0]),
    }


def test_Sum_as_sub_sytem_with_RefNodes():
    class A(SystemNode):
        x1 = Input(0)
        x2 = Input(0)
        x3 = Input(0)
        a1 = Sum(0, x1, x2, x3)
        a2 = Sum(3, 4, a1, a1.args[1], Sum(5, 3))
        # todo: a3 = a1 + a2
        a3 = Sum(a1, a2)
        def __return__(a2, a3):
            return 10*a2

    dm = DataManager()
    dm.eval(A(1, 3, 5))
    assert dm._data == {
        '__cmd__':              ([0], ['eval(A)']),
        'A.a2.args._0':         ([1], [3]),
        'A.a2.args._1':         ([2], [4]),
        'A.a1.args._0':         ([3], [0]),
        'A.x1':                 ([4], [1]),
        'A.x2':                 ([5], [3]),
        'A.x3':                 ([6], [5]),
        'A.a1':                 ([7], [9]),     # a1.args[1] = 1
        'A.a2.args._4.args._0': ([8], [5]),
        'A.a2.args._4.args._1': ([9], [3]),
        'A.a2.args._4':         ([10], [8]),
        'A.a2':                 ([11], [25]),   # a2 = 3 + 4 + 9 + 1 + 8 = 25
        'A.a3':                 ([12], [34]),
        'A':                    ([13], [250]),
    }


def test_001a():
    class MyFunc(SystemNode):
        def __return__():
            return 321

    dm = DataManager()
    dm.eval(MyFunc())
    assert dm._data == {
        '__cmd__': ([0], ['eval(MyFunc)']),
        'MyFunc':  ([1], [321]),
    }


if TODO:
    """ todo: implementation of @system decorator

        @system
        def system(func):
            class func.__name__(SystemNode):
                __return__ = func
            return sys

    This system-factory is useful to 'import' a function into a SystemNode

        class MySystem(SystemNode):
            zeros = system(math.zero_newton, arg_1, ...)
            ...

    Additionally, functions could be also evaluated by

            dm.eval(func, *args, **kwargs)

    This is useful when args are other SystemNode like Sweep(...)

    The implementation could be either

        * by an extra function or
        * by the MetaFilter: overwrite __init__ on subclassing with

            subcls.__init__ = __init_system_arguments__
            SystemNode.__init__ = __import_function_decorator__

    """
    def myfunc():
        return 321
    dm = DataManager()
    dm.eval(myfunc())
    assert dm._data == {'MyFunc': ([1], [321]), 'myfunc': ([2], [321])}


def test_002a_return_args_as_namespace_attributes():
    class MyScale(SystemNode):
        x = 1
        def __return__(x, gain, offs):
            return gain*x + offs
        gain = 1
        offs = 10

    dm = DataManager()
    dm.eval(MyScale())
    assert dm._data == {
        '__cmd__':      ([0], ['eval(MyScale)']),
        'MyScale.x':    ([1], [1]),
        'MyScale.gain': ([2], [1]),
        'MyScale.offs': ([3], [10]),
        'MyScale':      ([4], [11]),
    }


def test_002b_different_types_of_system_input_arguments():
    """
    todo: It is possible to implement node(x=1)._eval() w/o subclassing?

        Yes, if the input arguments are saved in the data-manager, isn't?

            * dm must read-out node._arguments
            * node._eval() must ask dm if an input-value is NOTHING

        ... subclassing is for named configuration-setup which can be
        used (or overwritten) multiple times.
    """
    class MyScale(SystemNode):
        x = Input()
        gain = Input(1)
        args = Input(kind=_VAR_POSITIONAL)
        debug = Input(False, kind=_KEYWORD_ONLY)
        kwargs = Input(kind=_VAR_KEYWORD)
        def __return__(x, gain, offs=0):
            return gain*x + offs

    # set system inputs as positional arguments
    dm = DataManager()
    dm.eval(MyScale(3))
    dm.eval(MyScale(3, 2))  # todo: dm didn't know the changed inputs
    assert dm._data == {
        '__cmd__':      ([0, 5], ['eval(MyScale)', 'eval(MyScale)']),
        'MyScale.x':    ([1], [3]),
        'MyScale.gain': ([2], [1]),
        'MyScale.offs': ([3], [0]),
        'MyScale':      ([4], [3]),
    }

    dm = DataManager()
    dm.eval(MyScale(3, 2))
    assert dm._data == {
        '__cmd__':      ([0], ['eval(MyScale)']),
        'MyScale.x':    ([1], [3]),
        'MyScale.gain': ([2], [2]),
        'MyScale.offs': ([3], [0]),
        'MyScale':      ([4], [6])}

    s = MyScale(3, 2, 1, 0)
    assert s._arguments == {'x': 3, 'gain': 2, 'args': (1, 0)}

    s = MyScale(3, gain=4, debug=True, blah=321)
    assert s._arguments == {'x': 3, 'gain': 4, 'debug': True,
                            'kwargs': {'blah': 321}}


if TODO:
    # set system inputs both as argument and as attribute
    myscale = MyScale(x=3)
    myscale.gain = 5
    dm = DataManager()
    dm.eval(myscale)
    assert dm._data == {'MyScale': ([1], [15])}


def test_002c_first_dependency():
    class MyFunc(SystemNode):
        def __return__(value=321):
            return value

    class MyScale(SystemNode):
        x = Input()
        gain = Input(1)
        def __return__(x, gain, offs=0):
            return gain*x + offs

    # style: interactive
    dm = DataManager()
    myscale = MyScale(x=MyFunc(), gain=2)
    dm.eval(myscale)
    assert dm._data == {
        '__cmd__':         ([0], ['eval(MyScale)']),
        'MyScale.x.value': ([1], [321]),
        'MyScale.x':       ([2], [321]),
        'MyScale.gain':    ([3], [2]),
        'MyScale.offs':    ([4], [0]),
        'MyScale':         ([5], [642]),
    }
    assert list(myscale._iter_pathnames()) == [
            'MyScale.x',
            'MyScale.x.value',
            'MyScale.gain',
            'MyScale.offs',
    ]

    # style: test-bench
    class Test_002c(SystemNode):
        myscale = MyScale(x=MyFunc(), gain=2)
        myscale_nom = 642
        def __return__(myscale, myscale_nom):
            return myscale == myscale_nom

    dm = DataManager()
    dm.eval(Test_002c())
    assert dm._data == {
        '__cmd__':                   ([0], ['eval(Test_002c)']),
        'Test_002c.myscale.x.value': ([1], [321]),
        'Test_002c.myscale.x':       ([2], [321]),
        'Test_002c.myscale.gain':    ([3], [2]),
        'Test_002c.myscale.offs':    ([4], [0]),
        'Test_002c.myscale':         ([5], [642]),
        'Test_002c.myscale_nom':     ([6], [642]),
        'Test_002c':                 ([7], [True]),
    }
    assert list(Test_002c()._namespace()) == [
            'Test_002c.myscale',
            'Test_002c.myscale.x',
            'Test_002c.myscale.x.value',
            'Test_002c.myscale.gain',
            'Test_002c.myscale.offs',
            'Test_002c.myscale_nom',
    ]

if TODO:  # Test 003-a: func-style
    @system
    def myscale(x, gain, offs=0):
        return gain*x + offs
    # not good
    myscale.gain = 2  # append as attrib to myscale-class!
    myscale.x = 5

    # better
    s = myscale(5)
    s.gain = 2

    dm = DataManager()
    dm.eval(s)
    assert dm._data == {
        'myscale.x':    ([1],  [5]),
        'myscale.gain': ([2],  [2]),
        'myscale':      ([3], [10]),
    }


if TODO:  # Test 003-b: func style inputs: all func-args become system inputs!
    @system
    def myscale(x, gain, offs=0):
        return gain*x + offs

    s = myscale(5, 2, 0.5)
    dm = DataManager()
    dm.eval(s)
    assert dm._data == {'myscale': ([1], [10.5])}


def test_004a_multiple_sub_system_instances():
    class MyScale(SystemNode):
        x = Input()
        gain = Input(1)
        def __return__(x, gain, offs=0):
            return gain*x + offs

    class MySystem(SystemNode):
        a = MyScale(5, 2)
        b = MyScale(5, 3)
        def __return__(a, b):
            return a, b

    dm = DataManager()
    dm.eval(MySystem())
    assert dm._data == {
        '__cmd__':         ([0], ['eval(MySystem)']),
        'MySystem.a.x':    ([1], [5]),
        'MySystem.a.gain': ([2], [2]),
        'MySystem.a.offs': ([3], [0]),
        'MySystem.a':      ([4], [10]),
        'MySystem.b.x':    ([5], [5]),
        'MySystem.b.gain': ([6], [3]),
        'MySystem.b.offs': ([7], [0]),
        'MySystem.b':      ([8], [15]),
        'MySystem':        ([9], [(10, 15)]),
    }


@pytest.fixture
def MyScale():
    class MyScale(SystemNode):
        x = Input()
        gain = Input(1)
        def __return__(x, gain, offs=0):
            return gain*x + offs
    return MyScale


def test004b_sub_system_dependency(MyScale):
    class MySystem(SystemNode):
        a = MyScale(5, 2)
        b = MyScale(5, a)
        def __return__(a, b):
            return a, b

    dm = DataManager()
    dm.eval(MySystem())
    assert dm._data == {
        '__cmd__':         ([0], ['eval(MySystem)']),
        'MySystem.a.x':    ([1], [5]),
        'MySystem.a.gain': ([2], [2]),
        'MySystem.a.offs': ([3], [0]),
        'MySystem.a':      ([4], [10]),
        'MySystem.b.x':    ([5], [5]),
        'MySystem.b.offs': ([6], [0]),
        'MySystem.b':      ([7], [50]),
        'MySystem':        ([8], [(10, 50)]),
    }


def test_004c_raise_exception_on_namespace_conflict(MyScale):
    try:
        class MySystem(SystemNode):
            a = MyScale(5, 2)
            b = MyScale(5, 3)
            def __return__(a=b, b=a):
                return a, b
    except ValueError as e:
        assert str(e) == "can not overwrite 'a', it is already an attribute"


@pytest.fixture
def MyGain():
    class MyGain(SystemNode):
        x = Input()
        gain = Input(11)
        coeff = 3

        def mygain(gain, coeff=2):
            return coeff*gain

        def myoffs(coeff=x):
            return 1000*coeff

        def __return__(x, mygain, myoffs):
            return x*mygain + myoffs
    return MyGain


def test_005a_system_with_multiple_functions(MyGain):
    dm = DataManager()
    s = MyGain(3)
    dm.eval(s.mygain)
    dm.eval(s)
    assert dm._data == {
        '__cmd__': ([0, 4], ['eval(MyGain.mygain)', 'eval(MyGain)']),
        'MyGain.gain':         ([1], [11]),
        'MyGain.mygain.coeff': ([2], [2]),
        'MyGain.mygain':       ([3], [22]),  # lazy eval (only for new inputs)
        'MyGain.x':            ([5], [3]),
        'MyGain.myoffs':       ([6], [3000]),
        'MyGain':              ([7], [3066]),
    }


def test_005b_references_to_sub_systems(MyGain):
    class Quad(SystemNode):
        x = Input()

        a = MyGain(3, gain=2)
        b = MyGain(a, gain=2.4)

        def __return__(x, offs=a.mygain):
            return x**2 + offs

        def aux(x, z=b.mygain):
            return x + z + 1

    q = Quad(5)
    assert q.aux.z._parent == q.aux

    dm = DataManager()
    dm.eval(q.aux)
    dm.eval(q)
    assert dm._data == {
        '__cmd__': ([0, 6], ['eval(Quad.aux)', 'eval(Quad)']),
        'Quad.x':              ([1], [5]),
        'Quad.b.gain':         ([2], [2.4]),
        'Quad.b.mygain.coeff': ([3], [2]),
        'Quad.b.mygain':       ([4], [4.8]),
        'Quad.aux':            ([5], [10.8]),
        'Quad.a.gain':         ([7], [2]),
        'Quad.a.mygain.coeff': ([8], [2]),
        'Quad.a.mygain':       ([9], [4]),
        'Quad':                ([10], [29]),
    }



def test_006_test_nested_references():
    class SUM(SystemNode):
        a = Input(2)
        b = Input(3)
        def __return__(a, b):
            return a + b

    class A(SystemNode):
        inp = Input(0)
        eingang = inp

        x = SUM(1, 2)
        y = SUM(7, 8)

        def __return__(eingang, x, y):
            return eingang, x, y

    class B(SystemNode):
        g = A(10)
        h = A(20)

        g_ref = g
        h_ref = h
        h_x = h.x

        def __return__(g, h_y=h.y):
            return g, h_y

    b = B()
    dm = DataManager()
    dm.eval(b)
    assert dm._data == {
        '__cmd__': ([0], ['eval(B)']),
        'B.g.inp': ([1], [10]),
        'B.g.x.a': ([2], [1]),
        'B.g.x.b': ([3], [2]),
        'B.g.x':   ([4], [3]),
        'B.g.y.a': ([5], [7]),
        'B.g.y.b': ([6], [8]),
        'B.g.y':   ([7], [15]),
        'B.g':     ([8], [(10, 3, 15)]),
        'B.h.y.a': ([9], [7]),
        'B.h.y.b': ([10], [8]),
        'B.h.y':   ([11], [15]),
        'B':       ([12], [((10, 3, 15), 15)]),
    }
    # contruct the complete namespace from b - especially the input parameters!
    #   these are important for plots
    assert list(b._namespace()) == [
        'B.g',
        'B.g.inp',
        'B.g.eingang',
        'B.g.x',
        'B.g.x.a',
        'B.g.x.b',
        'B.g.y',
        'B.g.y.a',
        'B.g.y.b',
        'B.h',
        'B.h.inp',
        'B.h.eingang',
        'B.h.x',
        'B.h.x.a',
        'B.h.x.b',
        'B.h.y',
        'B.h.y.a',
        'B.h.y.b',
        'B.g_ref',
        'B.h_ref',
        'B.h_x',
        'B.h_y',
    ]


def test_007_subclassing():
    class MyScale(SystemNode):
        x = Input()
        gain = Input(1)
        def __return__(x, gain, offs=0):
            return gain*x + offs

    class MyDouble(MyScale):  # very important, see LinSweep!
        gain = 2


    s = MyDouble(3)
    dm = DataManager()
    dm.eval(s)
    assert dm._data == {
        '__cmd__':       ([0], ['eval(MyDouble)']),
        'MyDouble.x':    ([1], [3]),
        'MyDouble.gain': ([2], [2]),
        'MyDouble.offs': ([3], [0]),
        'MyDouble':      ([4], [6]),
    }


