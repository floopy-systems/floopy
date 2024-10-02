from floopy import Task, Input, Output, DataManager, loop, Squeeze
import pandas as pd
import pytest


def _test_output_read_task_3():
    class MyTask(Task):
        x = Input()
        offs = Input(0)

        @Output(ltl=5, utl=7)
        def y(x):
            return 2*x

    class TaskPlan(Task):
        offs = loop(0, 100)

        test_odd = MyTask(x=loop(1, 3, 5), offs=offs)
        test_even = MyTask(x=loop(2, 4, 6), offs=offs)

    # todo: need to squeeze the inner loop (MyTask.x)

    tp = TaskPlan()
    dm = DataManager()
    dm.run(tp)
    df = dm.read_task(tp)
    assert repr(df).split('\n') == []


def test_output_read_task_2():
    class MyTask(Task):
        x = Input()
        offs = Input(0)

        @Output(ltl=5, utl=7)
        def y(x):
            return 2*x

    class TaskPlan(Task):
        offs = loop(0, 100)

        test_odd = MyTask(x=3, offs=offs)
        test_even = MyTask(x=4, offs=offs)

    tp = TaskPlan()
    dm = DataManager()
    dm.run(tp)
    df = dm.read_task(tp)
    assert repr(df).split('\n') == [
        '     test_odd       test_even       ',
        '            y check         y  check',
        'offs                                ',
        '0           6  True         8  False',
        '100         6  True         8  False']


def test_output_read_task_1():
    class MyTask(Task):
        x = Input()
        offs = Input(0)

        def task(x):
            return x

        @Output(ltl=5, utl=7)
        def y(x=Squeeze(task)):
            return 2*x.mean()

    class TaskPlan(Task):
        offs = loop(0, 100)

        test_odd = MyTask(x=loop(1, 3, 5), offs=offs)
        test_even = MyTask(x=loop(2, 4, 6), offs=offs)

    tp = TaskPlan()
    dm = DataManager()
    dm.run(tp)
    df = dm.read_task(tp)
    assert repr(df).split('\n') == [
        '     test_odd       test_even       ',
        '            y check         y  check',
        'offs                                ',
        '0         6.0  True       8.0  False',
        '100       6.0  True       8.0  False']

    df = dm.read_task(tp.test_even)
    assert repr(df).split('\n') == [
        '              test_even       ',
        '                      y  check',
        'TaskPlan.offs                 ',
        '0                   8.0  False',
        '100                 8.0  False']

    df = dm.read_task(tp.test_even.y)
    assert repr(df).split('\n') == [
        '                       x  __return__',
        'TaskPlan.offs                       ',
        '0              [2, 4, 6]         8.0',
        '100            [2, 4, 6]         8.0']

    df = dm.read_task(tp.test_even.task)
    assert repr(df).split('\n') == [
        '                                    x  __return__',
        'TaskPlan.offs TaskPlan.test_even.x               ',
        '0             2                     2           2',
        '              4                     4           4',
        '              6                     6           6',
        '100           2                     2           2',
        '              4                     4           4',
        '              6                     6           6']


def test_output_fields():
    class MyTask(Task):
        x = 5
        y = Output(min=0, ltl=2, nom=x, utl=8, max=9, unit='V', fmt=':.2')

    tp = MyTask()
    dm = DataManager()

    assert tp.y.min._value == 0
    assert tp.y.ltl._value == 2
    assert tp.y.nom._eval(dm) == 5
    assert tp.y.utl._value == 8
    assert tp.y.unit._value == 'V'
    assert tp.y.fmt._value == ':.2'


def test_output_decorator():
    class MyTask(Task):
        x = 5
        y = Output(min=0, ltl=2, nom=x, utl=8, max=9, unit='V', fmt=':.2')
        z = 2

        @Output
        def y(nom, z):
            return 2*nom + z

    tp = MyTask()
    dm = DataManager()

    assert tp.y._eval(dm) == 12


def test_output_decorator_overwrite():
    class MyTask(Task):
        x = 5
        y = Output(min=0, ltl=2, nom=x, utl=8, max=9, unit='V', fmt=':.2')
        z = 2

        @Output(nom=7)
        def y(nom, z):
            return 2*nom + z

    tp = MyTask()
    dm = DataManager()

    assert tp.y._eval(dm) == 16


def test_output_decorator_only():
    class MyTask(Task):
        x = 5

        @Output(nom=8)
        def y(nom, x):
            return 2*nom + x

    tp = MyTask()
    dm = DataManager()

    assert tp.y._eval(dm) == 21


def test_output_check():
    class MyTask(Task):
        x = 5

        @Output(ltl=5, utl=30)
        def quad_true(x):
            return x**2

        @Output(ltl=5, utl=20)
        def quad_false(x):
            return x**2

        @Output
        def quad_None(x):
            return x**2

    tp = MyTask()
    dm = DataManager()

    assert tp.quad_true.check._eval(dm) == True
    assert tp.quad_false.check._eval(dm) == False
    assert tp.quad_None.check._eval(dm) is None


def test_output_order():
    with pytest.raises(ValueError,
                       match='\\(ltl = 7\\) \\<= \\(nom = 5\\)',
    ):
        class MyTask(Task):
            y = Output(ltl=7, nom=5)
