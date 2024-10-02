from floopy.basesystem import Task, Input, Squeeze, ConcatLoop, LoopLin
from floopy import DataManager
import floopy as fly
import numpy as np


def test_squeezing_prev_task():
    class TaskQuad(fly.Task):
        x = fly.Input()

        def task(x):
            return x**2


    class TestPlan(fly.Task):
        ### task section ###
        id = fly.loop(1, 10)

        task_quad = TaskQuad(x = fly.loop_lin(2, 5))

        def task_sum(id, y = fly.Squeeze(task_quad.task)):
            return id * y

        ### final section ###
        def final_sum(y = fly.Squeeze(task_quad.task)):
            return y.sum()


    tp = TestPlan()
    dm = fly.DataManager()
    dm.run(tp)

    _, vals = dm.read_raw(tp.task_sum.y)
    assert (vals[0] == np.array([ 4,  9, 16, 25])).all()
    assert (vals[1] == np.array([ 4,  9, 16, 25])).all()

    _, vals = dm.read_raw(tp.task_sum)
    assert (vals[0] == np.array([  4,  9,  16,  25])).all()
    assert (vals[1] == np.array([ 40, 90, 160, 250])).all()

    _, vals = dm.read_raw(tp.final_sum.y)
    assert (vals == np.array([ 4,  9, 16, 25,  4,  9, 16, 25])).all()
