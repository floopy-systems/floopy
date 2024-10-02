FLooPy
======
> Flow-based Loops with Python

A Python library Manage tasks with loops in a clean and structured way.

#### Tasks
* Tasks are plain python functions
* Every task-function and variable ...
    - has an unique name via **namespace** (e.g. `ErrorCheck.test_error.button`)
    - is stored as a **time-signal** (with timestamps)
    - can be **sub-classed**
* Tasks can organized into **hierarchical** sub-tasks
* Data dependencies are resolved by a graph-based approach

#### Loops
* Flexible loop configuration with minimal care for data-structure
* Loop structures: nested (default), concatenated, zipped
* Loop types: `loop_items`, `loop_lin`, `loop_log`, `loop_bisect`
* Loops can have feedback from the task (e.g. `loop_fmin`)

Motivation
----------

In the field of engineering science one common experiment task is to
explore the behavior of a black-box under different inputs. The
black-box can be a pure mathematical function, a numerical algorithm,
the results of a complex simulation or even an experimental measurement.
In many cases the input variation are done by nested for-loops.

While a nested for-loop iteration is simple to code the data management
can be become quite complicated. This is even true when you want

1.  to quickly change the *loop configuration* (nested loops vs. zipped
  loops)
2.  to define *data dependencies* between different experiments
3.  to have an *error recovery* of the loop state because each
  iteration step takes a reasonable amount of time
4.  to *reload* the measurement data later for postprocessing tasks
5.  to write *readable code* which can be shared for collaboration

Especially the last point requires you to split the specific part of
an experiment from its administration (remaining points above). A very
natural way of splitting is to use a function. Everything inside the
function describes the specific experiment. The function arguments and
return values are used for the administration of the experiment.

Use LoopyTask for Testing
-------------------------
* Use top-level-task for configuration
* (HV-) loop sections: **setup, task/test, teardown, final**
* Handling **non-reentrant** test-functions by an virtual (dut-) state-graph
* Auto-Save for offline-postprocessing or loop-recovery (experimental)


```python
from loopytask import Task, Input, loop, loop_lin, Squeeze
from loopytask import DataManager
```


```python
class MyTestCase(Task):
    dut = Input()
    x = Input(0)  # can be a constant value or a loop

    def setup(dut):
        ...

    def task_measure(dut, x):
        # write to dut-object and read-out measurement results
        return x

    def teardown(dut):
        ...

    def final_processing(values=Squeeze(task_measure)):
        return values.mean()  # do some scaling/fitting/...

    def __return__(param=final_processing):
        return 3.5 < param < 5  # check if parameter is in limits (assert)
```


```python
class TestPlan(Task):
    dut = 'object to stimulate device-under-test and measure response'

    test_value = MyTestCase(dut, x=3.3)

    test_sweep = MyTestCase(dut, x=loop(loop_lin(1, 5, num=3),
                                        7,
                                        loop_lin(5, 1, num=3)))
```


```python
tp = TestPlan()
dm = DataManager()
```


```python
dm.run_live(tp)
```
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    ╭─── TestPlan ───╮
    │  ✔  test_value │
    │  ✔  test_sweep │
    ╰────────────────╯
    Summery: <span style="color: #dc322f; text-decoration-color: #dc322f">1 failed</span>, <span style="color: #859900; text-decoration-color: #859900">1 passed</span>
    File: dm/dm_2024-07-31_23:07:31_TestPlan.json
</pre>


```python
dm.run(tp)
```
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_value.setup</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_value.task_measure</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 3.3
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_value.teardown</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_value.final_processing</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 3.3
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_value</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = <span style="color: #dc322f; text-decoration-color: #dc322f; font-weight: bold; font-style: italic">False</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.setup</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.task_measure</span> | test_sweep.<span style="color: #cb4b16; text-decoration-color: #cb4b16">x</span>=1.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 1.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.task_measure</span> | test_sweep.<span style="color: #cb4b16; text-decoration-color: #cb4b16">x</span>=3.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 3.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.task_measure</span> | test_sweep.<span style="color: #cb4b16; text-decoration-color: #cb4b16">x</span>=5.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 5.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.task_measure</span> | test_sweep.<span style="color: #cb4b16; text-decoration-color: #cb4b16">x</span>=7
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 7
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.task_measure</span> | test_sweep.<span style="color: #cb4b16; text-decoration-color: #cb4b16">x</span>=5.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 5.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.task_measure</span> | test_sweep.<span style="color: #cb4b16; text-decoration-color: #cb4b16">x</span>=3.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 3.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.task_measure</span> | test_sweep.<span style="color: #cb4b16; text-decoration-color: #cb4b16">x</span>=1.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 1.0
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.teardown</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep.final_processing</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 3.5714285714285716
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[13:55:22]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_sweep</span>
    <span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">13:55:22</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = <span style="color: #859900; text-decoration-color: #859900; font-weight: bold; font-style: italic">True</span>

    Summery: <span style="color: #dc322f; text-decoration-color: #dc322f">1 failed</span>, <span style="color: #859900; text-decoration-color: #859900">1 passed</span>
    File: dm/dm_2024-07-31_16:48:53_TestPlan.json
</pre>

Install
-------

Simply download the repository, change into
the folder `LoopyTask` and

    pip install . --user

If necessary you can run the tests with

    python -m pytest

Licence
-------

LoopyPlot is licenced under GPL 3.
