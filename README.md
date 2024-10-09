FLooPy
======
> *Flow-based Loops with Python*

*FLooPy* is a **tester-agnostic sequencer** for **hardware-testing.** It has a **low-code** approach in which the same test can be performed within different **test-environments.**

#### Tests
* ... can organized in **hierarchical** layers with (sub-) tasks
* Every variable
    - ... has an unique name via **namespace**
    - ... is stored as a **time-signal** with timestamps
    - ... can be **sub-classed**
* Data dependencies between functions are resolved by a graph-based approach
* Auto-Save for loop-recovery and post-processing (experimental)
* Can handle **non-reentrant** tests by an extra extension (unpublished)

#### Loops
* Flexible **loop configuration** (nested, zipped, concatenated)
* Standard loops: `loop_items`, `loop_lin`, `loop_log`, `loop_bisect`
* Feedback for HiL possible

---

## Test-Structure by Example

For a Getting-Started example look into the *[FLooPy-Tutorial](./tutorial.ipynb)*.

Suppose we want to test if the internal resistance of an battery `rbat` is inside the test-limits:

* The test-environment is abstracted by the namespace of the `dut` input and must be specified later, just before running the test-case.
* The loop over the load-current can be either configured locally inside the test-case or later, just before running the test-case.

#### Test-Case


```python
import floopy as fly


class Test_Rbat_Charged(fly.Task):
    dut = fly.Input()  # object to stimulate device-under-test and measure response
   
    i_load = fly.Input(min=0, max=1.5, unit='A', default=fly.loop_lin(num=5))
    
    def task(dut, i_load):
        dut.current = i_load
        return dut.voltage

    def final_fit(i=fly.Squeeze(task.i_load), v=fly.Squeeze(task)):
        import numpy as np
        return np.polyfit(i, v, deg=1)
        
    @fly.Output(min=0, ltl=0.2, utl=2, max=10, unit='ohm')
    def rbat(coeffs=final_fit):
        return coeffs[0]
```

#### Test-Plan

Now, the test-case `Test_Rbat_Charged` can be used with different loop-configurations and within different test-environments. For demonstration we just use the simple resistor equation as a simulation-environment.


```python
class TestPlan_BatterCheck(fly.Task):
    def dut():
        class Dut:
            @property
            def voltage(self):
                resistance = 0.8  # ohm
                return resistance * self.current
        return Dut()

    test_rbat_charged = Test_Rbat_Charged(dut, i_load=fly.loop_log(0.1, 1, num=3))
```


```python
dm = fly.DataManager()
dm.run_log(TestPlan_BatterCheck)
```

Output:

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[22:35:02]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_rbat_charged.task</span> | test_rbat_charged.<span style="color: #cb4b16; text-decoration-color: #cb4b16">i_load</span>=0.1
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">22:35:02</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 0.08000000000000002
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[22:35:02]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_rbat_charged.task</span> | test_rbat_charged.<span style="color: #cb4b16; text-decoration-color: #cb4b16">i_load</span>=0.31622776601683794
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">22:35:02</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 0.2529822128134704
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[22:35:02]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_rbat_charged.task</span> | test_rbat_charged.<span style="color: #cb4b16; text-decoration-color: #cb4b16">i_load</span>=1.0
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">22:35:02</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 0.8
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[22:35:02]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_rbat_charged.final_fit</span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">22:35:02</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = <span style="color: #268bd2; text-decoration-color: #268bd2">array</span><span style="font-weight: bold">([</span>8.00000000e-01, 1.59768407e-16<span style="font-weight: bold">])</span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[22:35:02]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_rbat_charged.rbat</span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">22:35:02</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = 0.7999999999999999
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[22:35:02]</span>  <span style="color: #268bd2; text-decoration-color: #268bd2; font-weight: bold">test_rbat_charged.rbat.check</span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">[</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">22:35:02</span><span style="color: #7fbfbf; text-decoration-color: #7fbfbf; font-weight: bold">]</span>      = <span style="color: #859900; text-decoration-color: #859900; font-weight: bold; font-style: italic">True</span>

Summery: <span style="color: #859900; text-decoration-color: #859900">1 passed</span>

File: dm/dm_2024-10-08_22:35:02.json
</pre>



For a real measurement we just need to replace the `dut` function with something like

```python
def dut():
    import instruments
    dut = instruments.PowerSupply()
    return dut
```

assuming an equal namespace

```python
dut.voltage
dut.current
```

with an equal behavior.

*Further information can be found in the **[FLooPy-Tutorial](./tutorial.ipynb)** which is more detailed.*

---

## Install


Simply do either

    pip install floopy

or download the repository for a more recent version, change into
the `floopy` folder and

    pip install . --user

If necessary you can run the tests with

    python -m pytest

## Licence

LoopyPlot is licenced under GPL 3.
