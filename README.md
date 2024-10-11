<h3 align="center">
    <img width="66%" alt="FLooPy" src="https://raw.githubusercontent.com/floopy-systems/floopy/refs/heads/main/logo-floopy.svg">
    <br><em>Tester-Agnostic Sequencer for Hardware-Testing in Python</em>
    <hr />
</h3>

### Tests
* ... can performed within different **test-environments**
* ... can organized in **hierarchical** layers with (sub-) tasks
* ... can have `setup`, `task/test`, `teardown`, `final` and output functions
* Every variable
    - ... has an unique **namespace**
    - ... is stored as a **time-signal** with timestamps
    - ... can be **sub-classed**
* Data **dependencies** between functions are **resolved** by a graph-based approach
* Auto-Save for loop-recovery and post-processing (experimental)
* Can handle **non-reentrant** tests by an extra extension (unpublished)

### Loops
* Flexible **loop configuration** (nested, zipped, concatenated)
* Standard loops: `loop_items`, `loop_lin`, `loop_log`, `loop_bisect`
* Feedback for **HiL** possible

## Getting Started

Suppose we want to test if the internal resistance of an battery `rbat` is inside the test-limits:

* The test-environment is abstracted by the namespace of the `dut` input and must be specified later, just before running the test-case.
* The loop over the load current `i_load` can be either configured locally inside the test-case or later, just before running the test-case.

### Test-Case


```python
import floopy as fly


class Test_Rbat_Charged(fly.Task):
    dut = fly.Input()  # object to stimulate device-under-test and measure response
   
    i_load = fly.Input(min=0, max=1.5, unit='A', default=fly.loop_lin(num=3))
    
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

### Test-Plan

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

    test_rbat_charged = Test_Rbat_Charged(dut, i_load=fly.loop_log(0.1, 1, num=5))
```

Note, that we changed the default loop-configuration of `i_load` to a logarithmic scaling.

In order to perform the Test-Plan we need a `DataManager` saving all loop-states and measurement results as time-signals in a json-file.

```python
>>> dm = fly.DataManager()
>>> dm.run_live(TestPlan_BatterCheck)

╭─ TestPlan_BatterCheck ─╮
│  ✔  test_rbat_charged  │
╰────────────────────────╯
File: dm/dm_2024-10-10_16:00:29.json
```

The test results and all variables can be analyzed via pandas-tables.

```python
>>> dm.read_task(TestPlan_BatterCheck)
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th colspan="2" halign="left">test_rbat_charged</th>
    </tr>
    <tr>
      <th>rbat</th>
      <th>check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.8</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

```python
>>> tp = TestPlan_BatterCheck()
>>> dm.read_task(tp.test_rbat_charged.task)
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>i_load</th>
      <th>__return__</th>
    </tr>
    <tr>
      <th>TestPlan_BatterCheck.test_rbat_charged.i_load</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.100000</th>
      <td>0.100000</td>
      <td>0.080000</td>
    </tr>
    <tr>
      <th>0.177828</th>
      <td>0.177828</td>
      <td>0.142262</td>
    </tr>
    <tr>
      <th>0.316228</th>
      <td>0.316228</td>
      <td>0.252982</td>
    </tr>
    <tr>
      <th>0.562341</th>
      <td>0.562341</td>
      <td>0.449873</td>
    </tr>
    <tr>
      <th>1.000000</th>
      <td>1.000000</td>
      <td>0.800000</td>
    </tr>
  </tbody>
</table>
</div>

For a real measurement we leave the test-case unchanged and just need to replace the `dut` function in the test-plan with something like

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
