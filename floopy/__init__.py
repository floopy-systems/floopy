from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('floopy')
except PackageNotFoundError:
    __version__ = 'unknown'


from .basesystem import (
    Task,
    Input,
    Output,
    Squeeze,
    SystemNode,
    _ValueNode,
    _TupleNode,
    RefNode,
    SelfRefNode,
    StateNode,
    Sum,
    ConditionalNode,
    _LoopSystem,
    CountingLoop,
)
from .pathnode import _PathNode
from .basesystem import CountingLoop as loop_count
from .basesystem import ConcatLoop as loop
loop_items = loop
from .basesystem import LoopLin as loop_lin
from .basesystem import LoopLog as loop_log
from .basesystem import LoopLog as loop_log
from .basesystem import LoopBisect as loop_bisect

from .datamanager import DataManager
