import pytest
from floopy.pathnode import _PathNode, _update_reorder


@pytest.fixture
def c():
    class A(_PathNode):
        class a(_PathNode):
            pass
        a = a()


    class B(_PathNode):
        class b(_PathNode):
            pass
        b = b()

        class a1(A):
            pass
        a1 = a1()

        class a2(A):
            # overwrite attribute A.a
            class a(_PathNode):
                pass
            a = a()
        a2 = a2()


    class C(B):
        class c(_PathNode):
            pass
        c = c()

    return C()


def test_root(c):
    assert c.a1.a._root == c


def test_path(c):
    assert c.a2.a._path == (c, c.a2, c.a2.a)


def test_pathnames(c):
    assert c.c._pathname    == 'C.c'
    assert c.b._pathname    == 'C.b'
    assert c.a1._pathname   == 'C.a1'
    assert c.a1.a._pathname == 'C.a1.a'
    assert c.a2._pathname   == 'C.a2'
    assert c.a2.a._pathname == 'C.a2.a'


def test_namespace(c):
    # test pathnames on attributes overwritten by subclasses
    assert list(c.a2._namespace()) == ['C.a2.a']
    # test all pathnames of namespace
    assert list(c._namespace()) == [
        'C.b',
        'C.a1',
        'C.a1.a',
        'C.a2',
        'C.a2.a',
        'C.c',
    ]
    # test all pathnames
    assert list(c._namespace()) == list(c._iter_pathnames())


def test_get_parent(c):
    assert c.a1.a._get_parent(0) == c.a1.a
    assert c.a1.a._get_parent(1) == c.a1
    assert c.a1.a._get_parent(2) == c
    assert c.a1.a._get_parent() == c.a1.a._parent == c.a1


def test_get_subnode(c):
    assert c._get_subnode( ('a1', 'a') ) == c.a1.a


def test_preserve_order_of_overwritten_attributes():
    class D(_PathNode):
        class a(_PathNode):
            pass
        a = a()
        class b(_PathNode):
            pass
        b = b()
        class c(_PathNode):
            pass
        c = c()

    class d(D):
        # overwrite attribute D.c and then D.c
        class c(_PathNode):
            pass
        c = c()
        class b(_PathNode):
            pass
        b = b()

    # test if 'c' and 'b' are swapped
    assert list(D()._namespace()) == ['D.a', 'D.b', 'D.c']
    assert list(d()._namespace()) == ['d.a', 'd.c', 'd.b']


def test_update_reorder_1():
    d1 = {}
    d2 = {'a': 1, 'b': 2, 'c': 3}
    d12 = _update_reorder(d1, d2)
    assert d12 == {'a': 1, 'b': 2, 'c': 3}


def test_update_reorder_2():
    d1 = {'a': 1, 'b': 2, 'c': 3}
    d2 = {'a': 10, 'b': 20, 'c': 30}
    d12 = _update_reorder(d1, d2)
    assert d12 == {'a': 10, 'b': 20, 'c': 30}


def test_update_reorder_3():
    d1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    d2 = {'c': 10, 'b': 20, 'zz': 99, 'a': 30, 'e': 50}
    d12 = _update_reorder(d1, d2)
    assert d12 == {'c': 10, 'b': 20, 'zz': 99, 'a': 30, 'd': 4, 'e': 50}


def test_update_reorder_4():
    d1 = {'start': 1, 'x': 2, 'middle': 3, 'y': 3, 'end': 4}
    d2 = {'y': 222, 'x': 333,}
    d12 = _update_reorder(d1, d2)
    assert d12 == {'start': 1, 'y': 222, 'middle': 3, 'x': 333, 'end': 4}


def test_update_reorder_5():
    d1 = {'start': 1, 'x': 2, 'middle': 3, 'y': 3, 'end': 4}
    d2 = {'a': 111, 'y': 222, 'x': 333,}
    d12 = _update_reorder(d1, d2)
    assert d12 == {'start': 1, 'a':111, 'y': 222, 'middle': 3, 'x': 333, 'end': 4}


def test_update_reorder_6():
    d1 = {'a': 1, 'b': 2, 'c': 3, 'd': 3, 'e': 4, 'f': 5}
    d2 = {'d': 111, 'e': 222, 'b': 333}
    d12 = _update_reorder(d1, d2)
    assert d12 == {'a': 1, 'd': 111, 'c': 3, 'e': 222, 'b': 333, 'f': 5}


def test_update_reorder_7():
    d1 = {'offs': 1, 'x': 2, 'y': 3, 'numbers': 4}
    d2 = {'a': 111, 'y': 222, 'x': 333, 'z': 444}
    d12 = _update_reorder(d1, d2)
    assert d12 == {'offs': 1, 'a': 111, 'y': 222, 'x': 333, 'z': 444, 'numbers': 4}

