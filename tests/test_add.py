from predictCO2.add import add


def test_add():
    res = add(1, 1)
    assert res == 2