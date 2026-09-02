from cfb_model.util import haversine_miles, parse_efficiency, parse_possession, shrink


def test_parse_efficiency_rate():
    made, att, rate = parse_efficiency("3-7")
    assert made == 3
    assert att == 7
    assert abs(rate - 3 / 7) < 1e-9


def test_parse_possession_clock():
    assert parse_possession("35:42") == 35 * 60 + 42
    assert parse_possession(100) == 100


def test_shrinkage_weights():
    # n=0 -> current (or prior if current missing handled by caller)
    assert shrink(10, 0, 0, k=4) == 0
    # n=4, k=4 -> 50/50
    assert shrink(10, 0, 4, k=4) == 5
    # large n trusts current
    assert abs(shrink(10, 0, 100, k=4) - 10) < 0.5


def test_haversine_zero_and_nonzero():
    assert haversine_miles(40.0, -75.0, 40.0, -75.0) == 0
    miles = haversine_miles(33.75, -84.39, 42.28, -83.75)  # Atlanta to Ann Arbor
    assert 500 < miles < 800
