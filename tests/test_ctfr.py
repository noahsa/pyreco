import ctfr.ctfr

def test_get_divisors():
    output = ctfr.ctfr.get_divisors(12)


    expected_output = [1, 2, 3, 4, 6, 12]

    assert (list(output) == expected_output)


def test_thf_tols():
    output = ctfr.ctfr.thf_tools(12)
