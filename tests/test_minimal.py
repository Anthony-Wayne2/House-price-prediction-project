"""
Minimal test that should always pass in CI.
"""
def test_always_passes():
    """This test always passes."""
    assert True

def test_python_version():
    """Check Python version."""
    import sys
    assert sys.version_info >= (3, 8)
