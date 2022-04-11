import unittest


class ImportTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_import(self):
        try:
            import vdb_to_numpy
        except ImportError:
            self.fail("vdb_to_numpy not properly installed, please run `make`")

    def test_import_pybind(self):
        try:
            import vdb_to_numpy.pybind
        except ImportError:
            self.fail("vdb_to_numpy not properly installed, please run `make`")
