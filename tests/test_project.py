import unittest

import shafieedocai


class ProjectTests(unittest.TestCase):
    def test_package_version_exists(self):
        self.assertEqual(shafieedocai.__version__, "0.1.0")

    def test_package_name(self):
        self.assertEqual(shafieedocai.__name__, "shafieedocai")


if __name__ == "__main__":
    unittest.main()