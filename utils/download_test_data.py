#!/usr/bin/env python3
import tests.test_data

print("Downloading all the test_data...")
[
    cls()  # call default constructor fo TestData objects
    for name, cls in tests.test_data.__dict__.items()
    if isinstance(cls, type) and name not in "abstractclassmethod"
]
