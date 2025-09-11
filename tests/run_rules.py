import inspect
import sys

import test_rules as tr


def main():
    tests = [name for name, fn in inspect.getmembers(tr, inspect.isfunction) if name.startswith("test_")]
    failures = 0
    for name in tests:
        try:
            getattr(tr, name)()
            print(f"PASS: {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {name}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR: {name}: {e}")
    print(f"Summary: {len(tests)-failures} passed, {failures} failed, total {len(tests)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
