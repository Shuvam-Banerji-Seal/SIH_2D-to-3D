"""Allow ``python -m drone3d``."""

import sys

from drone3d.cli import main

if __name__ == "__main__":
    sys.exit(main())
