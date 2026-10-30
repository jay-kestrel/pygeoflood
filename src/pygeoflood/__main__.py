"""python -m pygeoflood — equivalent to the `pygeoflood` console script.

    python -m pygeoflood version
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
