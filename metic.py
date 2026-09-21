"""Backward-compatible alias for workflows that use the historical typo `metic.py`."""
from metric import *  # noqa: F401,F403
from metric import main


if __name__ == "__main__":
    main()

