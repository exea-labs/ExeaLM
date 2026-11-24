import shutil
import tempfile

import pytest


@pytest.fixture
def short_cache():
    """A cache directory with a short path. The datasets library builds a lock file name from the whole cache path, which
    goes past the 260 character Windows limit inside a pytest temp directory."""
    path = tempfile.mkdtemp(prefix="xlm")
    yield path
    shutil.rmtree(path, ignore_errors=True)
