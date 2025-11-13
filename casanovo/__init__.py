try:
 from .version import _get_version
 __version__ = _get_version()
except:
 __version__ = "5.0.0"
