try:
    from collections import namedtuple
except ImportError:
    from collections.abc import namedtuple

SearchNode = namedtuple('SearchNode', ['cost', 'parent'])
