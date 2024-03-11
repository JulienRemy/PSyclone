from psyclone.psyir.nodes import Container

from psyclone.autodiff.psyir.nodes import ADNode

# NOTE: this could use an ADScopingNode parent
# iff the symbol table needs more info for AD?
class ADContainer(Container, ADNode):
    pass