from psyclone.psyir.nodes import Call

from psyclone.autodiff.psyir.nodes import ADNode

# NOTE: to be decided:
# - are these subroutine calls only?
# - does ADIntrinsicCall inherit from this or ADNode?
class ADCall(Call, ADNode):
    pass