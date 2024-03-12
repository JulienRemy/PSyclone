from psyclone.psyir.nodes import IntrinsicCall

from psyclone.autodiff.psyir.nodes import ADCall


# NOTE: to be decided:
# - does this inherit from ADCall (if only subroutines) or ADNode?
class ADIntrinsicCall(IntrinsicCall, ADCall):
    pass
