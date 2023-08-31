# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2019, Science and Technology Facilities Council
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
# Author: J. Remy, Université Grenoble Alpes, Inria

"""PSyclone automatic differentiation transformations module.
"""

from psyclone.autodiff.transformations.ad_trans import ADTrans
from psyclone.autodiff.transformations.ad_container_trans import (
    ADContainerTrans,
)
from psyclone.autodiff.transformations.ad_routine_trans import ADRoutineTrans
from psyclone.autodiff.transformations.ad_element_trans import ADElementTrans
from psyclone.autodiff.transformations.ad_assignment_trans import (
    ADAssignmentTrans,
)
from psyclone.autodiff.transformations.ad_operation_trans import (
    ADOperationTrans,
)
from psyclone.autodiff.transformations.ad_call_trans import ADCallTrans

from psyclone.autodiff.transformations.reverse_mode.ad_reverse_container_trans import (
    ADReverseContainerTrans,
)
from psyclone.autodiff.transformations.reverse_mode.ad_reverse_routine_trans import (
    ADReverseRoutineTrans,
)
from psyclone.autodiff.transformations.reverse_mode.ad_reverse_assignment_trans import (
    ADReverseAssignmentTrans,
)
from psyclone.autodiff.transformations.reverse_mode.ad_reverse_operation_trans import (
    ADReverseOperationTrans,
)
from psyclone.autodiff.transformations.reverse_mode.ad_reverse_call_trans import (
    ADReverseCallTrans,
)

from psyclone.autodiff.transformations.forward_mode.ad_forward_container_trans import (
    ADForwardContainerTrans,
)
from psyclone.autodiff.transformations.forward_mode.ad_forward_routine_trans import (
    ADForwardRoutineTrans,
)
from psyclone.autodiff.transformations.forward_mode.ad_forward_assignment_trans import (
    ADForwardAssignmentTrans,
)
from psyclone.autodiff.transformations.forward_mode.ad_forward_operation_trans import (
    ADForwardOperationTrans,
)
from psyclone.autodiff.transformations.forward_mode.ad_forward_call_trans import (
    ADForwardCallTrans,
)

# The entities in the __all__ list are made available to import directly from
# this package e.g. 'from psyclone.autodiff.transformations import ADContainerTrans'
__all__ = [
    # Abstract transformations
    "ADTrans",
    "ADContainerTrans",
    "ADRoutineTrans",
    "ADElementTrans",
    "ADAssignmentTrans",
    "ADOperationTrans",
    "ADCallTrans",
    
    # Reverse-mode transformations
    "ADReverseContainerTrans",
    "ADReverseRoutineTrans",
    "ADReverseAssignmentTrans",
    "ADReverseOperationTrans",
    "ADReverseCallTrans",

    # Forward-mode transformations
    "ADForwardContainerTrans",
    "ADForwardRoutineTrans",
    "ADForwardAssignmentTrans",
    "ADForwardOperationTrans",
    "ADForwardCallTrans",
]
