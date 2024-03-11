# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2024, Science and Technology Facilities Council
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

"""PSyclone automatic differentiation PSyIR nodes derived classes.
"""

from psyclone.autodiff.psyir.nodes.ad_node import ADNode
from psyclone.autodiff.psyir.nodes.ad_datanode import ADDataNode
from psyclone.autodiff.psyir.nodes.ad_reference import ADReference
from psyclone.autodiff.psyir.nodes.ad_operation import ADOperation, ADUnaryOperation, ADBinaryOperation
from psyclone.autodiff.psyir.nodes.ad_literal import ADLiteral
from psyclone.autodiff.psyir.nodes.ad_assignment import ADAssignment
from psyclone.autodiff.psyir.nodes.ad_call import ADCall
from psyclone.autodiff.psyir.nodes.ad_intrinsic_call import ADIntrinsicCall
from psyclone.autodiff.psyir.nodes.ad_if_block import ADIfBlock
from psyclone.autodiff.psyir.nodes.ad_loop import ADLoop
from psyclone.autodiff.psyir.nodes.ad_schedule import ADSchedule
from psyclone.autodiff.psyir.nodes.ad_routine import ADRoutine
from psyclone.autodiff.psyir.nodes.ad_container import ADContainer


# The entities in the __all__ list are made available to import directly from
# this package
# e.g. 'from psyclone.autodiff.psyir.symbols import ADVariableSymbol'
__all__ = [
    "ADNode",
    "ADDataNode",
    "ADReference",
    "ADOperation",
    "ADUnaryOperation",
    "ADBinaryOperation",
    "ADLiteral",
    "ADAssignment",
    "ADCall",
    "ADIntrinsicCall",
    "ADSchedule",
    "ADIfBlock",
    "ADLoop",
    "ADRoutine",
    "ADContainer",
]
