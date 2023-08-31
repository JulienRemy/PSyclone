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

"""PSyclone automatic differentiation module.
"""

from psyclone.autodiff.ad_reversal_schedule import (
    ADReversalSchedule,
    ADSplitReversalSchedule,
    ADJointReversalSchedule,
    ADLinkReversalSchedule,
)
from psyclone.autodiff.utils import (
    one,
    zero,
    minus,
    inverse,
    power,
    sqrt,
    log,
    mul,
    sub,
    add,
    increment,
    assign,
    assign_zero,
    sin,
    cos,
    square,
    datanode,
    div,
    exp,
    sign,
    own_routine_symbol
)
from psyclone.autodiff.simplify import simplify_node
from psyclone.autodiff.subroutine_generator import SubroutineGenerator
from psyclone.autodiff.comparator_generator import ComparatorGenerator
from psyclone.autodiff.numerical_comparator import NumericalComparator

__all__ = [
    "ADReversalSchedule",
    "ADSplitReversalSchedule",
    "ADJointReversalSchedule",
    "ADLinkReversalSchedule",
    "own_routine_symbol",
    "one",
    "zero",
    "minus",
    "inverse",
    "power",
    "sqrt",
    "log",
    "mul",
    "sub",
    "add",
    "increment",
    "assign",
    "assign_zero",
    "sin",
    "cos",
    "square",
    "datanode",
    "div",
    "exp",
    "sign",
    "simplify_node",
    "ComparatorGenerator",
    "NumericalComparator",
    "SubroutineGenerator"
]
