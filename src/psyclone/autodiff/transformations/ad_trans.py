# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2021-2023, Science and Technology Facilities Council.
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
# Author J. Remy, Inria

"""This module provides an abstract Transformation for reverse-mode automatic 
differentiation of PSyIR nodes."""

from abc import ABCMeta

from psyclone.psyGen import Transformation

class ADTrans(Transformation, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations.
    Requires an ADRoutineTrans instance as context, where the symbols of
    adjoints, temporaries, etc. are stored.
    """

    @property
    def routine_trans(self):
        """Contextual ADRoutineTrans from which this ADTrans subclass instance \
        was created.

        :return: contextual ADRoutineTrans
        :rtype: :py:class:`psyclone.autodiff.transformations.ADRoutineTrans`
        """
        return self._routine_trans
    
    @routine_trans.setter
    def routine_trans(self, routine_trans):
        from psyclone.autodiff.transformations import ADRoutineTrans
        if not isinstance(routine_trans, ADRoutineTrans):
            raise TypeError(
                f"Argument should be of type 'ADRoutineTrans' "
                f"but found '{type(routine_trans).__name__}'."
            )
        self._routine_trans = routine_trans

    def __init__(self, routine_trans):
        #
        from psyclone.autodiff.transformations import ADRoutineTrans
        if not isinstance(routine_trans, ADRoutineTrans):
            raise TypeError(
                f"Argument should be of type 'ADRoutineTrans' "
                f"but found '{type(routine_trans).__name__}'."
            )
        self.routine_trans = routine_trans

