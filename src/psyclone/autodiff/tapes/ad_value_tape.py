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

"""This module provides a class for reverse-mode automatic differentiation 
"taping" (storing and recovering) of function values.
"""

from psyclone.psyir.nodes import Assignment, Reference
from psyclone.psyir.symbols import ScalarType, ArrayType

from psyclone.autodiff.tapes import ADTape


class ADValueTape(ADTape):
    """A class for recording and recovering function values in reverse-mode \
    automatic differentiation. \
    The **prevalues** of references are recorded. \
    Based on static arrays storing a single type of data rather than a LIFO \
    stack. \
    Provides methods to create the PSyIR assignments for recording and \
    restoring operations.

    :param name: name of the value_tape (after a prefix).
    :type object: str
    :param datatype: datatype of the elements of the value_tape.
    :type datatype: Union[:py:class:`psyclone.psyir.symbols.ScalarType`,
                          :py:class:`psyclone.psyir.symbols.ArrayType`]

    :raises TypeError: if name is of the wrong type.
    :raises TypeError: if datatype is of the wrong type.
    :raises NotImplementedError: if datatype is not of type 'ScalarType'.
    """

    _node_types = (Reference,)
    _tape_prefix = "value_tape_"

    def __init__(self, name, datatype):
        # if not isinstance(name, str):
        #    raise TypeError(f"'name' argument should be of type "
        #                    f"'str' but found '{type(name).__name__}'.")
        if not isinstance(datatype, (ScalarType, ArrayType)):
            raise TypeError(
                f"'datatype' argument should be of type "
                f"'ScalarType' or 'ArrayType' but found "
                f"'{type(datatype).__name__}'."
            )
        if not isinstance(datatype, ScalarType):
            raise NotImplementedError(
                f"Only ScalarType symbols can be value_taped "
                f"for now but found DataType "
                f"'{datatype}' instead."
            )

        super().__init__(name, datatype)

    def record(self, reference):
        """Add the reference as last element of the value_tape and return the \
        Assignment node to record the prevalue to the tape.

        :param reference: reference whose prevalue should be recorded.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if reference is of the wrong type.
        :raises TypeError: if the intrinsic of reference's datatype is not the \
                                same as the intrinsic of the value_tape's elements \
                                datatype.
        :raises NotImplementedError: if the reference's datatype is ArrayType.

        :return: an Assignment node for recording the prevalue of the reference \
                    as last element of the value_tape.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        if not isinstance(reference, Reference):
            raise TypeError(f"'reference' argument should be of type "
                           f"'Reference' but found "
                           f"'{type(reference).__name__}'.")
        else:
            if reference.datatype.intrinsic != self.datatype.intrinsic:
                raise TypeError(
                    f"The intrinsic datatype of the 'reference' argument "
                    f"should be {self.datatype.intrinsic} but found "
                    f"{reference.datatype.intrinsic}."
                )
            if isinstance(reference.datatype, ArrayType):
                raise NotImplementedError("Taping arrays is not implemented yet.")

        value_tape_ref = super().record(reference)

        return Assignment.create(value_tape_ref, reference.copy())

    def restore(self, reference):
        """Restore the last element of the value_tape if it is the symbol argument \
         and return the Assignment node to restore the prevalue to the variable.

        :param reference: reference whose prevalue should be restored from the value_tape.
        :type reference: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :raises TypeError: if reference is of the wrong type.

        :return: an Assignment node for restoring the prevalue of the reference \
                    from the last element of the value_tape.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of type "
                f"'Reference' but found "
                f"'{type(reference).__name__}'."
            )

        value_tape_ref = super().restore(reference)

        return Assignment.create(reference.copy(), value_tape_ref)
