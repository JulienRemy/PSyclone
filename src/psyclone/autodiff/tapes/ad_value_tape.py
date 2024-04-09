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
# Author: J. Remy, Université Grenoble Alpes, Inria

"""This module provides a class for reverse-mode automatic differentiation 
"taping" (storing and recovering) of function values.
"""

from psyclone.psyir.nodes import (
    Assignment,
    Reference,
    IntrinsicCall,
    Call,
    Literal,
    ArrayReference,
    Operation,
    Routine,
)
from psyclone.psyir.symbols import ScalarType, ArrayType, INTEGER_TYPE
from psyclone.psyir.backend.fortran import FortranWriter

from psyclone.autodiff.tapes import ADTape


class ADValueTape(ADTape):
    """A class for recording and recovering function values in reverse-mode \
    automatic differentiation. 
    The **prevalues** of references are recorded. 
    Based on static arrays storing a single type of data rather than a LIFO \
    stack. 
    Provides methods to create the PSyIR assignments for recording and \
    restoring operations.

    :param name: name of the value_tape (after a prefix).
    :type object: str
    :param datatype: datatype of the elements of the value_tape.
    :type datatype: :py:class:`psyclone.psyir.symbols.ScalarType`.
    :param is_dynamic_array: whether to make the Fortran array dynamic \
                             (allocatable) or not. Optional, defaults to False.
    :type is_dynamic_array: Optional[bool]

    :raises TypeError: if name is of the wrong type.
    :raises TypeError: if datatype is of the wrong type.
    :raises TypeError: if use_offsets is of the wrong type.
    :raises TypeError: if is_dynamic_array is of the wrong type.
    """

    _node_types = (Reference, Operation, Call)
    _tape_prefix = "value_tape_"

    def __init__(self, name, datatype, is_dynamic_array=False):
        if not isinstance(datatype, (ScalarType)):
            raise TypeError(
                f"'datatype' argument should be of type "
                f"'ScalarType' but found "
                f"'{type(datatype).__name__}'."
            )

        super().__init__(name, datatype, is_dynamic_array)

    def record(self, reference, do_loop=False):
        """Add the reference as last element of the value_tape and return the \
        Assignment node to record the prevalue to the tape.

        :param reference: reference whose prevalue should be recorded.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if reference is of the wrong type.
        :raises TypeError: if do_loop is of the wrong type.
        :raises TypeError: if the intrinsic of reference's datatype is not the \
                           same as the intrinsic of the value_tape's elements \
                           datatype.

        :return: an Assignment node for recording the prevalue of the reference\
                 as last element of the value_tape.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        # pylint: disable=arguments-renamed

        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of type "
                f"'Reference' but found "
                f"'{type(reference).__name__}'."
            )

        if isinstance(reference.datatype, ScalarType):
            scalar_type = reference.datatype
        else:
            scalar_type = reference.datatype.datatype

        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        # Record the primal reference
        value_tape_ref = super().record(reference, do_loop)

        if scalar_type.intrinsic != self.datatype.intrinsic:
            raise TypeError(
                f"The intrinsic datatype of the 'reference' argument "
                f"should be {self.datatype.intrinsic} but found "
                f"{scalar_type.intrinsic}."
            )

        # This is a Reference to a scalar
        if isinstance(reference.datatype, ScalarType):
            assignment = Assignment.create(value_tape_ref, reference.copy())

        # This is an ArrayReference
        else:
            # to a vector
            if len(reference.datatype.shape) == 1:
                index = [i.copy() for i in reference.indices]
                assignment = Assignment.create(
                    value_tape_ref,
                    ArrayReference.create(reference.symbol, index),
                )

            # to a >1D array, RESHAPE
            else:
                # Create an IntrinsicCall to RESHAPE to reshape the reference 
                # array to a 1D vector
                fortran_writer = FortranWriter()
                size = self._array_size(reference)
                size_str = fortran_writer(size)
                shape_array = Literal(
                    f"(/ {size_str} /)", ArrayType(INTEGER_TYPE, [1])
                )
                reshaped = IntrinsicCall.create(
                    IntrinsicCall.Intrinsic.RESHAPE,
                    [reference.copy(), shape_array],
                )
                assignment = Assignment.create(value_tape_ref, reshaped)

        # Replace the recording by the actual recording assignment
        self._recordings[-1] = assignment

        return assignment

    def restore(self, reference, do_loop=False):
        """Restore the last element of the value_tape if it is the symbol \
        argument and return the Assignment node to restore the prevalue to the \
        variable.

        :param reference: reference whose prevalue should be restored from the \
                          value_tape.
        :type reference: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if reference is of the wrong type.
        :raises TypeError: if do_loop is of the wrong type.

        :return: an Assignment node for restoring the prevalue of the reference\
                 from the last element of the value_tape.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        # pylint: disable=arguments-renamed

        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of type "
                f"'Reference' but found "
                f"'{type(reference).__name__}'."
            )

        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        value_tape_ref = super().restore(reference, do_loop)

        # This is a Reference to a scalar
        if isinstance(reference.datatype, ScalarType):
            # if isinstance(reference.datatype, ScalarType):
            assignment = Assignment.create(reference.copy(), value_tape_ref)

        # This is an ArrayReference
        else:
            # to a vector
            if len(reference.datatype.shape) == 1:
                index = [i.copy() for i in reference.indices]
                assignment = Assignment.create(
                    ArrayReference.create(reference.symbol, index),
                    value_tape_ref,
                )

            # to a >1D array
            else:
                # Create an IntrinsicCall to RESHAPE to reshape the 
                # value_tape_ref slice to the dimensions of the reference array
                fortran_writer = FortranWriter()
                dimensions = self._array_dimensions(reference)
                str_dimensions = [fortran_writer(dim) for dim in dimensions]
                shape_str = "(/ " + ", ".join(str_dimensions) + " /)"
                shape_datatype = ArrayType(INTEGER_TYPE, [len(dimensions)])
                shape_array = Literal(shape_str, shape_datatype)
                reshaped = IntrinsicCall.create(
                    IntrinsicCall.Intrinsic.RESHAPE,
                    [value_tape_ref, shape_array],
                )
                assignment = Assignment.create(reference.copy(), reshaped)

        # Replace the restoring by the actual restoring assignment
        self._restorings[-1] = assignment

        return assignment
