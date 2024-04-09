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
from psyclone.psyir.symbols import (
    ScalarType,
    ArrayType,
    INTEGER_TYPE,
    RoutineSymbol,
)
from psyclone.psyir.backend.fortran import FortranWriter

from psyclone.autodiff.tapes import ADTape
from psyclone.autodiff import one


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

    def record(self, node, do_loop=False):
        """Add the node as last element of the tape and return the \
        ArrayReference node of the tape.

        :param node: node whose prevalue should be recorded.
        :type node: :py:class:`psyclone.psyir.nodes.Reference`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if the intrinsic of node's datatype is not the \
                           same as the intrinsic of the value_tape's \
                           elements datatype.
        :raises TypeError: if do_loop is of the wrong type.

        :return: the array node to the last element of the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )

        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        self._recorded_nodes.append(node)
        self._offset_mask.append(True)
        self._multiplicities.append(one())
        self._usefully_recorded_flags.append(True)
        self._recordings.append(None)
        self._restorings.append(None)

        # If static array, reshape to take the new length into account
        if not self.is_dynamic_array:
            self.reshape()

        record = self.create_record_call(
            node, self.first_index_of_last_element(do_loop)
        )

        self._recordings[-1] = record

        return record

    def restore(self, node, do_loop=False):
        """Check that node is the last element of the tape and return an \
        ArrayReference to it in the tape.

        :param node: node restore.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if do_loop is of the wrong type.

        :return: an ArrayReference node to the last element of the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        self._has_last(node)

        restore = self.create_restore_call(
            node, self.first_index_of_last_element(do_loop)
        )

        self.restorings[-1] = restore

        return restore

    @property
    def subroutines_source(self):
        datatype = self.datatype_fortran_string

        source = ""

        if self.taped_at_least_one_scalar:
            source += """subroutine record_{datatype}_scalar(tape, index, scalar)
    implicit none
    {datatype}, dimension(:), intent(inout) :: tape
    {datatype}, intent(in) :: scalar
    integer, intent(in) :: index

    tape(index) = scalar

end subroutine record_{datatype}_scalar

subroutine restore_{datatype}_scalar(tape, index, scalar)
    implicit none
    {datatype}, dimension(:), intent(in) :: tape ! in?
    {datatype}, intent(out) :: scalar
    integer, intent(in) :: index

    scalar = tape(index)

end subroutine restore_{datatype}_scalar

"""

        if self.taped_at_least_one_vector:
            source += f"""subroutine record_{datatype}_vector(tape, first, vector)
    implicit none
    {datatype}, dimension(:), intent(inout) :: tape
    integer, intent(in) :: first
    {datatype}, dimension(:), intent(in) :: vector

    tape(first:first + size(vector) - 1) = vector(:)

end subroutine record_{datatype}_vector

subroutine restore_{datatype}_vector(tape, first, vector)
    implicit none
    {datatype}, dimension(:), intent(in) :: tape
    integer, intent(in) :: first
    {datatype}, dimension(:), intent(out) :: vector

    vector(1:size(vector)) = tape(first:first + size(vector) - 1)

end subroutine restore_{datatype}_vector

"""

        if self.taped_at_least_one_matrix:
            source += f"""subroutine record_{datatype}_matrix(tape, first, matrix)
    implicit none
    {datatype}, dimension(:), intent(inout) :: tape
    integer, intent(in) :: first
    {datatype}, dimension(:, :), intent(in) :: matrix

    integer :: i, j

    do j = lbound(matrix, 2), ubound(matrix, 2)
        do i = lbound(matrix, 1), ubound(matrix, 1)
            tape(first + (j - 1) * size(matrix, 1) + i - 1) = matrix(i, j)
        end do
    end do

end subroutine record_{datatype}_matrix

subroutine restore_{datatype}_matrix(tape, first, matrix)
    implicit none
    {datatype}, dimension(:), intent(in) :: tape
    integer, intent(in) :: first
    {datatype}, dimension(:, :), intent(out) :: matrix

    integer :: i, j 

    do j = lbound(matrix, 2), ubound(matrix, 2)
        do i = lbound(matrix, 1), ubound(matrix, 1)
            matrix(i, j) = tape(first + (j - 1) * size(matrix, 1) + i - 1)
        end do
    end do

end subroutine restore_{datatype}_matrix

"""
        return source.format(datatype=datatype)

    @property
    def subroutines_pointers_source(self):
        datatype = self.datatype_fortran_string

        source = ""
        if self.taped_at_least_one_scalar:
            source += f"""subroutine restore_{datatype}_scalar_as_pointer(tape, index, ptr)
    implicit none
    {datatype}, dimension(:), target, intent(in) :: tape
    {datatype}, intent(inout), pointer :: ptr
    integer, intent(in) :: index

    ptr => tape(index)

end subroutine restore_{datatype}_scalar_as_pointer

"""

        if self.taped_at_least_one_vector:
            source += f"""subroutine restore_{datatype}_vector_as_pointer(tape, first, ptr)
    implicit none
    {datatype}, dimension(:), target, intent(in) :: tape
    integer, intent(in) :: first
    {datatype}, dimension(:), pointer, intent(out) :: ptr

    ptr(1:size(ptr)) => tape(first:first + size(ptr) - 1)

end subroutine restore_{datatype}_vector_as_pointer

"""

        if self.taped_at_least_one_matrix:
            source += f"""subroutine restore_{datatype}_matrix_as_pointer(tape, first, ptr)
    implicit none
    {datatype}, dimension(:), target, intent(in) :: tape
    integer, intent(in) :: first
    {datatype}, dimension(:, :), pointer, intent(out) :: ptr

    ptr(lbound(ptr, 1):ubound(ptr, 1), lbound(ptr, 2):ubound(ptr, 2)) => tape(first : first + size(ptr) - 1)

end subroutine restore_{datatype}_matrix_as_pointer

"""
        return source.format(datatype=datatype)

    @property
    def subroutines_nodes(self):
        from psyclone.psyir.frontend.fortran import FortranReader

        freader = FortranReader()
        nodes = freader.psyir_from_source(
            self.subroutines_source
        ).pop_all_children()

        return nodes

    @property
    def subroutines_pointers_nodes(self):
        from psyclone.psyir.frontend.fortran import FortranReader

        freader = FortranReader()
        nodes = freader.psyir_from_source(
            self.subroutines_pointers_source
        ).pop_all_children()

        return nodes

    def create_record_call(self, datanode, first_index):
        datatype = self.datatype_fortran_string

        if isinstance(datanode.datatype, ScalarType):
            dim = "scalar"
            self.taped_at_least_one_scalar = True
        elif isinstance(datanode.datatype, ArrayType):
            if len(datanode.datatype.shape) == 1:
                dim = "vector"
                self.taped_at_least_one_vector = True
            elif len(datanode.datatype.shape) == 2:
                dim = "matrix"
                self.taped_at_least_one_matrix = True
            else:
                raise NotImplementedError(
                    "Only vectors and matrices are implemented but got an "
                    f"array of dimension {len(datanode.datatype.shape)}."
                )
        else:
            raise NotImplementedError(
                "Only ScalarType and ArrayType are implemented but got "
                f"datatype {type(datanode.datatype).__name__}."
            )

        routine_name = f"record_{datatype}_{dim}"
        routine_symbol = RoutineSymbol(routine_name)

        return Call.create(
            routine_symbol,
            [Reference(self.symbol), first_index, datanode.copy()],
        )

    def create_restore_call(self, datanode, first_index):
        datatype = self.datatype_fortran_string

        if isinstance(datanode.datatype, ScalarType):
            dim = "scalar"
        elif isinstance(datanode.datatype, ArrayType):
            if len(datanode.datatype.shape) == 1:
                dim = "vector"
            elif len(datanode.datatype.shape) == 2:
                dim = "matrix"
            else:
                raise NotImplementedError(
                    "Only vectors and matrices are implemented but got an "
                    f"array of dimension {len(datanode.datatype.shape)}."
                )
        else:
            raise NotImplementedError(
                "Only ScalarType and ArrayType are implemented but got "
                f"datatype {type(datanode.datatype).__name__}."
            )

        routine_name = f"restore_{datatype}_{dim}"
        routine_symbol = RoutineSymbol(routine_name)

        return Call.create(
            routine_symbol,
            [Reference(self.symbol), first_index, datanode.copy()],
        )

    # def record(self, reference, do_loop=False):
    #     """Add the reference as last element of the value_tape and return the \
    #     Assignment node to record the prevalue to the tape.

    #     :param reference: reference whose prevalue should be recorded.
    #     :type reference: :py:class:`psyclone.psyir.nodes.Reference`
    #     :param do_loop: whether currently transforming a do loop. \
    #                     Optional, defaults to False.
    #     :type do_loop: Optional[bool]

    #     :raises TypeError: if reference is of the wrong type.
    #     :raises TypeError: if do_loop is of the wrong type.
    #     :raises TypeError: if the intrinsic of reference's datatype is not the \
    #                        same as the intrinsic of the value_tape's elements \
    #                        datatype.

    #     :return: an Assignment node for recording the prevalue of the reference\
    #              as last element of the value_tape.
    #     :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
    #     """
    #     # pylint: disable=arguments-renamed

    #     if not isinstance(reference, Reference):
    #         raise TypeError(
    #             f"'reference' argument should be of type "
    #             f"'Reference' but found "
    #             f"'{type(reference).__name__}'."
    #         )

    #     if isinstance(reference.datatype, ScalarType):
    #         scalar_type = reference.datatype
    #     else:
    #         scalar_type = reference.datatype.datatype

    #     if not isinstance(do_loop, bool):
    #         raise TypeError(
    #             f"'bool' argument should be of type "
    #             f"'bool' but found '{type(do_loop).__name__}'."
    #         )

    #     # Record the primal reference
    #     value_tape_ref = super().record(reference, do_loop)

    #     if scalar_type.intrinsic != self.datatype.intrinsic:
    #         # FIXME: this is a dirty hack to tape some integers...
    #         if (
    #             scalar_type.intrinsic is ScalarType.Intrinsic.INTEGER
    #             and self.datatype.intrinsic is ScalarType.Intrinsic.REAL
    #         ):
    #             reference = IntrinsicCall.create(
    #                 IntrinsicCall.Intrinsic.REAL, [reference.copy()]
    #             )
    #         else:
    #             raise TypeError(
    #                 f"The intrinsic datatype of the 'reference' argument "
    #                 f"should be {self.datatype.intrinsic} but found "
    #                 f"{scalar_type.intrinsic}."
    #             )

    #     # This is a Reference to a scalar
    #     if isinstance(reference.datatype, ScalarType):
    #         assignment = Assignment.create(value_tape_ref, reference.copy())

    #     # This is an ArrayReference
    #     else:
    #         # to a vector
    #         if len(reference.datatype.shape) == 1:
    #             index = [i.copy() for i in reference.indices]
    #             assignment = Assignment.create(
    #                 value_tape_ref,
    #                 ArrayReference.create(reference.symbol, index),
    #             )

    #         # to a >1D array, RESHAPE
    #         else:
    #             # Create an IntrinsicCall to RESHAPE to reshape the reference
    #             # array to a 1D vector
    #             fortran_writer = FortranWriter()
    #             size = self._array_size(reference)
    #             size_str = fortran_writer(size)
    #             shape_array = Literal(
    #                 f"(/ {size_str} /)", ArrayType(INTEGER_TYPE, [1])
    #             )
    #             reshaped = IntrinsicCall.create(
    #                 IntrinsicCall.Intrinsic.RESHAPE,
    #                 [reference.copy(), shape_array],
    #             )
    #             assignment = Assignment.create(value_tape_ref, reshaped)

    #     # Replace the recording by the actual recording assignment
    #     self._recordings[-1] = assignment

    #     return assignment

    # def restore(self, reference, do_loop=False):
    #     """Restore the last element of the value_tape if it is the symbol \
    #     argument and return the Assignment node to restore the prevalue to the \
    #     variable.

    #     :param reference: reference whose prevalue should be restored from the \
    #                       value_tape.
    #     :type reference: :py:class:`psyclone.psyir.symbols.DataSymbol`
    #     :param do_loop: whether currently transforming a do loop. \
    #                     Optional, defaults to False.
    #     :type do_loop: Optional[bool]

    #     :raises TypeError: if reference is of the wrong type.
    #     :raises TypeError: if do_loop is of the wrong type.

    #     :return: an Assignment node for restoring the prevalue of the reference\
    #              from the last element of the value_tape.
    #     :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
    #     """
    #     # pylint: disable=arguments-renamed

    #     if not isinstance(reference, Reference):
    #         raise TypeError(
    #             f"'reference' argument should be of type "
    #             f"'Reference' but found "
    #             f"'{type(reference).__name__}'."
    #         )

    #     if not isinstance(do_loop, bool):
    #         raise TypeError(
    #             f"'bool' argument should be of type "
    #             f"'bool' but found '{type(do_loop).__name__}'."
    #         )

    #     value_tape_ref = super().restore(reference, do_loop)

    #     # FIXME: this is a dirty hack to tape some integers...
    #     if (
    #         isinstance(reference.datatype, ScalarType)
    #         and reference.datatype.intrinsic is ScalarType.Intrinsic.INTEGER
    #         and self.datatype.intrinsic is ScalarType.Intrinsic.REAL
    #     ):
    #         value_tape_ref = IntrinsicCall.create(
    #             IntrinsicCall.Intrinsic.INT, [value_tape_ref]
    #         )

    #     # This is a Reference to a scalar
    #     if isinstance(reference.datatype, ScalarType):
    #         # if isinstance(reference.datatype, ScalarType):
    #         assignment = Assignment.create(reference.copy(), value_tape_ref)

    #     # This is an ArrayReference
    #     else:
    #         # to a vector
    #         if len(reference.datatype.shape) == 1:
    #             index = [i.copy() for i in reference.indices]
    #             assignment = Assignment.create(
    #                 ArrayReference.create(reference.symbol, index),
    #                 value_tape_ref,
    #             )

    #         # to a >1D array
    #         else:
    #             # Create an IntrinsicCall to RESHAPE to reshape the
    #             # value_tape_ref slice to the dimensions of the reference array
    #             fortran_writer = FortranWriter()
    #             dimensions = self._array_dimensions(reference)
    #             str_dimensions = [fortran_writer(dim) for dim in dimensions]
    #             shape_str = "(/ " + ", ".join(str_dimensions) + " /)"
    #             shape_datatype = ArrayType(INTEGER_TYPE, [len(dimensions)])
    #             shape_array = Literal(shape_str, shape_datatype)
    #             reshaped = IntrinsicCall.create(
    #                 IntrinsicCall.Intrinsic.RESHAPE,
    #                 [value_tape_ref, shape_array],
    #             )
    #             assignment = Assignment.create(reference.copy(), reshaped)

    #     # Replace the restoring by the actual restoring assignment
    #     self._restorings[-1] = assignment

    #     return assignment
