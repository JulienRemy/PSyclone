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

"""This module provides a Transformation for foward-mode automatic 
differentiation of PSyIR Schedule nodes."""

from psyclone.psyir.nodes import (
    Routine,
    Assignment,
    Call,
    Reference,
    Schedule,
)
from psyclone.psyir.symbols import (
    REAL_DOUBLE_TYPE,
    SymbolTable,
    DataSymbol,
    ScalarType,
    ArrayType,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface
from psyclone.core import VariablesAccessInfo

from psyclone.autodiff.transformations import ADScheduleTrans


class ADForwardScheduleTrans(ADScheduleTrans):
    """A class for automatic differentation transformations of Schedule nodes.
    Requires an ADForwardContainerTrans instance as context, where the definitions of
    the routines called inside the schedule to be transformed can be found.

    :param container_trans: ADContainerTrans context instance
    :type container_trans: :py:class:`psyclone.autodiff.transformations.ADContainerTrans`

    :raises TypeError: if the container_trans argument is of the wrong type.
    """

    _number_of_schedules = 1
    _differential_prefix = ""
    _differential_postfix = "_d"
    _differential_table_index = 0

    def __init__(self, container_trans):
        super().__init__(container_trans)

        # Transformations need to know about the ADForwardScheduleTrans calling them
        # to access the attributes defined above
        # Import here to avoid circular dependencies
        from psyclone.autodiff.transformations import (
            ADForwardOperationTrans,
            ADForwardAssignmentTrans,
            ADForwardCallTrans,
        )

        # Initialize the sub transformations
        # self.adjoint_symbol_trans = ADAdjointSymbolTrans(self)
        self.assignment_trans = ADForwardAssignmentTrans(self)
        self.operation_trans = ADForwardOperationTrans(self)
        self.call_trans = ADForwardCallTrans(self)

    def apply(
        self, schedule, dependent_vars, independent_vars, options=None
    ):
        """Applies the transformation, generating the transformed \
        schedules that correspond to automatic differentiation of this Schedule \
        using forward-mode.

        Options:
        - bool 'verbose' : toggles preceding comment before the Jacobian \
            routine definition.
        - bool 'simplify': True to apply simplifications after applying AD \
            transformations. Defaults to True.
        - int 'simplify_n_times': number of time to apply simplification \
            rules to BinaryOperation nodes. Defaults to 5.

        :param schedule: schedule Node to the transformed.
        :type schedule: :py:class:`psyclone.psyir.nodes.Schedule`
        :param dependent_vars: list of dependent variables names to be \
            differentiated.
        :type dependent_vars: `List[str]`
        :param independent_vars: list of independent variables names to \
            differentiate with respect to.
        :type independent_vars: `List[str]`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises NotImplementedError: if no transformation rule has yet been \
            implemented for one of the children of schedule.

        :return: transformed schedule.
        :rtype: :py:class:`psyclone.psyir.nodes.Schedule`
        """
        self.validate(schedule, dependent_vars, independent_vars, options)

        self._was_applied = True

        self.schedule = schedule
        self.dependent_variables = dependent_vars
        self.independent_variables = independent_vars

        # Get the variables access information (to determine overwrites and taping)
        self.variables_info = VariablesAccessInfo(schedule)

        # Empty transformed schedule with symbol table
        self.transformed = self.create_transformed_schedules()

        # Process all symbols in the table, generating derivative symbols
        self.process_data_symbols(options)

        # Transform the statements found in the Schedule
        self.transform_children(options)

        # Simplify the BinaryOperation and Assignment nodes
        # in the transformed schedule
        simplify = self.unpack_option("simplify", options)
        if simplify:
            self.simplify(self.transformed[0], options)

        return self.transformed[0]

    def transform_assignment(self, assignment, options=None):
        """Transforms an Assignment child of the schedule and adds the \
        statements to the transformed schedule.

        :param assignment: assignment to transform.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if assignment is of the wrong type.

        """
        super().transform_assignment(assignment, options)

        transformed = []

        # Apply the transformation
        result = self.assignment_trans.apply(assignment, options)
        transformed.extend(result)

        # Insert in the transformed schedule
        self.add_children(self.transformed[0], transformed)

    def transform_call(self, call, options=None):
        """Transforms a Call child of the schedule and adds the \
        statements to the transformed schedule.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if call is of the wrong type.
        :raises NotImplementedError: if a recursive call is detected.
        """
        super().transform_call(call, options)

        # Apply an ADForwardCallTrans
        result = self.call_trans.apply(call, options)

        # Add the statements to the transformed schedule
        self.add_children(self.transformed[0], [result])


    @property
    def derivative_symbols(self):
        """Returns all the derivatives symbols used in transforming the Schedule.

        :return: list of all derivative symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        """
        return list(self.data_symbol_differential_map.values())
