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

"""This module provides a Transformation for forward-mode automatic 
differentiation of PSyIR Routine nodes."""

from psyclone.core import VariablesAccessInfo
from psyclone.psyir.nodes import (
    Routine,
    Call,
    Reference,
    ArrayReference,
    Literal,
)
from psyclone.psyir.symbols import (
    INTEGER_TYPE,
    SymbolTable,
    DataSymbol,
    ArrayType,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface, AutomaticInterface
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff import assign_zero, own_routine_symbol, assign, one
from psyclone.autodiff.transformations import (
    ADForwardContainerTrans,
    ADForwardScheduleTrans,
    ADRoutineTrans
)


class ADForwardRoutineTrans(ADRoutineTrans):
    """A class for automatic differentation transformations of Routine nodes. 
    Requires an ADForwardContainerTrans instance as context, where the definitions of  \
    the routines called inside the one to be transformed can be found.
    Uses an ADForwardScheduleTrans internally.
    """

    _tangent_prefix = ""
    _tangent_postfix = "_tangent"

    _routine_prefixes = (_tangent_prefix, )
    _routine_postfixes = (_tangent_postfix, )

    _number_of_schedules = ADForwardScheduleTrans._number_of_schedules
    _differential_prefix = ADForwardScheduleTrans._differential_prefix
    _differential_postfix = ADForwardScheduleTrans._differential_postfix
    _differential_table_index = ADForwardScheduleTrans._differential_table_index

    def __init__(self, container_trans):
        super().__init__(container_trans)

        self.schedule_trans = ADForwardScheduleTrans(container_trans)
        self.assignment_trans = self.schedule_trans.assignment_trans
        self.assignment_trans.routine_trans = self
        self.operation_trans = self.schedule_trans.operation_trans
        self.operation_trans.routine_trans = self
        self.call_trans = self.schedule_trans.call_trans
        self.call_trans.routine_trans = self

    def apply(
        self, routine, dependent_vars, independent_vars, options=None
    ):
        """Applies the transformation, generating the transformed routine \
        using forward-mode automatic differentiation.

        Options:
        - bool 'jacobian': whether to generate the Jacobian routine. Defaults \
            to False.
        - bool 'verbose' : toggles explanatory comments. Defaults to False.
        - bool 'simplify': True to apply simplifications after applying AD \
            transformations. Defaults to True.
        - int 'simplify_n_times': number of time to apply simplification \
            rules to BinaryOperation nodes. Defaults to 5.

        :param routine: routine Node to the transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
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
            implemented for one of the children of routine.

        :return: transformed Routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        self.validate(routine, dependent_vars, independent_vars, options)

        self.routine = routine
        self.dependent_variables = dependent_vars
        self.independent_variables = independent_vars
        
        # Get the variables access information (to determine overwrites and taping)
        self.variables_info = VariablesAccessInfo(routine)

        # Add this transformation to the container_trans map
        # Do it before apply below or ordering is not from outer to inner routines
        self.container_trans.add_routine_trans(self)

        # Apply the ADForwardScheduleTrans
        schedule = self.schedule_trans.apply(routine, dependent_vars, independent_vars, options)

        # Raise the transformed schedule to routine
        self.transformed = self.schedules_to_routines([schedule])
        self.data_symbol_differential_map = self.schedule_trans.data_symbol_differential_map
        self.temp_symbols = self.schedule_trans.temp_symbols

        # Add the transformed routines symbol to the container_trans map
        self.container_trans.add_transformed_routine(
            self.routine_symbol, self.transformed_symbols[0]
        )

        # All dependent and independent variables names
        # list(set(...)) to avoid duplicates
        diff_variables = list(set(self.differential_variables))

        # Add the necessary adjoints as arguments of the returning routine
        self.add_derivative_arguments(diff_variables, options)

        # Add the transformed routine to the container
        self.container_trans.container.addchild(self.transformed[0])

        jacobian = self.unpack_option("jacobian", options)

        if jacobian:
            jacobian_routine = self.jacobian_routine(
                dependent_vars, independent_vars, options
            )
            self.container_trans.container.addchild(jacobian_routine)

        return self.transformed[0]

    def add_derivative_arguments(self, diff_variables, options=None):
        """Add the derivatives of all differentiation variables \
        ie. dependent and independent ones \
        as arguments of the transformed routine, preserving intent. \

        :param variables: list of (in)dependent variables names, unique.
        :type variables: List[str]
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if diff_variables is of the wrong type.
        """
        if not isinstance(diff_variables, list):
            raise TypeError(
                f"'diff_variables' argument should be of "
                f"type 'List[str]' but found"
                f"'{type(diff_variables).__name__}'."
            )
        for var in diff_variables:
            if not isinstance(var, str):
                raise TypeError(
                    f"'diff_variables' argument should be of "
                    f"type 'List[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )

        for var in diff_variables:
            # Get the symbol associated to the name, then the derivative symbol
            symbol = self.transformed_tables[0].lookup(var, scope_limit=self.transformed[0])

            # Use the original symbol (not the copy) to get its derivative
            derivative_symbol = self.data_symbol_differential_map[symbol]
            # Same intent as the argument
            derivative_symbol.interface = symbol.interface

            # Insert the adjoint in the argument list
            # After the argument
            self.add_to_argument_list(
                self.transformed_tables[0], derivative_symbol, after=symbol
            )
