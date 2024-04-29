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

"""This module provides a Transformation for forward-mode automatic 
differentiation of PSyIR Routine nodes."""

from psyclone.core import VariablesAccessInfo
from psyclone.psyir.nodes import (
    Call,
    Assignment,
    IfBlock,
    Loop
)
from psyclone.psyir.symbols import ScalarType

from psyclone.autodiff.transformations import ADRoutineTrans


class ADForwardRoutineTrans(ADRoutineTrans):
    """A class for automatic differentation transformations of Routine nodes. 
    Requires an ADForwardContainerTrans instance as context, where the \
    definitions of  the routines called inside the one to be transformed \
    can be found.

    :param container_trans: ADForwardContainerTrans context instance
    :type container_trans: \
        :py:class:`psyclone.autodiff.transformations.ADForwardContainerTrans`

    :raises TypeError: if the container_trans argument is of the wrong type.
    """
    # pylint: disable=too-many-instance-attributes

    # Pre- and posfix of the transformed routine name
    _tangent_prefix = ""
    _tangent_postfix = "_tangent"
    _routine_prefixes = (_tangent_prefix,)
    _routine_postfixes = (_tangent_postfix,)

    # Redefining parent class attributes
    _number_of_routines = 1
    _differential_prefix = ""
    _differential_postfix = "_d"
    _differential_table_index = 0

    def __init__(self, container_trans):
        super().__init__()

        # Contextual container trans
        self.container_trans = container_trans

        # Transformations need to know about the ADForwardScheduleTrans 
        # calling them to access the attributes defined above
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import (
            ADForwardOperationTrans,
            ADForwardAssignmentTrans,
            ADForwardCallTrans,
            ADForwardIfBlockTrans,
            ADForwardLoopTrans
        )

        # Initialize the sub transformations
        self.assignment_trans = ADForwardAssignmentTrans(self)
        self.operation_trans = ADForwardOperationTrans(self)
        self.call_trans = ADForwardCallTrans(self)
        self.if_block_trans = ADForwardIfBlockTrans(self)
        self.loop_trans = ADForwardLoopTrans(self)

    @property
    def container_trans(self):
        """Returns the ADForwardContainerTrans this instance uses.

        :return: container transformation, forward-mode.
        :rtype: \
          :py:class:`psyclone.autodiff.transformations.ADForwardContainerTrans`
        """
        return self._container_trans

    @container_trans.setter
    def container_trans(self, container_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADForwardContainerTrans

        if not isinstance(container_trans, ADForwardContainerTrans):
            raise TypeError(
                f"Argument should be an 'ADForwardContainerTrans' "
                f"but found '{type(container_trans)}.__name__'."
            )

        self._container_trans = container_trans

    @property
    def assignment_trans(self):
        """Returns the ADForwardAssignmentTrans this instance uses.

        :return: assignment transformation, forward-mode.
        :rtype: \
          :py:class:`psyclone.autodiff.transformations.ADForwardAssignmentTrans`
        """
        return self._assignment_trans

    @assignment_trans.setter
    def assignment_trans(self, assignment_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADForwardAssignmentTrans

        if not isinstance(assignment_trans, ADForwardAssignmentTrans):
            raise TypeError(
                f"Argument should be an 'ADForwardAssignmentTrans' "
                f"but found '{type(assignment_trans)}.__name__'."
            )

        self._assignment_trans = assignment_trans

    @property
    def operation_trans(self):
        """Returns the ADForwardOperationTrans this instance uses.

        :return: operation transformation, forward-mode.
        :rtype: \
          :py:class:`psyclone.autodiff.transformations.ADForwardOperationTrans`
        """
        return self._operation_trans

    @operation_trans.setter
    def operation_trans(self, operation_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADForwardOperationTrans

        if not isinstance(operation_trans, ADForwardOperationTrans):
            raise TypeError(
                f"Argument should be an 'ADForwardOperationTrans' "
                f"but found '{type(operation_trans)}.__name__'."
            )

        self._operation_trans = operation_trans

    @property
    def call_trans(self):
        """Returns the ADForwardCallTrans this instance uses.

        :return: call transformation, forward-mode.
        :rtype: :py:class:`psyclone.autodiff.transformations.ADForwardCallTrans`
        """
        return self._call_trans

    @call_trans.setter
    def call_trans(self, call_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADForwardCallTrans

        if not isinstance(call_trans, ADForwardCallTrans):
            raise TypeError(
                f"Argument should be an 'ADForwardCallTrans' "
                f"but found '{type(call_trans)}.__name__'."
            )

        self._call_trans = call_trans

    @property
    def if_block_trans(self):
        """Returns the ADForwardIfBlockTrans this instance uses.

        :return: if block transformation, forward-mode.
        :rtype: :py:class:`psyclone.autodiff.transformations.\
                           ADForwardIfBlockTrans`
        """
        return self._if_block_trans

    @if_block_trans.setter
    def if_block_trans(self, if_block_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADForwardIfBlockTrans

        if not isinstance(if_block_trans, ADForwardIfBlockTrans):
            raise TypeError(
                f"Argument should be an 'ADForwardIfBlockTrans' "
                f"but found '{type(if_block_trans)}.__name__'."
            )

        self._if_block_trans = if_block_trans

    @property
    def loop_trans(self):
        """Returns the ADForwardLoopTrans this instance uses.

        :return: loop transformation, forward-mode.
        :rtype: :py:class:`psyclone.autodiff.transformations.ADForwardLoopTrans`
        """
        return self._loop_trans

    @loop_trans.setter
    def loop_trans(self, loop_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADForwardLoopTrans

        if not isinstance(loop_trans, ADForwardLoopTrans):
            raise TypeError(
                f"Argument should be an 'ADForwardLoopTrans' "
                f"but found '{type(loop_trans)}.__name__'."
            )

        self._loop_trans = loop_trans

    def apply(self, routine, dependent_vars, independent_vars, options=None):
        """Applies the transformation, generating the transformed routine \
        using forward-mode automatic differentiation.

        | Options:
        | - bool 'jacobian': whether to generate the Jacobian routine. \
                           Defaults to False.
        | - bool 'verbose' : toggles explanatory comments. Defaults to False.
        | - bool 'simplify': True to apply simplifications after applying AD \
                           transformations. Defaults to True.
        | - int 'simplify_n_times': number of time to apply simplification \
                                  rules to BinaryOperation nodes. Defaults to 5.

        :param routine: routine Node to the transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param dependent_vars: list of dependent variables names to be \
                               differentiated.
        :type dependent_vars: `List[Str]`
        :param independent_vars: list of independent variables names to \
                                 differentiate with respect to.
        :type independent_vars: `List[Str]`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises NotImplementedError: if no transformation rule has yet been \
                                     implemented for one of the children of \
                                     routine.

        :return: transformed Routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        self.validate(routine, dependent_vars, independent_vars, options)

        # Transformation can only be applied once
        self._was_applied = True

        self.routine = routine
        self.dependent_variables = dependent_vars
        self.independent_variables = independent_vars

        # Get the variables access information (to determine overwrites 
        # and taping)
        self.variables_info = VariablesAccessInfo(routine)

        # Add this transformation to the container_trans map
        # Do it before apply below or ordering is not from outer 
        # to inner routines
        self.container_trans.add_routine_trans(self)

        # Empty transformed routine with symbol table
        self.transformed = self.create_transformed_routines()

        # Process all symbols in the table, generating derivative symbols
        self.process_data_symbols(options)

        # Transform the statements found in the Schedule
        self.transform_children(options)

        # Postprocess the transformed routine (simplify the BinaryOperation
        # and Assignment nodes)
        self.postprocess(self.transformed[0], options)

        # Add the transformed routines symbol to the container_trans map
        self.container_trans.add_transformed_routine(
            self.routine_symbol, self.transformed_symbols[0]
        )

        # All dependent and independent variables names
        # list(set(...)) to avoid duplicates
        diff_variables = list(set(self.differential_variables))

        # Add the necessary adjoints as arguments of the returning routine
        self.add_derivative_arguments(diff_variables, options)

        # Add the assignments of 0 to other (non-argument) derivatives
        self.add_differentials_zero_assignments(self.transformed[0], options)

        # Add the transformed routine to the container
        self.container_trans.container.addchild(self.transformed[0])

        jacobian = self.unpack_option("jacobian", options)

        if jacobian:
            jacobian_routine = self.jacobian_routine(
                "forward", dependent_vars, independent_vars, options
            )
            self.container_trans.container.addchild(jacobian_routine)

        return self.transformed[0]

#    def transform_assignment(self, assignment, options=None):
#        """Transforms an Assignment child of the routine and adds the \
#        statements to the transformed routine.
#
#        :param assignment: assignment to transform.
#        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
#        :param options: a dictionary with options for transformations, \
#                        defaults to None.
#        :type options: Optional[Dict[Str, Any]]
#
#        :raises TypeError: if assignment is of the wrong type.
#
#        :return: list of nodes that correspond to the transformation of this \
#                 Assignment.
#        :rtype: List[:py:class:`psyclone.psyir.nodes.Assignment`]
#        """
#        if not isinstance(assignment, Assignment):
#            raise TypeError(
#                f"'assignment' argument should be of "
#                f"type 'Assignment' but found"
#                f"'{type(assignment).__name__}'."
#            )
#
#        # Apply the transformation
#        return self.assignment_trans.apply(assignment, options)
#
#        ## Insert in the transformed routine
#        # self.add_children(self.transformed[0], result)
#
#    def transform_call(self, call, options=None):
#        """Transforms a Call child of the routine and adds the \
#        statements to the transformed routine.
#
#        :param call: call to transform.
#        :type call: :py:class:`psyclone.psyir.nodes.Call`
#        :param options: a dictionary with options for transformations, \
#                        defaults to None.
#        :type options: Optional[Dict[Str, Any]]
#
#        :raises TypeError: if call is of the wrong type.
#
#        :return: single element list of nodes that correspond to the \
#                 transformation of this Call.
#        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`]
#        """
#        if not isinstance(call, Call):
#            raise TypeError(
#                f"'call' argument should be of "
#                f"type 'Call' but found"
#                f"'{type(call).__name__}'."
#            )
#
#        # Apply an ADForwardCallTrans
#        return [self.call_trans.apply(call, options)]
#
#        ## Add the statements to the transformed routine
#        # self.add_children(self.transformed[0], [result])
#
#    def transform_if_block(self, if_block, options=None):
#        """Transforms an IfBlock child of the routine and adds the \
#        statements to the transformed routine.
#
#        :param if_block: if_block to transform.
#        :type if_block: :py:class:`psyclone.psyir.nodes.IfBlock`
#        :param options: a dictionary with options for transformations, \
#                        defaults to None.
#        :type options: Optional[Dict[Str, Any]]
#
#        :raises TypeError: if if_block is of the wrong type.
#
#        :return: single element list of nodes that correspond to the \
#                 transformation of this IfBlock.
#        :rtype: List[:py:class:`psyclone.psyir.nodes.IfBlock`]
#        """
#        if not isinstance(if_block, IfBlock):
#            raise TypeError(
#                f"'if_block' argument should be of "
#                f"type 'IfBlock' but found"
#                f"'{type(if_block).__name__}'."
#            )
#
#        # Apply an ADForwardCallTrans
#        return [self.if_block_trans.apply(if_block, options)]
#
#        ## Add the statements to the transformed routine
#        #self.add_children(self.transformed[0], [result])

    def transform_children(self, options=None):
        """Transforms all the children of the routine being transformed \
        and adds the statements to the transformed routine.

        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises NotImplementedError: if the child transformation is not \
                                     implemented yet. For now only those for \
                                     Assignment, IfBlock and Call instances are.
        """
        # Go line by line through the Routine
        for child in self.routine.children:
            if isinstance(child, Assignment):
                result = self.assignment_trans.apply(child, options)
            elif isinstance(child, Call):
                result = [self.call_trans.apply(child, options)]
            elif isinstance(child, IfBlock):
                result = [self.if_block_trans.apply(child, options)]
            elif isinstance(child, Loop):
                result = [self.loop_trans.apply(child, options)]
            else:
                raise NotImplementedError(
                    f"Transforming a Routine child of "
                    f"type '{type(child).__name__}' is "
                    f"not implemented yet."
                )

            self.add_children(self.transformed[0], result)

    @property
    def derivative_symbols(self):
        """Returns all the derivatives symbols used in transforming the \
        Routine.

        :return: list of all derivative symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        """
        return list(self.data_symbol_differential_map.values())

    def add_derivative_arguments(self, diff_variables, options=None):
        """Add the derivatives of all differentiation variables \
        ie. dependent and independent ones \
        as arguments of the transformed routine, preserving intent. \

        :param variables: list of (in)dependent variables names, unique.
        :type variables: List[str]
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

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
            symbol = self.transformed_tables[0].lookup(
                var, scope_limit=self.transformed[0]
            )
            if symbol.datatype.intrinsic is ScalarType.Intrinsic.REAL:
                # Use the original symbol (not the copy) to get its derivative
                derivative_symbol = self.data_symbol_differential_map[symbol]
                # Same intent as the argument
                derivative_symbol.interface = symbol.interface

                # Insert the adjoint in the argument list
                # After the argument
                self.add_to_argument_list(
                    self.transformed_tables[0], derivative_symbol, after=symbol
                )

    def postprocess(self, routine, options=None):
        """Apply postprocessing steps (simplification) to the 
        'routine' argument.
        
        | Options:
        | - bool 'simplify': True to apply simplifications. Defaults to True.
        | - int 'simplify_n_times': number of time to apply simplification \
                                  rules to BinaryOperation nodes. Defaults to 5.

        :param routine: routine to postprocess.
        :type routine: py:class:`psyclone.psyir.nodes.Routine`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """

        # Simplify the BinaryOperation and Assignment nodes
        # in the transformed routine
        simplify = self.unpack_option("simplify", options)
        if simplify:
            self.simplify(routine, options)
