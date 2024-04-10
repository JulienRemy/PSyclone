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

"""This module provides a Transformation for reverse-mode automatic 
differentiation of PSyIR Routine nodes."""

from types import NoneType
from psyclone.core import VariablesAccessInfo
from psyclone.psyir.nodes import (
    Call,
    Reference,
    ArrayReference,
    Assignment,
    Literal,
    Operation,
    IntrinsicCall,
    IfBlock,
    Loop,
    Schedule,
    DataNode,
    OMPRegionDirective,
)
from psyclone.psyir.symbols import (
    REAL_TYPE,
    BOOLEAN_TYPE,
    DataSymbol,
    ScalarType,
    ArrayType,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface

from psyclone.autodiff import own_routine_symbol, assign_zero
from psyclone.autodiff.tapes import ADTape, ADValueTape, ADControlTape
from psyclone.autodiff.transformations import ADRoutineTrans


class ADReverseRoutineTrans(ADRoutineTrans):
    """A class for automatic differentation transformations of Routine nodes. 
    Requires an ADReverseContainerTrans instance as context, where the \
    definitions of the routines called inside the one to be transformed can be \
    found.

    :param container_trans: ADReverseContainerTrans context instance
    :type container_trans: 
           :py:class:`psyclone.autodiff.transformations.ADReverseContainerTrans`

    :raises TypeError: if the container_trans argument is of the wrong type.
    """

    # Pre- and postfix of the three transformed routines
    _recording_prefix = ""
    _recording_postfix = "_rec"

    _returning_prefix = ""
    _returning_postfix = "_ret"

    _reversing_prefix = ""
    _reversing_postfix = "_rev"

    _routine_prefixes = (
        _recording_prefix,
        _returning_prefix,
        _reversing_prefix,
    )
    _routine_postfixes = (
        _recording_postfix,
        _returning_postfix,
        _reversing_postfix,
    )

    # Redefining parent class attributes
    _number_of_routines = 3  # Recording, returning, reversing
    _differential_prefix = ""
    _differential_postfix = "_adj"
    _differential_table_index = 1  # Adjoints are created in the returning table

    _operation_adjoint_name = "op_adj"
    # _call_adjoint_name = "call_adj"

    # TODO: correct datatype
    # _default_value_tape_datatype = REAL_TYPE
    _default_control_tape_datatype = BOOLEAN_TYPE

    def __init__(self, container_trans):
        super().__init__()

        # self.substitution_map = dict()

        # Contextual container trans
        self.container_trans = container_trans

        # DataSymbol => adjoint DataSymbol
        # self.data_symbol_differential_map = dict()

        # Lists of adjoint symbols for operations
        self.operation_adjoints = []

        # Transformations need to know about the ADReverseRoutineTrans
        # calling them
        # to access the attributes defined above
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import (
            ADReverseOperationTrans,
            ADReverseAssignmentTrans,
            ADReverseCallTrans,
            ADReverseIfBlockTrans,
            ADReverseLoopTrans,
            ADReverseOMPRegionDirectiveTrans,
        )

        # Initialize the sub transformations
        self.assignment_trans = ADReverseAssignmentTrans(self)
        self.operation_trans = ADReverseOperationTrans(self)
        self.call_trans = ADReverseCallTrans(self)
        self.if_block_trans = ADReverseIfBlockTrans(self)
        self.loop_trans = ADReverseLoopTrans(self)
        self.omp_region_trans = ADReverseOMPRegionDirectiveTrans(self)

    @property
    def container_trans(self):
        """Returns the ADReverseContainerTrans this instance uses.

        :return: container transformation, reverse-mode.
        :rtype: \
           :py:class:`psyclone.autodiff.transformations.ADReverseContainerTrans`
        """
        return self._container_trans

    @container_trans.setter
    def container_trans(self, container_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseContainerTrans

        if not isinstance(container_trans, ADReverseContainerTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseContainerTrans' "
                f"but found '{type(container_trans)}.__name__'."
            )

        self._container_trans = container_trans

    @property
    def assignment_trans(self):
        """Returns the ADReverseAssignmentTrans this instance uses.

        :return: assignment transformation, reverse-mode.
        :rtype: \
          :py:class:`psyclone.autodiff.transformations.ADReverseAssignmentTrans`
        """
        return self._assignment_trans

    @assignment_trans.setter
    def assignment_trans(self, assignment_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseAssignmentTrans

        if not isinstance(assignment_trans, ADReverseAssignmentTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseAssignmentTrans' "
                f"but found '{type(assignment_trans)}.__name__'."
            )

        self._assignment_trans = assignment_trans

    @property
    def operation_trans(self):
        """Returns the ADReverseOperationTrans this instance uses.

        :return: operation transformation, reverse-mode.
        :rtype: \
           :py:class:`psyclone.autodiff.transformations.ADReverseOperationTrans`
        """
        return self._operation_trans

    @operation_trans.setter
    def operation_trans(self, operation_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseOperationTrans

        if not isinstance(operation_trans, ADReverseOperationTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseOperationTrans' "
                f"but found '{type(operation_trans)}.__name__'."
            )

        self._operation_trans = operation_trans

    @property
    def call_trans(self):
        """Returns the ADReverseCallTrans this instance uses.

        :return: call transformation, reverse-mode.
        :rtype: :py:class:`psyclone.autodiff.transformations.ADReverseCallTrans`
        """
        return self._call_trans

    @call_trans.setter
    def call_trans(self, call_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseCallTrans

        if not isinstance(call_trans, ADReverseCallTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseCallTrans' "
                f"but found '{type(call_trans)}.__name__'."
            )

        self._call_trans = call_trans

    @property
    def if_block_trans(self):
        """Returns the ADReverseIfBlockTrans this instance uses.

        :return: if block transformation, reverse-mode.
        :rtype: :py:class:`psyclone.autodiff.transformations.\
                           ADReverseIfBlockTrans`
        """
        return self._if_block_trans

    @if_block_trans.setter
    def if_block_trans(self, if_block_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseIfBlockTrans

        if not isinstance(if_block_trans, ADReverseIfBlockTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseIfBlockTrans' "
                f"but found '{type(if_block_trans)}.__name__'."
            )

        self._if_block_trans = if_block_trans

    @property
    def loop_trans(self):
        """Returns the ADReverseLoopTrans this instance uses.

        :return: loop transformation, reverse-mode.
        :rtype: :py:class:`psyclone.autodiff.transformations.ADReverseLoopTrans`
        """
        return self._loop_trans

    @loop_trans.setter
    def loop_trans(self, loop_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseLoopTrans

        if not isinstance(loop_trans, ADReverseLoopTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseLoopTrans' "
                f"but found '{type(loop_trans)}.__name__'."
            )

        self._loop_trans = loop_trans

    @property
    def omp_region_trans(self):
        """Returns the ADReverseOMPRegionDirectiveTrans this instance uses.

        :return: omp_region transformation, reverse-mode.
        :rtype: :py:class:`psyclone.autodiff.transformations.ADReverseOMPRegionDirectiveTrans`
        """
        return self._omp_region_trans

    @omp_region_trans.setter
    def omp_region_trans(self, omp_region_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import (
            ADReverseOMPRegionDirectiveTrans,
        )

        if not isinstance(omp_region_trans, ADReverseOMPRegionDirectiveTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseOMPRegionDirectiveTrans' "
                f"but found '{type(omp_region_trans)}.__name__'."
            )

        self._omp_region_trans = omp_region_trans

    @property
    def recording(self):
        """Returns the recording routine being generated.

        :return: recording routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self.transformed[0]

    @property
    def returning(self):
        """Returns the returning routine being generated.

        :return: returning routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self.transformed[1]

    @property
    def reversing(self):
        """Returns the reversing routine being generated.

        :return: reversing routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self.transformed[2]

    @property
    def recording_table(self):
        """Returns the symbol table of the recording routine being generated.

        :return: recording routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.recording.symbol_table

    @property
    def returning_table(self):
        """Returns the symbol table of the returning routine being generated.

        :return: returning routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.returning.symbol_table

    @property
    def reversing_table(self):
        """Returns the symbol table of the reversing routine being generated.

        :return: reversing routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.reversing.symbol_table

    @property
    def recording_symbol(self):
        """Returns the routine symbol of the recording routine being generated.

        :return: recording routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return own_routine_symbol(self.recording)

    @property
    def returning_symbol(self):
        """Returns the routine symbol of the returning routine being generated.

        :return: returning routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return own_routine_symbol(self.returning)

    @property
    def reversing_symbol(self):
        """Returns the routine symbol of the reversing routine being generated.

        :return: reversing routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return own_routine_symbol(self.reversing)

    @property
    def value_tapes(self):
        """Returns the value tapes used by the transformation, as a \
        dictionnary containing ScalarType as a key and ADValueTape as a value.

        :return: value tapes dictionnary.
        :rtype: Dict[:py:class:`psyclone.psyir.symbols.ScalarType`, \
                     :py:class:`psyclone.autodiff.ADValueTape`]
        """
        return self._value_tapes

    @value_tapes.setter
    def value_tapes(self, value_tapes):
        if not isinstance(value_tapes, dict):
            raise TypeError(
                f"'value_tapes' argument should be of "
                f"type 'dict' but found "
                f"'{type(value_tapes).__name__}'."
            )
        for key, value in value_tapes.items():
            if not isinstance(key, ScalarType):
                raise TypeError(
                    f"'value_tape' argument should be of "
                    f"a dict with keys of type 'ScalarType' but found "
                    f"'{type(key).__name__}'."
                )
            if not isinstance(value, ADValueTape):
                raise TypeError(
                    f"'value_tape' argument should be of "
                    f"a dict with values of type 'ADValueTape' but found "
                    f"'{type(key).__name__}'."
                )
        self._value_tapes = value_tapes

    def get_value_tape_for(self, datanode):
        """Returns the value tape associated with the datanode datatype.

        :param datanode: datanode whose associated value tape to get.
        :type datanode: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if datanode is of the wrong type.
        :raises NotImplementedError: if the datatype of datanode is neither \
                                     ScalarType nor ArrayType.

        :return: the correct value tape to use for datanode.
        :rtype: :py:class:`psyclone.autodiff.ADValueTape`
        """
        if not isinstance(datanode, DataNode):
            raise TypeError(
                f"'datanode' argument should be of "
                f"type 'DataNode' but found "
                f"'{type(datanode).__name__}'."
            )
        datatype = datanode.datatype
        while isinstance(datatype, ArrayType):
            datatype = datatype.datatype
        # if isinstance(datanode.datatype, ScalarType):
        #     datatype = datanode.datatype
        # elif isinstance(datanode.datatype, ArrayType):
        #     datatype = datanode.datatype.datatype
        # else:
        #     raise NotImplementedError(
        #         "Only ScalarType and ArrayType datanodes "
        #         "taping are implemented but found "
        #         f"{datanode.datatype.__name__}."
        #     )

        return self.value_tapes[datatype]

    @property
    def control_tape(self):
        """Returns the control tape used by the transformation, or None if \
        it doesn't use one.

        :return: control tape or None.
        :rtype: Union[:py:class:`psyclone.autodiff.ADControlTape`, NoneType]
        """
        return self._control_tape

    @control_tape.setter
    def control_tape(self, control_tape):
        if not isinstance(control_tape, (ADControlTape, NoneType)):
            raise TypeError(
                f"'control_tape' argument should be of "
                f"type 'ADControlTape' or 'NoneType' but found "
                f"'{type(control_tape).__name__}'."
            )
        self._control_tape = control_tape

    @property
    def adjoint_symbols(self):
        """Returns all the adjoint symbols used in transforming the Routine,
            ie. adjoints of data symbols, operations and all temporary symbols.

        :return: list of all adjoint symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        """
        symbols = self.operation_adjoints  # + self.function_call_adjoints
        symbols += self.temp_symbols + list(
            self.data_symbol_differential_map.values()
        )
        return symbols

    def validate(
        self,
        routine,
        dependent_vars,
        independent_vars,
        value_tapes=None,
        control_tape=None,
        options=None,
    ):
        """Validates the arguments of the `apply` method.

        :param routine: routine Node to the transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param dependent_vars: list of dependent variables names to be \
                               differentiated.
        :type dependent_vars: `List[Str]`
        :param independent_vars: list of independent variables names to \
                                 differentiate with respect to.
        :type independent_vars: `List[Str]`
        :param value_tapes: value tapes to use to transform the routine \
                                 as a dictionnary, with stored element \
                                 ScalarType as key and ADTape as value.
        :type value_tapes: Optional[\
                            Union[NoneType,  \
                              Dict[
                                :py:class:`psyclone.psyir.symbols.ScalarType`, \
                                :py:class:`psyclone.autodiff.ADValueTape`]]]
        :param control_tape: control tape to use to transform the routine.
        :type control_tape: Optional[Union[NoneType, ADControlTape]]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if value_tape is of the wrong type.
        :raises TypeError: if control_tape is of the wrong type.
        """
        # pylint: disable=arguments-renamed, too-many-arguments

        super().validate(routine, dependent_vars, independent_vars, options)

        if not isinstance(value_tapes, (dict, NoneType)):
            raise TypeError(
                f"'value_tapes' argument should be of "
                f"type 'dict' but found "
                f"'{type(value_tapes).__name__}'."
            )
        if isinstance(value_tapes, dict):
            for key, value in value_tapes.items():
                if not isinstance(key, ScalarType):
                    raise TypeError(
                        f"'value_tape' argument should be of "
                        f"a dict with keys of type 'ScalarType' but found "
                        f"'{type(key).__name__}'."
                    )
                if not isinstance(value, ADValueTape):
                    raise TypeError(
                        f"'value_tape' argument should be of "
                        f"a dict with values of type 'ADValueTape' but found "
                        f"'{type(key).__name__}'."
                    )

        if not isinstance(control_tape, (ADControlTape, NoneType)):
            raise TypeError(
                f"'control_tape' argument should be of "
                f"type 'ADControlTape' or 'NoneType' but found an element of "
                f"type '{type(control_tape).__name__}'."
            )

    def apply(
        self,
        routine,
        dependent_vars,
        independent_vars,
        value_tapes=None,
        control_tape=None,
        options=None,
    ):
        """Applies the transformation, generating the recording and returning \
        routines that correspond to automatic differentiation of this Routine \
        using reverse-mode.

        | Options:
        | - bool 'jacobian': whether to generate the Jacobian routine. Defaults\
                           to False.
        | - bool 'verbose' : toggles explanatory comments. Defaults to False.
        | - bool 'simplify': True to apply simplifications after applying AD \
                           transformations. Defaults to True.
        | - int 'simplify_n_times': number of time to apply simplification \
                                  rules to BinaryOperation nodes. Defaults to 5.
        | - bool 'inline_operation_adjoints': True to inline all possible \
                                            operation adjoints definitions. \
                                            Defaults to True.

        :param routine: routine Node to the transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param dependent_vars: list of dependent variables names to be \
                               differentiated.
        :type dependent_vars: `List[Str]`
        :param independent_vars: list of independent variables names to \
                                 differentiate with respect to.
        :type independent_vars: `List[Str]`
        :param value_tapes: value tapes to use to transform the routine \
                                 as a dictionnary, with stored element \
                                 ScalarType as key and ADValueTape as value.
        :type value_tapes: Optional[\
                            Union[NoneType,  \
                              Dict[
                                :py:class:`psyclone.psyir.symbols.ScalarType`, \
                                :py:class:`psyclone.autodiff.ADValueTape`]]]
        :param control_tape: control tape to use to transform the routine.
        :type control_tape: Optional[Union[NoneType, ADControlTape]]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises NotImplementedError: if no transformation rule has yet been \
                                     implemented for one of the children of \
                                     routine.

        :return: couple composed of the recording and returning Routines \
                 that correspond to the transformation of this Routine.
        :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Routine`, \
                      :py:class:`psyclone.psyir.nodes.Routine`]
        """
        # pylint: disable=arguments-renamed, too-many-arguments

        self.validate(
            routine,
            dependent_vars,
            independent_vars,
            value_tapes,
            control_tape,
            options,
        )

        # Transformation can only be applied once
        self._was_applied = True

        self.routine = routine
        self.dependent_variables = dependent_vars
        self.independent_variables = independent_vars

        # Get the variables access information (to determine overwrites
        # and taping)
        self.variables_info = VariablesAccessInfo(routine)

        # Value tape for the transformation
        # create one for each datatype, unless it was provided as an argument
        # of the transformation
        self.value_tapes = value_tapes if value_tapes else dict()
        name = routine.name
        scalar_datatypes = []
        for datasymbol in routine.symbol_table.datasymbols:
            if isinstance(datasymbol.datatype, ScalarType):
                scalar_datatypes.append(datasymbol.datatype)
            elif isinstance(datasymbol.datatype, ArrayType):
                scalar_datatypes.append(datasymbol.datatype.datatype)
        for datatype in set(scalar_datatypes):
            if datatype not in self.value_tapes:
                self.value_tapes[datatype] = ADValueTape(name, datatype)

        # Control tape for the transformation
        # - none provided, create one
        if control_tape is None:  # and routine.walk(IfBlock) != []:
            name = routine.name
            self.control_tape = ADControlTape(
                name, self._default_control_tape_datatype
            )
        # - use the provided one
        else:
            self.control_tape = control_tape

        # Add this transformation to the container_trans map
        # Do it before apply below or ordering is not from outer to
        # inner routines
        self.container_trans.add_routine_trans(self)

        # Empty transformed routines with symbol tables
        self.transformed = self.create_transformed_routines()

        # Process all symbols in the table, generating adjoint symbols
        self.process_data_symbols(options)

        # Transform the statements found in the Routine
        self.transform_children(options)

        # Add the transformed routines symbols to the container_trans map
        self.container_trans.add_transformed_routines(
            self.routine_symbol, self.transformed_symbols
        )

        # Tape all the values that are not written back to the parent
        # routine/scope
        self.value_tape_non_written_values(options)

        # # Add the value tapes to the container_trans map
        # NoneTypeue_tapes:
        #     self.container_trans.add_value_tape(self.routine_symbol, value_tape)

        # # Add the control tape to the container_trans map
        # self.container_trans.add_control_tape(
        #     self.routine_symbol, self.control_tape
        # )

        # All dependent and independent variables names
        # list(set(...)) to avoid duplicates
        diff_variables = list(set(self.differential_variables))

        # Add the necessary adjoints as arguments of the returning routine
        self.add_adjoint_arguments(diff_variables, options)

        # Change the intents as needed
        # NOTE: this replaces the non-adjoint symbols in the returning
        # symbol table
        # so it breaks the adjoint map...
        self.set_argument_accesses(options)

        # Add the assignments of 0 to non-argument adjoints at the beginning of
        # the returning routine
        self.add_differentials_zero_assignments(self.returning, options)

        # ########################################################################
        # ########################################################################
        # # TODO: make this optional!
        # ########################################################################
        # ########################################################################
        # # Update the tapes usefully_recorded_flags lists
        # for tape in (self.value_tape,):
        #     useful_restorings = self.get_all_useful_restorings(tape)
        #     print(f"Useful restorings are: {[rest.debug_string() for rest in useful_restorings]}")
        #     tape.update_usefully_recorded_flags(useful_restorings)
        #     for useless in tape.get_useless_recordings() + tape.get_useless_restorings():
        #         print(f"Detaching {useless.debug_string()}")
        #         useless.detach()

        # Add the value_tape as argument of both routines iff it's actually used
        # and also ALLOCATE and DEALLOCATE it in the reversing routine if it's
        # a dynamic array
        for value_tape in self.value_tapes.values():
            if len(value_tape.recorded_nodes) != 0:
                self.add_tape_argument(value_tape, options)

        # Add the control_tape as argument of both routines iff it's actually
        # used
        # and also ALLOCATE and DEALLOCATE it in the reversing routine if it's
        # a dynamic array
        if self.control_tape is not None:
            self.add_tape_argument(self.control_tape, options)

        ########################################################################
        ########################################################################
        # FIXME: this shouldn't be here, only here to keep the initial tape
        # length for now
        ########################################################################
        ########################################################################
        post_process_tbr = self.unpack_option("post_process_tbr", options)

        if post_process_tbr:
            # Some arrays can be taped outside under some
            # Update the tapes usefully_recorded_flags lists
            for tape in self.value_tapes.values():
                useful_restorings = self.get_all_useful_restorings(tape)
                # print(
                #     f"Useful restorings are: {[rest.debug_string()
                #                              for rest in useful_restorings]}"
                # )
                tape.update_usefully_recorded_flags(useful_restorings)
                for useless in (
                    tape.get_useless_recordings()
                    + tape.get_useless_restorings()
                ):
                    # print(f"Detaching {useless.debug_string()}")
                    useless.detach()
        ########################################################################
        ########################################################################
        # FIXME: this shouldn't be here, only here to keep the initial tape
        # length for now
        ########################################################################
        ########################################################################

        # Postprocess (simplify, substitute operation adjoints) the recording
        # and returning routines
        self.postprocess(self.recording, options)
        self.postprocess(self.returning, options)

        # Combine the calls to recording and returning in reversing
        self.add_calls_to_reversing(options)

        # Add the three routines to the container
        for transformed in self.transformed:
            self.container_trans.container.addchild(transformed)

        jacobian = self.unpack_option("jacobian", options)

        if jacobian:
            jacobian_routine = self.jacobian_routine(
                "reverse", dependent_vars, independent_vars, options
            )
            self.container_trans.container.addchild(jacobian_routine)

        return self.recording, self.returning, self.reversing

    # TODO: when using arrays, it may make sense to check indices?
    def is_written_before(self, reference):
        """Checks whether the reference was written before. \
        This only considers it appearing as lhs of assignments.

        :param reference: reference to check.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if reference is of the wrong type.

        :return: True is there are writes before, False otherwise.
        :rtype: Bool
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of "
                f"type 'Reference' but found"
                f"'{type(reference).__name__}'."
            )

        # Get the signature and indices of the assignment lhs
        sig, _ = reference.get_signature_and_indices()
        # Get the variable info
        info = self.variables_info[sig]

        # Check whether it was written before this
        return info.is_written_before(reference)

    # TODO: consider the intents
    def is_call_argument_before(self, reference):
        """Checks whether the reference appeared before as an argument \
        of a routine call.
        This does not consider the intents of the arguments in the called \
        routine.

        :param reference: reference to check.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if reference is of the wrong type.

        :return: True if it appeared as a call argument before, False otherwise.
        :rtype: Bool
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of "
                f"type 'Reference' but found"
                f"'{type(reference).__name__}'."
            )

        # All preceding nodes
        preceding = reference.preceding(routine=True)
        # All preceding calls
        preceding_calls = [node for node in preceding if isinstance(node, Call)]

        # If the reference is in a call, get it
        # then remove it from the calls to consider.
        # Otherwise we'll always get True if reference is a call argument.
        parent_call = reference.ancestor(Call)
        if parent_call in preceding_calls:
            preceding_calls.remove(parent_call)

        # Check all arguments of the calls
        # if the same symbol as reference is present
        # then it was possibly written before as an argument
        for call in preceding_calls:
            refs = call.walk(Reference)
            syms = [ref.symbol for ref in refs]
            if reference.symbol in syms:
                return True

        return False

    def is_overwrite(self, reference, options=None):
        """Checks whether a reference was written before in the Routine
        being transformed or if appeared as a call argument before. 
        Used to determine whether to value_tape its prevalue or not.

        :param reference: reference to check.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`
        :param options: a dictionary with options for transformations, \
                       defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if reference is of the wrong type.

        :return: True if this reference is overwriting a prevalue, \
                 False otherwise.
        :rtype: Bool
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of "
                f"type 'Reference' but found"
                f"'{type(reference).__name__}'."
            )

        # Arguments with intent(in) cannot be assigned to/modified in calls
        if reference.symbol in self.recording_table.argument_list:
            if (
                reference.symbol.interface.access
                == ArgumentInterface.Access.READ
            ):
                return False

        # Check whether the reference is written before
        # this only considers assignments lhs it seems
        overwriting = self.is_written_before(reference)

        # Also check if it appears as argument in a call
        overwriting = overwriting or self.is_call_argument_before(reference)

        # Check whether it is an argument and has intent other than out
        variable_is_in_arg = (
            reference.symbol in self.routine_table.argument_list
            and reference.symbol.interface.access
            != ArgumentInterface.Access.WRITE
        )

        return overwriting or variable_is_in_arg

    def transform_assignment(self, assignment, options=None):
        """Transforms an Assignment child of the routine and returns the \
        statements to add to the recording and returning routines.

        :param assignment: assignment to transform.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if assignment is of the wrong type.

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this Assignment.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Assignment`], \
                List[:py:class:`psyclone.psyir.nodes.Assignment`]
        """
        if not isinstance(assignment, Assignment):
            raise TypeError(
                f"'assignment' argument should be of "
                f"type 'Assignment' but found"
                f"'{type(assignment).__name__}'."
            )

        overwriting = self.is_overwrite(assignment.lhs)

        recording = []
        returning = []

        # Tape record and restore first
        if overwriting:
            value_tape = self.get_value_tape_for(assignment.lhs)

            value_tape_record = value_tape.record(assignment.lhs)
            recording.append(value_tape_record)

            value_tape_restore = value_tape.restore(assignment.lhs)
            returning.append(value_tape_restore)

            # verbose_comment += ", overwrite"

        # Apply the transformation
        rec, ret = self.assignment_trans.apply(assignment, options)
        recording.extend(rec)
        returning.extend(ret)

        return recording, returning

        ## Insert in the recording routine
        # self.add_children(self.recording, recording)

        ## Insert in the returning routine
        # self.add_children(self.returning, returning, reverse=True)

    def transform_call(self, call, options=None):
        """Transforms a Call child of the routine and returns the \
        statements to add to the recording and returning routines.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if call is of the wrong type.

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this Call.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`], \
                List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of "
                f"type 'Call' but found"
                f"'{type(call).__name__}'."
            )

        recording = []
        returning = []

        # Call RoutineSymbol
        call_symbol = call.routine
        # Routine
        routine = self.container_trans.routine_from_symbol(call_symbol)
        routine_arguments = routine.symbol_table.argument_list

        # Tape record/restore the Reference arguments of the Call
        # Symbols already value_taped due to this call
        # to avoid taping multiple times if it appears as multiple
        # arguments of the call
        value_taped_symbols = []
        # accumulate the restores for now
        value_tape_restores = []
        for call_arg, routine_arg in zip(call.children, routine_arguments):
            if isinstance(call_arg, Reference):
                # Check whether the argument variable was written before
                # or if it is an argument of a call before
                overwriting = self.is_overwrite(call_arg)

                if overwriting and (
                    routine_arg.interface.access
                    is not ArgumentInterface.Access.READ
                ):
                    # Symbol wasn't value_taped yet
                    if call_arg.symbol not in value_taped_symbols:
                        value_tape = self.get_value_tape_for(call_arg)
                        # Tape record in the recording routine
                        value_tape_record = value_tape.record(call_arg)
                        recording.append(value_tape_record)

                        # Associated value_tape restore in the returning routine
                        value_tape_restore = value_tape.restore(call_arg)
                        value_tape_restores.append(value_tape_restore)

                        # Don't value_tape the same symbol again in this call
                        value_taped_symbols.append(call_arg.symbol)

        # Apply an ADReverseCallTrans
        rec, returning = self.call_trans.apply(call, options)

        # Add transformed call to recording statements
        recording.extend(rec)

        # Add value tapes restores before returning statements
        returning = value_tape_restores + returning

        return recording, returning

    def transform_if_block(self, if_block, options=None):
        """Transforms an IfBlock child of the routine and and returns the \
        statements to add to the recording and returning routines.

        :param if_block: if block to transform.
        :type if_block: :py:class:`psyclone.psyir.nodes.IfBlock`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if if_block is of the wrong type.

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this IfBlock.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`], \
                List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        if not isinstance(if_block, IfBlock):
            raise TypeError(
                f"'if_block' argument should be of "
                f"type 'IfBlock' but found"
                f"'{type(if_block).__name__}'."
            )

        recording = []
        returning = []

        # Tape record the condition value first
        control_tape_record = self.control_tape.record(if_block.condition)
        recording.append(control_tape_record)

        # Get the ArrayReference of the control tape element
        control_tape_ref = self.control_tape.restore(if_block.condition)

        # Apply the transformation
        rec, ret = self.if_block_trans.apply(
            if_block, control_tape_ref, options
        )

        recording.append(rec)
        returning.append(ret)

        return recording, returning

    def transform_loop(self, loop, options=None):
        """Transforms an Loop child of the routine and and returns the \
        statements to add to the recording and returning routines.

        :param loop: loop to transform.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if loop is of the wrong type.

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this Loop.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Loop`], \
                List[:py:class:`psyclone.psyir.nodes.Loop`]
        """
        if not isinstance(loop, Loop):
            raise TypeError(
                f"'loop' argument should be of "
                f"type 'Loop' but found"
                f"'{type(loop).__name__}'."
            )

        # Apply the transformation
        recording, returning = self.loop_trans.apply(loop, options)

        return recording, returning

    def transform_children(self, options=None):
        """Transforms all the children of the routine being transformed \
        and adds the statements to the transformed routines.

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
                (recording, returning) = self.transform_assignment(
                    child, options
                )
            elif isinstance(child, Call):
                (recording, returning) = self.transform_call(child, options)
            elif isinstance(child, IfBlock):
                (recording, returning) = self.transform_if_block(child, options)
            elif isinstance(child, Loop):
                (recording, returning) = self.transform_loop(child, options)
            elif isinstance(child, OMPRegionDirective):
                (recording, returning) = self.omp_region_trans.apply(
                    child, options
                )
            else:
                raise NotImplementedError(
                    f"Transforming a Routine child of "
                    f"type '{type(child).__name__}' is "
                    f"not implemented yet."
                )

            # Insert in the recording routine
            self.add_children(self.recording, recording)

            # Insert in the returning routine
            self.add_children(self.returning, returning, reverse=True)

    def new_operation_adjoint(self, operation):
        """Creates a new adjoint symbol for an Operation or IntrinsicCall node \
        in the returning table. Also appends it to the operation_adjoints list.

        :param operation: operation node.
        :type operation: Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]

        :raises TypeError: if operation is of the wrong type.

        :return: the adjoint symbol generated.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        if not isinstance(operation, (Operation, IntrinsicCall)):
            raise TypeError(
                f"'operation' argument should be of type "
                f"'Operation' or 'IntrinsicCall' but found "
                f"'{type(operation).__name__}'."
            )

        datatype = operation.datatype
        if not isinstance(datatype, (ScalarType, ArrayType)):
            raise TypeError(
                f"'datatype' attribute of 'operation' argument should be of "
                f"type 'ScalarType' or 'ArrayType' but found "
                f"'{type(datatype).__name__}'."
            )

        adjoint = self.returning_table.new_symbol(
            self._operation_adjoint_name,
            symbol_type=DataSymbol,
            datatype=datatype,
        )
        self.operation_adjoints.append(adjoint)

        return adjoint

    def inline_operation_adjoints(self, routine, options=None):
        """Inline the definitions of operations adjoints, ie. the RHS of 
        Assignment nodes with LHS being an operation adjoint, 
        everywhere it's possible in the 'routine', ie. except 
        for those used as Call arguments.

        :param routine: routine to simplify.
        :type routine: py:class:`psyclone.psyir.nodes.Routine`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=protected-access, too-many-locals

        # NOTE: must NOT be done for independent adjoints...
        all_assignments = routine.walk(Assignment)
        all_calls = routine.walk(Call)
        # Only assignments to operation adjoints
        op_adj_assignments = [
            assignment
            for assignment in all_assignments
            if assignment.lhs.name.startswith(self._operation_adjoint_name)
        ]
        for assignment in op_adj_assignments:
            call_args = []
            for call in all_calls:
                call_args.extend(call.children)
            if assignment.lhs in call_args:
                continue

            # Used to look at other assignments after this one only
            i = all_assignments.index(assignment)
            # Get the occurences of this operation adjoint on the rhs of
            # other assignments
            rhs_occurences = []
            for other_assignment in all_assignments[i + 1 :]:
                refs_in_rhs = other_assignment.rhs.walk(Reference)
                for ref in refs_in_rhs:
                    if ref == assignment.lhs:
                        rhs_occurences.append(ref)
                        # If already 1 occurence, we won't inline unless rhs
                        # is a Reference or Literal
                        # so stop there
                        if (
                            not isinstance(assignment.rhs, (Reference, Literal))
                        ) and (len(rhs_occurences) == 2):
                            break

            # Substitute if the RHS is a Reference or Literal
            # (to avoid unnecessary declarations of operation adjoints)
            # or if it only occurs once in a RHS.
            if len(rhs_occurences) == 1 or isinstance(
                assignment.rhs, (Reference, Literal)
            ):
                substitute = assignment.rhs.detach()
                assignment.detach()
                all_assignments.remove(assignment)
                # TODO: this might not be right for vectors...
                routine.symbol_table._symbols.pop(assignment.lhs.name)
                for rhs_occurence in rhs_occurences:
                    rhs_occurence.replace_with(substitute.copy())

    def postprocess(self, routine, options=None):
        """Apply postprocessing steps (simplification, operation adjoints 
        substitution) to the 'routine' argument.

        | Options:
        | - bool 'simplify': True to apply simplifications. Defaults to True.
        | - int 'simplify_n_times': number of time to apply simplification \
                                  rules to BinaryOperation nodes. Defaults to 5.
        | - bool 'inline_operation_adjoints': True to inline all possible \
                                            operation adjoints definitions. \
                                            Defaults to True.

        :param routine: routine to postprocess.
        :type routine: py:class:`psyclone.psyir.nodes.Routine`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """

        # Inline the operation adjoints definitions
        # (rhs of Assignment nodes whose LHS is an operation adjoint)
        # if only used once (unecessary declaration)
        inline_operation_adjoints = self.unpack_option(
            "inline_operation_adjoints", options
        )
        if inline_operation_adjoints:
            self.inline_operation_adjoints(routine, options)

        # Simplify the BinaryOperation and Assignment nodes
        # in the returning routine
        simplify = self.unpack_option("simplify", options)
        if simplify:
            self.simplify(routine, options)

        # Inline the operation adjoints again (in case the RHS simplified to a
        # Reference)
        # eg. 'op_adj = x_adj' should be substituted.
        if inline_operation_adjoints:
            self.inline_operation_adjoints(routine, options)

    def value_tape_non_written_values(self, options):
        """Record and restore the last values of non-argument REAL variables \
        using the value_tape.
        Indeed these are not returned by the call but could affect the results.

        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        
        """
        # pylint: disable=protected-access

        # Values that are not written back to the calling routine
        # are all non-arguments variables of REAL type.
        for var in self.recording_table.datasymbols:
            if var not in self.recording_table.argument_list:
                # "fake" reference to value_tape the last value
                # Either the whole array or the scalar variable
                if isinstance(var.datatype, ArrayType):
                    ref = ArrayReference.create(
                        var, [":"] * len(var.datatype.shape)
                    )
                    datatype = var.datatype.datatype
                else:
                    ref = Reference(var)
                    datatype = var.datatype

                # Record and restore in the right tape
                tape = self.value_tapes[datatype]
                value_tape_record = tape.record(ref)
                self.recording.addchild(value_tape_record)
                value_tape_restore = tape.restore(ref)
                self.returning.addchild(value_tape_restore, index=0)

    def set_argument_accesses(self, options):
        """Sets the intents of all non-adjoint arguments with original intents \
        different from intent(in) to intent(inout) in the returning routine.
        Indeed, all overwritable arguments are either recorded or returned by \
        the recording routine and restores or taken as argument by the \
        returning routine.
        Note that intent(in) in the returning routine would not allow tape \
        restores.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=protected-access

        # Every argument of the returning routine
        for arg in self.returning_table.argument_list:
            # Non-adjoint ones
            if arg in self.data_symbol_differential_map:
                # Only change intent(in)
                if arg.interface.access != ArgumentInterface.Access.READ:
                    returning_arg = DataSymbol(arg.name, arg.datatype)
                    # to intent(inout)
                    returning_arg.interface = ArgumentInterface(
                        ArgumentInterface.Access.READWRITE
                    )

                    # Swap in the returning table
                    # NOTE: SymbolTable.swap doesn't accept DataSymbols
                    name = self.returning_table._normalize(arg.name)
                    self.returning_table._symbols[name] = returning_arg

                    # Also swap in the returning argument list
                    index = self.returning_table._argument_list.index(arg)
                    self.returning_table._argument_list[index] = returning_arg

    def add_adjoint_arguments(self, diff_variables, options=None):
        """Add the adjoints of all differentiation variables \
        ie. dependent and independent ones \
        as intent(inout) arguments of the returning and reverting routines. \

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
            # Get the symbol associated to the name, then the adjoint symbol
            symbol = self.returning_table.lookup(
                var, scope_limit=self.returning
            )

            # Use the original symbol (not the copy) to get its adjoint
            adjoint_symbol = self.data_symbol_differential_map[symbol]
            adjoint_symbol.interface = ArgumentInterface(
                ArgumentInterface.Access.READWRITE
            )

            # Insert the adjoint in the returning argument list
            # After the argument
            self.add_to_argument_list(
                self.returning_table, adjoint_symbol, after=symbol
            )

            # Insert the adjoint in the reverting argument list
            self.reversing_table.add(adjoint_symbol)
            self.add_to_argument_list(
                self.reversing_table, adjoint_symbol, after=symbol
            )

    def add_tape_argument(self, tape, options=None):
        """Add a tape as argument of both the transformed routines.

        :param tape: tape to be added.
        :type tape: :py:class:`psyclone.autodiff.ADTape`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if tape is of the wrong type.
        """
        # pylint: disable=protected-access

        if not isinstance(tape, ADTape):
            raise TypeError(
                f"'tape' argument should be of "
                f"type 'ADTape' but found "
                f"'{type(tape).__name__}'."
            )

        use_sympy = self.unpack_option("use_sympy", options)

        # If using sympy, simplify the length expression of the tape
        if use_sympy:
            tape.simplify_length_expression_with_sympy(self.routine_table)

        # Get three symbols for the value_tape, one per routine
        symbols = [tape.symbol.copy() for i in range(3)]

        # Use the correct intents
        # intent(out) for the recording routine
        symbols[0].interface = ArgumentInterface(ArgumentInterface.Access.WRITE)
        # intent(in) for the returning routine
        symbols[1].interface = ArgumentInterface(ArgumentInterface.Access.READ)
        # The reversing routine declares the value_tape,
        # so the default AutomaticInterface is correct

        # Add the tape to all tables
        for table, symbol in zip(self.transformed_tables, symbols):
            table.add(symbol)

        # Append it to the arguments lists of the recording and returning
        # routines only
        for table, symbol in zip(self.transformed_tables[:-1], symbols[:-1]):
            table._argument_list.append(symbol)

            # # Get Reference nodes in said routine
            # references = table.node.walk(Reference)
            # # Check if one is a Reference to the tape do_offset symbol
            # # if so, add the symbol to the table
            # for ref in references:
            #     if ref.symbol == tape.do_offset_symbol:
            table.add(tape.do_offset_symbol)
            # break  # Only add it once

            # # Same for the offset_symbol
            # for ref in references:
            #     if ref.symbol == tape.offset_symbol:
            table.add(tape.offset_symbol)
            # break  # Only add it once

        # The tape is not an argument of the reversing routine

        # Add an assignment of the tape offset at the very beginning of the
        # recording and returning routines, respectively assigning 0 and its
        # last value to it
        self.recording.addchild(assign_zero(tape.offset), 0)
        self.returning.addchild(tape.offset_assignment, 0)

        # Also add ALLOCATE statements using the tape length at the beginning
        # of the reversing routine if it's a dynamic array
        if tape.is_dynamic_array:
            allocate = tape.allocate(tape.length)
            self.reversing.addchild(allocate, 0)
            deallocate = tape.deallocate()
            self.reversing.addchild(deallocate, len(self.reversing.children))

        # Apply simplifications using sympy if needed to:
        # - the indices/slices used in references to the tape
        # - the expressions assigned to the tape offsets
        # in both the recording and returning routines
        if use_sympy:
            for ref in self.recording.walk(ArrayReference):
                if ref.name == tape.name:
                    # print(f"Simplifying {ref.name}")
                    ref.replace_with(
                        tape.simplify_expression_with_sympy(
                            ref, self.recording_table
                        )
                    )
            for ref in self.returning.walk(ArrayReference):
                if ref.name == tape.name:
                    ref.replace_with(
                        tape.simplify_expression_with_sympy(
                            ref, self.returning_table
                        )
                    )
            for assignment in self.recording.walk(Assignment):
                if assignment.lhs.name in (
                    tape.do_offset_symbol.name,
                    tape.offset_symbol.name,
                ):
                    assignment.rhs.replace_with(
                        tape.simplify_expression_with_sympy(
                            assignment.rhs, self.recording_table
                        )
                    )
            for assignment in self.returning.walk(Assignment):
                if assignment.lhs.name in (
                    tape.do_offset_symbol.name,
                    tape.offset_symbol.name,
                ):
                    assignment.rhs.replace_with(
                        tape.simplify_expression_with_sympy(
                            assignment.rhs, self.returning_table
                        )
                    )

    def add_calls_to_reversing(self, options=None):
        """Inserts two calls, to the recording and returning routines, in the \
        reversing routine.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # Combine the recording and returning routines in the reversing routine
        call_rec = Call.create(
            own_routine_symbol(self.recording),
            [Reference(sym) for sym in self.recording_table.argument_list],
        )
        call_ret = Call.create(
            own_routine_symbol(self.returning),
            [Reference(sym) for sym in self.returning_table.argument_list],
        )
        self.reversing.addchild(call_rec)
        self.reversing.addchild(call_ret)

    def get_all_useful_restorings(self, tape):
        """Goes through the returning routine, all restorings from tape and
        all reads and returns a list of the ones that are actually useful.
        Used for post-processing TBR (to be recorded) analysis.

        :param tape: tape whose restorings to analyze.
        :type tape: :py:class:`psyclone.autodiff.ADTape`

        :raises TypeError: is tape is of the wrong type.

        :return: list of useful restorings nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        if not isinstance(tape, ADTape):
            raise TypeError(
                f"'tape' argument should be of type "
                f"'ADTape' but found "
                f"'{type(tape).__name__}'."
            )
        restorings_map = tape.get_recorded_symbols_to_restorings_map()
        reads_in_returning_map = tape.get_all_recorded_symbols_reads_in_routine(
            self.returning
        )

        # print("Symbols are:", [sym_name for sym_name
        #                        in restorings_map.keys()])

        assert set(restorings_map.keys()) == set(reads_in_returning_map.keys())

        useful_restorings = []
        for recorded_symbol_name in restorings_map.keys():
            restorings = restorings_map[recorded_symbol_name]
            reads = reads_in_returning_map[recorded_symbol_name]

            # print(f"Sym: {recorded_symbol_name}")
            # print(
            #     f"restorings: {[restoring.debug_string()
            #                     for restoring in restorings]}"
            # )
            # print(f"reads: {[read.parent.debug_string() for read in reads]}")

            very_last_node = self.returning
            while len(very_last_node.children) != 0:
                very_last_node = very_last_node.children[-1]

            for restoring, next_restoring in zip(
                restorings, restorings[1:] + [very_last_node]
            ):
                # print(f"restoring : {restoring.debug_string()}")
                # print(f"next_restoring : {next_restoring.debug_string()}")
                was_usefully_taped = False
                # reads_between_restorings = []

                # Easy case, restoring is not in a loop, look at reads between
                # it and the next_restoring (which may be in a loop itself)
                if restoring.ancestor(Loop) is None:
                    for read in reads:
                        if (
                            read.abs_position > restoring.abs_position
                            and read.abs_position < next_restoring.abs_position
                        ):
                            was_usefully_taped = True
                            # print("Useful, case 1")
                            break
                            # reads_between_restorings.append(read)
                # Trickier, restoring is in a loop
                else:
                    # If restoring and next_restoring are in the same loop,
                    # we only need to look at reads between them
                    if restoring.ancestor(Loop) is next_restoring.ancestor(
                        Loop
                    ):
                        for read in reads:
                            if (
                                read.abs_position > restoring.abs_position
                                and read.abs_position
                                < next_restoring.abs_position
                            ):
                                was_usefully_taped = True
                                # print("Useful, case 2")
                                break
                                # reads_between_restorings.append(read)
                    # If next_restoring is not in the same (nested) loop(s),
                    # then we need to look at all reads in the whole loops and
                    # outside of them, before next_restoring
                    # ie. we look between the position of the enclosing loop
                    # of restoring and that of next_restoring
                    else:
                        # Get the uppermost ancestor that is not among
                        # next_restoring's Schedule ancestor (in case restoring
                        # is done without assignment)
                        next_restoring_schedule_ancestors = [
                            next_restoring.ancestor(Schedule)
                        ]
                        cursor = next_restoring.ancestor(Schedule)
                        while cursor.ancestor(Schedule) is not None:
                            next_restoring_schedule_ancestors.append(
                                cursor.ancestor(Schedule)
                            )
                            cursor = cursor.ancestor(Schedule)
                        restoring_ancestor = restoring
                        while (
                            restoring_ancestor.parent
                            not in next_restoring_schedule_ancestors
                        ):
                            # if restoring_ancestor.parent is None:
                            #     print(
                            #         f"Was dealing with {recorded_symbol_name} "
                            #         f"and got restoring_ancestor "
                            #         f"{restoring_ancestor.debug_string()}"
                            #     )
                            restoring_ancestor = restoring_ancestor.parent

                        for read in reads:
                            if (
                                read.abs_position
                                > restoring_ancestor.abs_position
                                and read.abs_position
                                < next_restoring.abs_position
                            ):
                                was_usefully_taped = True
                                # print("Useful, case 3")
                                break
                                # reads_between_restorings.append(read)

                if was_usefully_taped:
                    useful_restorings.append(restoring)

                #     print(
                #         f"{recorded_symbol_name} was usefully restored "
                #         f"as {restoring.debug_string()}"
                #     )

                # else:
                #     print(f"!!{restoring.debug_string()} is useless")

            # The last value of an intent([in]out) or unknown intent argument
            # is possibly returned so keep the last restoring
            out_args_names = [
                sym.name
                for sym in self.returning_table.argument_list
                if sym.interface.access is not ArgumentInterface.Access.READ
            ]

            if (
                recorded_symbol_name in out_args_names
                and len(restorings) != 0
                and restorings[-1] not in useful_restorings
            ):
                useful_restorings.append(restorings[-1])

        return useful_restorings
