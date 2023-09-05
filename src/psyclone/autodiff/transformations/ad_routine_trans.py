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

"""This module provides an abstract Transformation for automatic differentiation
of PSyIR Routine nodes.
"""

from abc import ABCMeta, abstractmethod

from psyclone.core import VariablesAccessInfo

from psyclone.psyir.nodes import (
    Node,
    Routine,
    Call,
    Reference,
    ArrayReference,
    Literal,
    Assignment,
)
from psyclone.psyir.symbols import (
    INTEGER_TYPE,
    REAL_TYPE,
    SymbolTable,
    DataSymbol,
    ArrayType,
)
from psyclone.psyir.symbols.interfaces import (
    ArgumentInterface,
    AutomaticInterface,
)
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff import assign_zero, own_routine_symbol, assign, one
from psyclone.autodiff import simplify_node
from psyclone.autodiff.transformations import ADTrans


class ADRoutineTrans(ADTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of \
    Routine nodes."""

    # pylint: disable=too-many-instance-attributes, too-many-public-methods

    # Pre- and postfix for temporary variables symbols names
    _temp_name_prefix = "temp_"
    _temp_name_postfix = ""

    # Default PSyIR datatype for the derivatives/adjoints
    # TODO: use the dependent variable type and precision
    _default_differential_datatype = REAL_TYPE

    # Attributes that need to be redefined by subclasses
    _number_of_routines = 0
    _differential_prefix = ""
    _differential_postfix = ""
    _differential_table_index = 0
    _routine_prefixes = tuple()
    _routine_postfixes = tuple()

    # Pre- and postfix for the jacobian routine symbol name
    _jacobian_prefix = ""
    _jacobian_postfix = "_jacobian"

    def __init__(self):
        # Transformation can only be applied once
        self._was_applied = False

        # Symbols for temporary variables
        self.temp_symbols = []

        # DataSymbol => derivative DataSymbol
        self.data_symbol_differential_map = dict()

    @property
    @abstractmethod
    def container_trans(self):
        """Contextual container transformation."""

    @property
    @abstractmethod
    def assignment_trans(self):
        """Used assignment transformation."""

    @property
    @abstractmethod
    def operation_trans(self):
        """Used operation transformation."""

    @property
    @abstractmethod
    def call_trans(self):
        """Used call transformation."""

    @property
    def routine(self):
        """Returns the routine node being transformed.

        :return: routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self._routine

    @routine.setter
    def routine(self, routine):
        if not isinstance(routine, Routine):
            raise TypeError(
                f"'routine' argument should be of "
                f"type 'Routine' but found"
                f"'{type(routine).__name__}'."
            )

        self._routine = routine

    @property
    def routine_table(self):
        """Returns the symbol table of the routine node being transformed.

        :return: symbol table.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.routine.symbol_table

    @property
    def routine_symbol(self):
        """Returns the symbol of the routine node being transformed.

        :return: routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return self.routine_table.lookup_with_tag("own_routine_symbol")

    @property
    def dependent_variables(self):
        """Names of the dependent variables used in transforming this Routine. \
        These are the variables being differentiated.

        :return: list of names.
        :rtype: `List[Str]`
        """
        return self._dependent_variables

    @dependent_variables.setter
    def dependent_variables(self, dependent_vars):
        if not isinstance(dependent_vars, list):
            raise TypeError(
                f"'dependent_vars' argument should be of "
                f"type 'List[Str]' but found "
                f"'{type(dependent_vars).__name__}'."
            )
        for var in dependent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'dependent_vars' argument should be of "
                    f"type 'List[Str]' but found "
                    f"an element of type "
                    f"'{type(var).__name__}'."
                )
        self._dependent_variables = dependent_vars

    @property
    def independent_variables(self):
        """Names of the independent variables used in transforming this Routine.
        These are the variables with respect to which we are differentiating.

        :return: list of names.
        :rtype: `List[Str]`
        """
        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, independent_vars):
        if not isinstance(independent_vars, list):
            raise TypeError(
                f"'independent_vars' argument should be of "
                f"type 'List[Str]' but found "
                f"'{type(independent_vars).__name__}'."
            )
        for var in independent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'independent_vars' argument should be of "
                    f"type 'List[Str]' but found "
                    f"an element of type "
                    f"'{type(var).__name__}'."
                )
        self._independent_variables = independent_vars

    @property
    def differential_variables(self):
        """Names of all differential variables, both dependent and independent.
        The list begins with independent variables. \
        Names may not be unique in it.

        :return: list of all differential variables.
        :rtype: `List[Str]`
        """
        return self.dependent_variables + self.independent_variables

    @property
    def transformed(self):
        """Returns the transformed routine(s) as a list.

        :return: list of transformed routines.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Routine`]
        """
        return self._transformed

    @transformed.setter
    def transformed(self, transformed):
        if not isinstance(transformed, list):
            raise TypeError(
                f"'transformed' argument should be of "
                f"type 'List[Routine]' but found "
                f"'{type(transformed).__name__}'."
            )
        for sym in transformed:
            if not isinstance(sym, Routine):
                raise TypeError(
                    f"'transformed' argument should be of "
                    f"type 'List[Routine]' but found "
                    f"an element of type "
                    f"'{type(sym).__name__}'."
                )
        self._transformed = transformed

    @property
    def transformed_tables(self):
        """Returns the symbol table(s) of the transformed routines as a \
        list.

        :return: list of transformed routine symbol tables.
        :rtype: List[:py:class:`psyclone.psyir.symbols.SymbolTable`]
        """
        return [routine.symbol_table for routine in self.transformed]

    @property
    def transformed_symbols(self):
        """Returns the routine symbols of the  3 transformed routines as a \
        list, these being the recording routine, the returning routine and \
        the reversing routine.

        :return: list of transformed routine symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbols.RoutineSymbol`]
        """
        return [own_routine_symbol(routine) for routine in self.transformed]

    @property
    def variables_info(self):
        """Returns the variables access information of the routine being \
        transformed.

        :return: variables access information.
        :rtype: :py:class:`psyclone.core.VariablesAccessInfo`
        """
        return self._variables_info

    @variables_info.setter
    def variables_info(self, variables_info):
        if not isinstance(variables_info, VariablesAccessInfo):
            raise TypeError(
                f"'variables_info' argument should be of "
                f"type 'VariablesAccessInfo' but found "
                f"'{type(variables_info).__name__}'."
            )
        self._variables_info = variables_info

    def process_data_symbols(self, options=None):
        """Process all the data symbols of the symbol table, \
        generating their derivative/adjoint symbols in the transformed table \
        and adding them to the data_symbol_differential_map.

        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """

        for symbol in self.routine_table.datasymbols:
            self.create_differential_symbol(symbol, options)

    def create_differential_symbol(self, datasymbol, options=None):
        """Create the derivative/adjoint symbol of the argument symbol in the \
        transformed table.

        :param datasymbol: data symbol whose derivative/adjoint to create.
        :param datasymbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if datasymbol is of the wrong type.

        :return: the derivative/adjoint symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        if not isinstance(datasymbol, DataSymbol):
            raise TypeError(
                f"'datasymbol' argument should be of "
                f"type 'DataSymbol' but found"
                f"'{type(datasymbol).__name__}'."
            )
        if not datasymbol.is_scalar:
            raise NotImplementedError(
                "'datasymbol' is not a scalar. "
                "Arrays are not implemented yet."
            )

        # NOTE: subclasses need to redefine the
        # _differential_prefix, _differential_postfix and _differential_table_index
        # class attributes.

        # TODO: use the dependent variable type and precision
        # TODO: this would depend on the result of activity analysis
        # Name using pre- and postfix
        differential_name = self._differential_prefix + datasymbol.name
        differential_name += self._differential_postfix
        # New adjoint symbol with unique name in the correct transformed table
        differential = self.transformed_tables[
            self._differential_table_index
        ].new_symbol(
            differential_name,
            symbol_type=DataSymbol,
            datatype=self._default_differential_datatype,
        )

        # Add it to the map
        self.data_symbol_differential_map[datasymbol] = differential

        return differential

    def validate(
        self, routine, dependent_vars, independent_vars, options=None
    ):
        """Validates the arguments of the `apply` method.

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
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if routine is of the wrong type.
        :raises NotImplementedError: if routine is a program.
        :raises NotImplementedError: if routine is a function.
        :raises NotImplementedError: if routine contains a recursive call \
                                     (to itself).
        :raises TransformationError: if the SymbolTable of routine doesn't \
                                     contain a symbol for each name in \
                                     independent_var.
        :raises TransformationError: if the SymbolTable of routine doesn't \
                                     contain a symbol for each name in \
                                     dependent_var.
        :raises TransformationError: if the argument list of routine doesn't \
                                     contain an argument of correct Access for \
                                     each name in independent_var.
        :raises TransformationError: if the argument list of routine doesn't \
                                     contain an argument of correct Access for \
                                     each name in dependent_var.
        """
        # pylint: disable=arguments-differ

        super().validate(routine, options)

        if self._was_applied:
            raise TransformationError(
                "ADRoutineTrans instance can only be applied once."
            )

        if not isinstance(routine, Routine):
            raise TransformationError(
                f"'routine' argument should be of "
                f"type 'Routine' but found"
                f"'{type(routine).__name__}'."
            )
        # TODO: extend this to functions and programs
        # - functions won't be pure if modifying the value_tape!
        # - programs would only work for ONE dependent variable,
        #   and only by making a single program out of the recording and
        #   returning routines
        if routine.is_program:
            raise NotImplementedError(
                "'routine' argument is a program, "
                "this is not implemented yet. "
                "For now ADRoutineTrans only transforms "
                "Fortran subroutines."
            )
        if routine.return_symbol is not None:
            raise NotImplementedError(
                "'routine' argument is a function, "
                "this is not implemented yet."
                "For now ADRoutineTrans only transforms "
                "Fortran subroutines."
            )

        if not isinstance(dependent_vars, list):
            raise TypeError(
                f"'dependent_vars' argument should be of "
                f"type 'list' but found"
                f"'{type(dependent_vars).__name__}'."
            )
        for var in dependent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'dependent_vars' argument should be of "
                    f"type 'List[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )

        if not isinstance(independent_vars, list):
            raise TypeError(
                f"'independent_vars' argument should be of "
                f"type 'list' but found"
                f"'{type(independent_vars).__name__}'."
            )
        for var in independent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'independent_vars' argument should be of "
                    f"type 'List[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )

        # Avoid dealing with recursive calls for now
        # TODO: these actually should work when using a joint reversal routine
        # TODO: make the link between the routine and itself always be weak?
        for call in routine.walk(Call):
            if call.routine.name == routine.name:
                raise NotImplementedError(
                    f"Found a recursive Call inside "
                    f"Routine '{self.routine.name}'."
                    f"This is not implemented yet."
                )

        symbol_table = routine.symbol_table
        data_symbols = symbol_table.symbols
        symbol_names = [symbol.name for symbol in data_symbols]

        args = symbol_table.argument_list
        dependent_args_names = []
        independent_args_names = []
        for arg in args:
            if arg.interface.access == ArgumentInterface.Access.READ:
                independent_args_names.append(arg.name)
            elif arg.interface.access == ArgumentInterface.Access.WRITE:
                # NOTE: This would also include routine.return_symbol
                # if it is a function
                dependent_args_names.append(arg.name)
            else:  # READWRITE or UNKNOWN
                independent_args_names.append(arg.name)
                dependent_args_names.append(arg.name)

        for var in dependent_vars:
            if var not in symbol_names:
                raise TransformationError(
                    f"Dependent variable name '{var}'"
                    f"was not found among the "
                    f"Symbol names in the Routine "
                    f"SymbolTable."
                )
            if var not in dependent_args_names:
                raise TransformationError(
                    f"Dependent variable name '{var}'"
                    f"was not found among the "
                    f"Routine arguments with "
                    f"ArgumentInterface.Access WRITE,"
                    f"READWRITE or UNKNOWN."
                )

        for var in independent_vars:
            if var not in symbol_names:
                raise TransformationError(
                    f"Independent variable name '{var}'"
                    f"was not found among the "
                    f"Symbol names in the Routine "
                    f"SymbolTable."
                )
            if var not in independent_args_names:
                raise TransformationError(
                    f"Inependent variable name '{var}'"
                    f"was not found among the "
                    f"Routine arguments with "
                    f"ArgumentInterface.Access READ,"
                    f"READWRITE or UNKNOWN."
                )

    @abstractmethod
    def apply(self, routine, dependent_vars, independent_vars, options=None):
        """Applies the transformation.

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
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=arguments-differ, unnecessary-pass
        pass

    def new_temp_symbol(self, symbol, symbol_table):
        """Creates a new temporary symbol for the symbol argument.
        Uses the name of the symbol.
        Inserts it in symbol_table with an unused name and in the temp_symbol \
        list.

        :param symbol: symbol for which a temporary symbol should be created.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param symbol_table: symbol table in which to insert it.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :raises TypeError: if symbol is of the wrong type.
        :raises TypeError: if symbol_table is of the wrong type.

        :return: temporary symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        if not isinstance(symbol, DataSymbol):
            raise TypeError(
                f"'symbol' argument should be of type "
                f"'DataSymbol' but found "
                f"'{type(symbol).__name__}'."
            )

        if not isinstance(symbol_table, SymbolTable):
            raise TypeError(
                f"'symbol_table' argument should be of type "
                f"'SymbolTable' but found "
                f"'{type(symbol_table).__name__}'."
            )

        name = self._temp_name_prefix + symbol.name + self._temp_name_postfix

        temp_symbol = symbol_table.new_symbol(
            name, symbol_type=DataSymbol, datatype=symbol.datatype
        )
        self.temp_symbols.append(temp_symbol)

        return temp_symbol

    def add_children(self, routine, children, reverse=False):
        """Adds the children from a list to a routine.
        Inserts them in the order of the list if reverse is False, \
        in the reversed order and at index 0 otherwise.

        :param routine: routine to add children to.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param children: list of children to add.
        :type children: List[:py:class:`psyclone.psyir.nodes.Routine`]
        :param reverse: whether to reverse and add at index 0, \
                        defaults to False.
        :type reverse: Optional[Bool]

        :raises TypeError: if routine is of the wrong type.
        :raises TypeError: if children is of the wrong type.
        :raises TypeError: if some child is of the wrong type.
        :raises TypeError: if reverse is of the wrong type.
        """
        if not isinstance(routine, Routine):
            raise TypeError(
                f"'routine' argument should be of type "
                f"'Routine' but found "
                f"'{type(routine).__name__}'."
            )
        if not isinstance(children, list):
            raise TypeError(
                f"'children' argument should be of type "
                f"'list' but found "
                f"'{type(children).__name__}'."
            )
        for child in children:
            if not isinstance(child, Node):
                raise TypeError(
                    f"Elements of 'children' argument list "
                    f"should be of type 'Node' but found "
                    f"'{type(child).__name__}'."
                )
        if not isinstance(reverse, bool):
            raise TypeError(
                f"'reverse' argument should be of type "
                f"'bool' but found "
                f"'{type(reverse).__name__}'."
            )

        # Leave the argument list unchanged
        children_copy = children.copy()

        index = None
        if reverse:
            children_copy.reverse()
            index = 0

        for child in children_copy:
            routine.addchild(child, index)

    def create_transformed_routines(self):
        """Create the empty transformed Routines.

        :return: all transformed routines as a list.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Routine`]
        """
        # NOTE: subclasses need to redefine the _number_of_routines
        # class attribute.

        # Shallow copy the symbol table
        tables = [
            self.routine_table.shallow_copy()
            for i in range(self._number_of_routines)
        ]
        original_table = self.routine_table.shallow_copy().detach()
        tables = [table.detach() for table in tables]
        # TODO: make this cleaner...
        original_table.attach(self.routine)

        # Remove the 'own_routine_symbol' symbols from their tables
        # This is required to use new names for the routines
        for table in tables:
            table.remove(table.lookup_with_tag("own_routine_symbol"))

        # Names using pre- and postfixes
        names = [
            pre + self.routine.name + post
            for pre, post in zip(
                self._routine_prefixes, self._routine_postfixes
            )
        ]

        # Create the routines
        routines = [
            Routine.create(
                name=name,
                symbol_table=table,
                children=[],
                is_program=False,
                return_symbol_name=None,
            )
            for name, table in zip(names, tables)
        ]

        return routines

    @abstractmethod
    def transform_assignment(self, assignment, options=None):
        """Transforms an Assignment child of the routine.

        :param assignment: assignment to transform.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """

    @abstractmethod
    def transform_call(self, call, options=None):
        """Transforms a Call child of the routine.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """

    def transform_children(self, options=None):
        """Transforms all the children of the routine being transformed \
        and adds the statements to the transformed routines.

        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises NotImplementedError: if a child is a recursive Call to the \
                                     Routine being transformed.
        :raises NotImplementedError: if the child transformation is not \
                                     implemented yet. For now only those for \
                                     Assignment and Call instances are.
        """
        # Go line by line through the Routine
        for child in self.routine.children:
            if isinstance(child, Assignment):
                self.transform_assignment(child, options)
            elif isinstance(child, Call):
                self.transform_call(child, options)
            else:
                raise NotImplementedError(
                    f"Transforming a Routine child of "
                    f"type '{type(child).__name__}' is "
                    f"not implemented yet."
                )

    def simplify(self, routine, options=None):
        """Apply simplifications to the BinaryOperation and Assignment nodes
        of the transformed routine provided as argument.

        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if routine is of the wrong type.
        :raises ValueError: if routine is not in self.transformed.
        """
        if not isinstance(routine, Routine):
            raise TypeError(
                f"'routine' argument should be of type "
                f"'Routine' but found "
                f"'{type(routine).__name__}'."
            )
        if routine not in self.transformed:
            raise ValueError(
                "'routine' argument should be in "
                "self.transformed but is not."
            )

        simplify_n_times = self.unpack_option("simplify_n_times", options)
        for i in range(simplify_n_times):
            # Reverse the walk result to apply from deepest operations to
            # shallowest
            all_nodes = routine.walk(Node)[::-1]
            for i, node in enumerate(all_nodes):
                simplified_node = simplify_node(node)
                # Simplification yields None if node should be removed
                if simplified_node is None:
                    node.detach()
                    all_nodes.pop(i)
                else:
                    # Replace the node if different
                    if simplified_node is not node:
                        node.replace_with(simplified_node)
                        all_nodes[i] = simplified_node

    def add_to_argument_list(self, symbol_table, argument, after=None):
        """Adds the argument to the symbol table's argument list, if it has \
        the correct interface.
        The argument is added after another if 'after' is provided or \
        appended at the end otherwise.

        :param symbol_table: symbol table whose argument_list will be augmented.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param argument: argument symbol to add.
        :type argument: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param after: optional argument symbol after which to insert, 
                      defaults to None.
        :type after: Optional[Union[:py:class:`psyclone.psyir.symbols.
                                               DataSymbol`, 
                                    `NoneType`]]
        :raises TypeError: if symbol_table is of the wrong type.
        :raises TypeError: if argument is of the wrong type.
        :raises TypeError: if argument's interface is not an ArgumentInterface.
        :raises TypeError: if after is of the wrong type.
        :raises ValueError: if after is not None and is not in the argument \
                            list.
        """
        # Accessing private SymbolTable._argument_list to avoid property check.
        # pylint: disable=protected-access

        if not isinstance(symbol_table, SymbolTable):
            raise TypeError(
                f"'symbol_table' argument should be of type "
                f"'SymbolTable' but found "
                f"'{type(symbol_table).__name__}'."
            )
        if not isinstance(argument, DataSymbol):
            raise TypeError(
                f"'argument' argument should be of type "
                f"'DataSymbol' but found "
                f"'{type(argument).__name__}'."
            )
        if not isinstance(argument.interface, ArgumentInterface):
            raise TypeError(
                f"'argument' argument's interface should be of type "
                f"'ArgumentInterface' but found "
                f"'{type(argument.interface).__name__}'."
            )
        if not isinstance(after, (DataSymbol, type(None))):
            raise TypeError(
                f"'after' argument should be of type "
                f"'DataSymbol' or 'NoneType' but found "
                f"'{type(after).__name__}'."
            )
        if (after is not None) and (after not in symbol_table._argument_list):
            raise ValueError(
                f"'after' argument DataSymbol named {after.name} "
                f"is not in the argument_list of symbol_table."
            )

        argument_list = symbol_table._argument_list

        if after is None:
            argument_list.append(argument)
        else:
            index = argument_list.index(after) + 1
            argument_list.insert(index, argument)

    # TODO: this is a mess.
    def jacobian_routine(
        self, mode, dependent_vars, independent_vars, options=None
    ):
        """Creates the Jacobian routine using automatic \
        differentation for the transformed routine and lists of \
        dependent and independent variables names.
        Options:
        - bool 'verbose' : preceding comment for the routine.

        :param mode: mode to use. Can be either 'forward' or 'reverse'.
        :type mode: str
        :param dependent_vars: list of dependent variables names to be \
                               differentiated.
        :type dependent_vars: `List[str]`
        :param independent_vars: list of independent variables names to \
                                 differentiate with respect to.
        :type independent_vars: `List[str]`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if mode is of the wrong type.
        :raises ValueError: if mode is neither 'forward' nor 'reverse'.
        :raises TypeError: if dependent_vars is of the wrong type.
        :raises TypeError: if at least one element of dependent_vars is \
                           of the wrong type.
        :raises TypeError: if independent_vars is of the wrong type.
        :raises TypeError: if at least one element of independent_vars is \
                           of the wrong type.
        :raises ValueError: if at least one element of dependent_vars is \
                            not in self.dependent_variables, so was not used \
                            in transforming the routine.
        :raises ValueError: if at least one element of independent_vars is \
                            not in self.independent_variables, so was not used \
                            in transforming the routine.

        :return: the routine computing the Jacobian.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        if not isinstance(mode, str):
            raise TypeError(
                f"'mode' argument should be of "
                f"type 'str' but found"
                f"'{type(mode).__name__}'."
            )
        if mode not in ("forward", "reverse"):
            raise ValueError(
                f"'mode' argument should either 'forward' or 'reverse' "
                f"but found '{mode}'."
            )
        if not isinstance(dependent_vars, list):
            raise TypeError(
                f"'dependent_vars' argument should be of "
                f"type 'list' but found"
                f"'{type(dependent_vars).__name__}'."
            )
        for var in dependent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'dependent_vars' argument should be of "
                    f"type 'List[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )
            if var not in self.dependent_variables:
                raise ValueError(
                    f"'dependent_vars' argument contains variable name {var} "
                    f"but it was not used as a dependent variable when "
                    f"transforming the routine."
                )

        if not isinstance(independent_vars, list):
            raise TypeError(
                f"'independent_vars' argument should be of "
                f"type 'list' but found"
                f"'{type(independent_vars).__name__}'."
            )
        for var in independent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'independent_vars' argument should be of "
                    f"type 'List[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )
            if var not in self.independent_variables:
                raise ValueError(
                    f"'independent_vars' argument contains variable name {var} "
                    f"but it was not used as a dependent variable when "
                    f"transforming the routine."
                )

        dependent_diff_symbols = []
        independent_diff_symbols = []

        if mode == "forward":
            transformed_table = self.transformed_tables[0]
            transformed_symbol = self.transformed_symbols[0]
        else:
            transformed_table = self.transformed_tables[2]  # reversing
            transformed_symbol = self.transformed_symbols[2]  # reversing

        jacobian_routine = Routine(
            self._jacobian_prefix + self.routine.name + self._jacobian_postfix
        )
        symbol_table = jacobian_routine.symbol_table

        diff_map = dict()
        diff_symbol_names = []
        other_args = []

        # All differential variables (dependent & independent)
        # They are added as arguments of jacobian with the same intent
        for var in dependent_vars + independent_vars:
            # Get the symbol from the reversing routine
            sym = transformed_table.lookup(var)

            # Add it to the jacobian table and as argument (same intent)
            if sym not in symbol_table._argument_list:
                symbol_table.add(sym)
                symbol_table._argument_list.append(sym)

                # Get and copy the associated diff symbol
                diff_sym_copy = self.data_symbol_differential_map[sym].copy()
                # Switch it to non-argument interface
                diff_sym_copy.interface = AutomaticInterface()
                # Add to the value_tape
                symbol_table.add(diff_sym_copy)

                # Keep track of the diffs associated with the differential
                # variables names
                diff_map[var] = diff_sym_copy
                diff_symbol_names.extend([sym.name, diff_sym_copy.name])

        # Remaining arguments of the reversing routine need to be added too
        for sym in transformed_table.argument_list:
            if sym.name not in diff_symbol_names:
                symbol_table.add(sym)
                symbol_table._argument_list.append(sym)
                other_args.append(sym.name)

        # Lists of diff symbols to fill the jacobian
        for var in independent_vars:
            diff_sym = diff_map[var]
            independent_diff_symbols.append(diff_sym)
        for var in dependent_vars:
            diff_sym = diff_map[var]
            dependent_diff_symbols.append(diff_sym)

        # Some arguments of the jacobian routine with intent(inout) or unknown
        # in the reversing routine could be overwritten
        # Store and restore them as needed
        temp_assigns = []
        temp_restores = []
        for arg in symbol_table._argument_list:
            # Filter out the diffs
            # if arg not in self.data_symbol_differential_map.values():
            # (dependent_diff_symbols + independent_diff_symbols):
            if arg.interface.access in (
                ArgumentInterface.Access.READWRITE,
                ArgumentInterface.Access.UNKNOWN,
            ):
                temp = symbol_table.new_symbol(
                    "temp_" + arg.name,
                    symbol_type=DataSymbol,
                    datatype=arg.datatype,
                )
                temp_assigns.append(assign(temp, arg))
                temp_restores.append(assign(arg, temp))
        self.add_children(jacobian_routine, temp_assigns)

        # Jacobian matrix symbol, with intent(out)
        rows = len(dependent_vars)
        cols = len(independent_vars)
        jacobian = symbol_table.new_symbol(
            "J_" + self.routine.name,
            symbol_type=DataSymbol,
            datatype=ArrayType(
                self._default_differential_datatype, [cols, rows]
            ),
        )
        jacobian.interface = ArgumentInterface(ArgumentInterface.Access.WRITE)
        symbol_table._argument_list.append(jacobian)

        if mode == "forward":
            first_diffs = independent_diff_symbols
            second_diffs = dependent_diff_symbols
        else:
            first_diffs = dependent_diff_symbols
            second_diffs = independent_diff_symbols

        for first_dim, first_diff in enumerate(first_diffs):
            # Restore overwritten arguments of the jacobian routine
            # First first_dim => first call so no restores
            if first_dim != 0:
                self.add_children(
                    jacobian_routine, [rest.copy() for rest in temp_restores]
                )

            # first_dim + 1 to get the Fortran index
            first_dim_literal = Literal(str(first_dim + 1), INTEGER_TYPE)

            # Set the independent derivative/dependent adjoint for the row/column to 1.0
            jacobian_routine.addchild(
                assign(first_diff, one(first_diff.datatype))
            )

            # Set all other independent derivatives/dependent adjoints to 0.0
            for other_first_diff in first_diffs:
                if other_first_diff != first_diff:
                    jacobian_routine.addchild(assign_zero(other_first_diff))

            # Set all dependent derivatives/independent adjoints to 0.0
            # TODO: check the indep = dep case
            for second_diff in second_diffs:
                if second_diff != first_diff:
                    jacobian_routine.addchild(assign_zero(second_diff))

            # Create the argument list from the transformed one
            rev_args = [
                Reference(sym) for sym in transformed_table.argument_list
            ]
            # Create the call, add it to the jacobian routine
            call = Call.create(transformed_symbol, rev_args)
            jacobian_routine.addchild(call)

            # Insert every independent diff at the right location
            # in the jacobian matrix
            for second_dim, second_diff in enumerate(second_diffs):
                # second_dim + 1 to get the Fortran index
                second_dim_literal = Literal(str(second_dim + 1), INTEGER_TYPE)

                if mode == "forward":
                    col_literal = first_dim_literal
                    row_literal = second_dim_literal
                else:
                    col_literal = second_dim_literal
                    row_literal = first_dim_literal

                jacobian_ref = ArrayReference.create(
                    jacobian, [col_literal.copy(), row_literal.copy()]
                )

                jacobian_routine.addchild(assign(jacobian_ref, second_diff))

        # Verbose description writes the dependent variables (columns),
        # the independent variables (rows), the other arguments to specify,
        # and the derivatives of the jacobian matrix as d_/d_
        verbose = self.unpack_option("verbose", options)

        if verbose:
            jacobian_routine.preceding_comment = (
                f"Independent variables as columns: {independent_vars}.\n! "
                + f"Dependent variables as rows: {dependent_vars}.\n! "
            )
            if len(other_args) != 0:
                jacobian_routine.preceding_comment += (
                    f"Also specify: {other_args}."
                )
            for dep in dependent_vars:
                jacobian_routine.preceding_comment += "\n! "
                for indep in independent_vars:
                    jacobian_routine.preceding_comment += f"d{dep}/d{indep} "

        return jacobian_routine
