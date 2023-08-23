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

"""This module provides an abstract Transformation for automatic 
differentiation of PSyIR Schedule nodes."""

from abc import ABCMeta, abstractmethod

from psyclone.psyir.nodes import (
    Assignment,
    Call,
    Node,
    Schedule,
)
from psyclone.psyir.symbols import DataSymbol, SymbolTable
from psyclone.core import VariablesAccessInfo
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff import simplify_node
from psyclone.autodiff.transformations import ADTrans, ADContainerTrans


class ADScheduleTrans(ADTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of Schedule nodes.
    Requires an ADContainerTrans instance as context, where the definitions of
    the routines called inside the schedule to be transformed can be found.

    :param container_trans: ADContainerTrans context instance
    :type container_trans: :py:class:`psyclone.autodiff.transformations.ADContainerTrans`

    :raises TypeError: if the container_trans argument is of the wrong type.
    """

    _temp_name_prefix = "temp_"
    _temp_name_suffix = ""

    def __init__(self, container_trans):
        if not isinstance(container_trans, ADContainerTrans):
            raise TypeError(
                f"'container_trans' argument should be of type "
                f"'ADContainerTrans' but found "
                f"'{type(container_trans).__name__}'."
            )
        # Transformation can only be applied once
        self._was_applied = False

        self.container_trans = container_trans

        # Symbols for temporary variables
        self.temp_symbols = []

    @property
    def schedule(self):
        """Returns the schedule node being transformed.

        :return: schedule.
        :rtype: :py:class:`psyclone.psyir.nodes.Schedule`
        """
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        if not isinstance(schedule, Schedule):
            raise TypeError(
                f"'schedule' argument should be of "
                f"type 'Schedule' but found"
                f"'{type(schedule).__name__}'."
            )

        self._schedule = schedule

    @property
    def dependent_variables(self):
        """Names of the dependent variables used in transforming this Schedule. \
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
        """Names of the independent variables used in transforming this Schedule. \
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
        The list begins with independent variables. Names may not be unique in it.

        :return: list of all differential variables.
        :rtype: `List[Str]`
        """
        return self.dependent_variables + self.independent_variables

    @property
    def schedule_table(self):
        """Returns the symbol table of the schedule node being transformed.

        :return: symbol table.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.schedule.symbol_table

    @property
    def container_trans(self):
        """Returns the contextual ADContainerTrans instance this \
        transformation was initialized with.

        :return: container transformation.
        :rtype: :py:class:`psyclone.autodiff.transformation.ADContainerTrans`
        """
        return self._container_trans

    @container_trans.setter
    def container_trans(self, container_trans):
        if not isinstance(container_trans, ADContainerTrans):
            raise TypeError(
                f"'container_trans' argument should be of "
                f"type 'ADContainerTrans' but found"
                f"'{type(container_trans).__name__}'."
            )

        self._container_trans = container_trans

    @property
    def transformed(self):
        """Returns the transformed schedule(s) as a list.

        :return: list of transformed schedules.
        :rtype: List[:py:class:`psyclone.psyir.node.Schedule`]
        """
        return self._transformed

    @transformed.setter
    def transformed(self, transformed):
        if not isinstance(transformed, list):
            raise TypeError(
                f"'transformed' argument should be of "
                f"type 'List[Schedule]' but found "
                f"'{type(transformed).__name__}'."
            )
        for sym in transformed:
            if not isinstance(sym, Schedule):
                raise TypeError(
                    f"'transformed' argument should be of "
                    f"type 'List[Schedule]' but found "
                    f"an element of type "
                    f"'{type(sym).__name__}'."
                )
        self._transformed = transformed

    @property
    def transformed_tables(self):
        """Returns the symbol table(s) of the transformed schedules as a \
        list.

        :return: list of transformed schedule symbol tables.
        :rtype: List[:py:class:`psyclone.psyir.symbols.SymbolTable`]
        """
        return [schedule.symbol_table for schedule in self.transformed]

    @property
    def variables_info(self):
        """Returns the variables access information of the schedule being \
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

    def validate(self, schedule, dependent_vars, independent_vars, options=None):
        """Validates the arguments of the `apply` method.

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

        :raises TransformationError: if the transformation has already been applied.
        :raises TransformationError: if schedule is of the wrong type.
        :raises TypeError: if dependent_vars is of the wrong type.
        :raises TypeError: if at least one element of dependent_vars is \
            of the wrong type.
        :raises TypeError: if independent_vars is of the wrong type.
        :raises TypeError: if at least one element of independent_vars is \
            of the wrong type.
        """
        super().validate(schedule, options)

        if self._was_applied:
            raise TransformationError(
                "ADScheduleTrans instance can only be " "applied once."
            )

        if not isinstance(schedule, Schedule):
            raise TransformationError(
                f"'schedule' argument should be of "
                f"type 'Schedule' but found"
                f"'{type(schedule).__name__}'."
            )
        # TODO: extend this to functions and programs
        # - functions won't be pure if modifying the value_tape!
        # - programs would only work for ONE dependent variable,
        #   and only by making a single program out of the recording and returning schedules

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

    @abstractmethod
    def apply(self, schedule, dependent_vars, independent_vars, options=None):
        """Applies the transformation.

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
        """

    @abstractmethod
    def transform_assignment(self, assignment, options=None):
        """Transforms an Assignment child of the schedule.

        :param assignment: assignment to transform.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if assignment is of the wrong type.

        """
        if not isinstance(assignment, Assignment):
            raise TypeError(
                f"'assignment' argument should be of "
                f"type 'Assignment' but found"
                f"'{type(assignment).__name__}'."
            )

    @abstractmethod
    def transform_call(self, call, options=None):
        """Transforms a Call child of the schedule.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if call is of the wrong type.
        """
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of "
                f"type 'Call' but found"
                f"'{type(call).__name__}'."
            )

    def transform_children(self, options=None):
        """Transforms all the children of the schedule being transformed \
        and adds the statements to the transformed schedules.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises NotImplementedError: if a child is a recursive Call to the \
            Schedule being transformed.
        :raises NotImplementedError: if the child transformation is not \
            implemented yet. For now only those for Assignment and Call are.
        """
        # Go line by line through the Schedule
        # Note that this creates the symbols for operation adjoints and temporaries
        for child in self.schedule.children:
            if isinstance(child, Assignment):
                self.transform_assignment(child, options)
            elif isinstance(child, Call):
                self.transform_call(child, options)
            else:
                raise NotImplementedError(
                    f"Transforming a Schedule child of "
                    f"type '{type(child).__name__}' is "
                    f"not implemented yet."
                )
            
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

        :return: temporary symbol
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

        name = self._temp_name_prefix + symbol.name + self._temp_name_suffix

        temp_symbol = symbol_table.new_symbol(
            name, symbol_type=DataSymbol, datatype=symbol.datatype
        )
        self.temp_symbols.append(temp_symbol)

        return temp_symbol

    def add_children(self, schedule, children, reverse=False):
        """Adds the children from a list to a schedule.
        Inserts them in the order of the list if reverse is False, \
        in the reversed order and at index 0 otherwise.

        :param schedule: schedule to add children to.
        :type schedule: :py:class:`psyclone.psyir.nodes.Schedule`
        :param children: list of children to add.
        :type children: List[:py:class:`psyclone.psyir.nodes.Schedule`]
        :param reverse: whether to reverse and add at index 0, \
            defaults to False..
        :type reverse: bool, optional

        :raises TypeError: if schedule is of the wrong type.
        :raises TypeError: if children is of the wrong type.
        :raises TypeError: if some child is of the wrong type.
        :raises TypeError: if reverse is of the wrong type.
        """
        if not isinstance(schedule, Schedule):
            raise TypeError(
                f"'schedule' argument should be of type "
                f"'Schedule' but found "
                f"'{type(schedule).__name__}'."
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
            schedule.addchild(child, index)

    @abstractmethod
    def simplify(self, schedule, options=None):
        """Apply simplifications to the BinaryOperation and Assignment nodes
        of the transformed schedule provided as argument.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if schedule is of the wrong type.
        :raises ValueError: if schedule is not in self.transformed.
        """
        if not isinstance(schedule, Schedule):
            raise TypeError(
                f"'schedule' argument should be of type "
                f"'Schedule' but found "
                f"'{type(schedule).__name__}'."
            )
        if schedule not in self.transformed:
            raise ValueError(
                "'schedule' argument should be in " "self.transformed but is not."
            )

        simplify_n_times = self.unpack_option("simplify_n_times", options)
        for i in range(simplify_n_times):
            # Reverse the walk result to apply from deepest operations to shallowest
            all_nodes = schedule.walk(Node)[::-1]
            for i, node in enumerate(all_nodes):
                simplified_node = simplify_node(node)
                if simplified_node is None:
                    node.detach()
                    all_nodes.pop(i)
                else:
                    if simplified_node is not node:
                        node.replace_with(simplified_node)
                        all_nodes[i] = simplified_node
