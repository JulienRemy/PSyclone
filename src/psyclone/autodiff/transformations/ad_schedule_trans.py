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
from psyclone.psyir.symbols import DataSymbol, SymbolTable, REAL_DOUBLE_TYPE
from psyclone.core import VariablesAccessInfo
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff import simplify_node
from psyclone.autodiff.transformations import ADScopeTrans, ADContainerTrans


class ADScheduleTrans(ADScopeTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of Schedule nodes.
    Requires an ADContainerTrans instance as context, where the definitions of
    the routines called inside the schedule to be transformed can be found.
    """

    def create_transformed_schedules(self):
        """Create the empty transformed Schedules.

        :return: all transformed schedules as a list.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Schedule`]
        """
        # NOTE: subclasses need to redefine the _number_of_schedules
        # class attribute.

        # Shallow copy the symbol table
        tables = [self.schedule_table.shallow_copy() 
                  for i in range(self._number_of_schedules)]
        original_table = self.schedule_table.shallow_copy().detach()
        tables = [table.detach() for table in tables]
        original_table.attach(self.schedule)

        # Create the schedules
        schedules = [Schedule(children=[], symbol_table=table) for table in tables]

        return schedules 

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
