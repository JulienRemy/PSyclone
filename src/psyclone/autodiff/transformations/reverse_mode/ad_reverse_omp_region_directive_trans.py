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
differentiation of PSyIR OMPRegionDirective nodes."""

from psyclone.psyir.nodes import (
    OMPRegionDirective,
    OMPParallelDirective,
    OMPParallelDoDirective,
    OMPDoDirective,
    Assignment,
    Call,
    Loop,
    IfBlock,
    OMPDefaultClause,
    OMPPrivateClause,
    OMPSharedClause,
    OMPFirstprivateClause,
    Reference,
    Schedule,
)

from psyclone.psyir.symbols import ScalarType

from psyclone.autodiff.transformations import ADOMPRegionDirectiveTrans

# TODO: FirstPrivate, Reduction, Atomic, Barrier, etc.


class ADReverseOMPRegionDirectiveTrans(ADOMPRegionDirectiveTrans):
    """A class for automatic differentation transformations of \
    OMPRegionDirective nodes and derived classes using reverse-mode.
    Requires an ADReverseRoutineTrans instance as context, where the \
    derivative symbols can be found.
    """

    def apply(self, omp_region, options=None):
        """Applies the transformation, generating the transformed \
        statement associated with this OMPRegionDirective.
            
        | Options:
        | - bool 'verbose' : toggles preceding and inline comments around the \
            derivatives of assignment statements in the transformed motion.

        :param omp_region: omp region directive to be transformed.
        :type omp_region: :py:class:`psyclone.psyir.nodes.OMPRegionDirective`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this OMP region.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Statement`], \
                List[:py:class:`psyclone.psyir.nodes.Statement`]
        """
        self.validate(omp_region, options)

        clauses = omp_region.clauses
        recording_clauses = [clause.copy() for clause in clauses]
        returning_clauses, new_private_adjoints = self.transform_clauses(clauses, options)

        body = omp_region.dir_body
        # Transform the if body to get both motions
        (recording_body, returning_body) = self.transform_body(body, options)
        returning_body.reverse()

        # This is either an OMPDoDirective or OMPParallelDoDirective
        if isinstance(omp_region, (OMPDoDirective, OMPParallelDoDirective)):
            # In this case the directive body is only made of the loop and all
            # offset and taping assignments need to be moved outside of it
            for i, node in enumerate(recording_body):
                if isinstance(node, Loop):
                    recording_loop_index = i
                    recording_loop = node
                    break
            recording_pre_loop = recording_body[:recording_loop_index]
            if len(recording_body) >= recording_loop_index:
                recording_post_loop = recording_body[recording_loop_index + 1 :]
            else:
                recording_post_loop = []
            recording_region = type(omp_region)(
                children=[recording_loop],
                omp_schedule=omp_region.omp_schedule,
                collapse=omp_region.collapse,
                reprod=omp_region.reprod,
            )
            for i, clause in enumerate(recording_clauses):
                recording_region.children[i + 1] = clause
            recording = (
                recording_pre_loop + [recording_region] + recording_post_loop
            )

            for i, node in enumerate(returning_body):
                if isinstance(node, Loop):
                    returning_loop_index = i
                    returning_loop = node
                    break
            returning_pre_loop = returning_body[:returning_loop_index]
            if len(returning_body) >= returning_loop_index:
                returning_post_loop = returning_body[returning_loop_index + 1 :]
            else:
                returning_post_loop = []
            returning_region = type(omp_region)(
                children=[returning_loop],
                omp_schedule=omp_region.omp_schedule,
                collapse=omp_region.collapse,
                reprod=omp_region.reprod,
            )
            returning_region = self.assign_zero_to_new_private_differentials(
                returning_region, new_private_adjoints
            )
            for i, clause in enumerate(returning_clauses):
                returning_region.children[i + 1] = clause
            returning = (
                returning_pre_loop + [returning_region] + returning_post_loop
            )

            return recording, returning

        # or an OMPParallelDirective
        elif type(omp_region) is OMPParallelDirective:
            recording = OMPParallelDirective(children=recording_body)
            returning = OMPParallelDirective(children=returning_body)
            returning = self.assign_zero_to_new_private_differentials(
                returning, new_private_adjoints
            )
            for i, clause in enumerate(recording_clauses):
                recording.children[i + 1] = clause
            for i, clause in enumerate(returning_clauses):
                returning.children[i + 1] = clause
            return [recording], [returning]

    def transform_clauses(self, clauses, options=None):
        """Transforms all OpenMP clauses of this OMPRegionDirective.
        In reverse-move this replicates the primal variable clause for the \
        associated derivative variable.

        :param clauses: list of OpenMP clauses.
        :type clauses: List[ \
                Union[:py:class:`psyclone.psyir.nodes.OMPPrivateClause`, \
                      :py:class:`psyclone.psyir.nodes.OMPFirstPrivateClause`, \
                      :py:class:`psyclone.psyir.nodes.OMPSharedClause`, \
                      :py:class:`psyclone.psyir.nodes.OMPDefaultClause`]]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if clauses is of the wrong type.
        :raises TypeError: if an item of clauses is of the wrong type.
        :raises NotImplementedError: if an unsupported clause is included.

        :return: list of transformed clauses and list of new private adjoints
                 symbols.
        :rtype: List[Union[\
                        :py:class:`psyclone.psyir.nodes.OMPPrivateClause`,\
                        :py:class:`psyclone.psyir.nodes.OMPFirstPrivateClause`,\
                        :py:class:`psyclone.psyir.nodes.OMPSharedClause`,\
                        :py:class:`psyclone.psyir.nodes.OMPDefaultClause`]], \
                List[:py:class:`psyclone.psyir.nodes.DataSymbol`]
        """
        if not isinstance(clauses, list):
            raise TypeError(
                f"'clauses' argument should be of type "
                f"'list' but found '{type(clauses).__name__}'."
            )
        for clause in clauses:
            if not isinstance(
                clause,
                (
                    OMPPrivateClause,
                    OMPFirstprivateClause,
                    OMPSharedClause,
                    OMPDefaultClause,
                ),
            ):
                raise TypeError(
                    f"'clauses' argument should be a list of OMP clauses "
                    f"but found an item of type '{type(clause).__name__}'."
                )

        transformed_clauses = []
        new_private_adjoints = []
        for clause in clauses:
            if isinstance(clause, OMPPrivateClause):
                symbols = []
                for reference in clause.children:
                    symbol = reference.symbol
                    symbols.append(symbol)
                    if symbol.datatype.intrinsic is ScalarType.Intrinsic.REAL:
                        diff_symbol = (
                            self.routine_trans.data_symbol_differential_map[symbol]
                        )
                        symbols.append(diff_symbol)
                        new_private_adjoints.append(diff_symbol)
                transformed_clause = OMPPrivateClause.create(symbols)
                transformed_clauses.append(transformed_clause)
            elif isinstance(clause, OMPSharedClause):
                raise NotImplementedError(
                    "OMPSharedClause is unused for now."
                )
                # references = []
                # for reference in clause.children:
                #     symbol = reference.symbol
                #     diff_symbol = (
                #         self.routine_trans.data_symbol_differential_map[symbol]
                #     )
                #     references.append(reference.copy())
                #     references.append(Reference(diff_symbol))
                # transformed_clause = OMPSharedClause(children=references)
                # transformed_clauses.append(transformed_clause)
            elif isinstance(clause, OMPDefaultClause):
                transformed_clauses.append(clause.copy())
            elif isinstance(clause, OMPFirstprivateClause):
                if len(clause.children) == 0:
                    transformed_clauses.append(clause.copy())
                else:
                    raise NotImplementedError(
                        "OMPFirstprivateClause with actual "
                        "variables declared as such is not "
                        "implemented yet."
                    )
            else:
                raise NotImplementedError(
                    "Only OMPPrivateClause, "
                    "OMPSharedClause "
                    "and OMPDefaultClause are "
                    "implemented but found "
                    f"{type(clause).__name__}."
                )

        return transformed_clauses, new_private_adjoints

    def transform_body(self, body, options=None):
        """Transforms all statements found in the OMPRegionDirective body.

        :param body: body to transform, as a PSyIR Schedule.
        :type body: :py:class:`psyclone.psyir.nodes.Schedule`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if body is of the wrong type.

        :return: couple formed of two lists of transformed nodes, the first \
                 for the recording motion and the second for the returning one.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`],
                List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        if not isinstance(body, Schedule):
            raise TypeError(
                f"'body' argument should be of type "
                f"'Schedule' but found '{type(body).__name__}'."
            )

        recording_body = []
        returning_body = []

        for child in body.children:
            # Transform the child, get lists of nodes for both motions
            if isinstance(child, Assignment):
                (recording, returning) = (
                    self.routine_trans.transform_assignment(child, options)
                )
            elif isinstance(child, Call):
                (recording, returning) = self.routine_trans.transform_call(
                    child, options
                )
            elif isinstance(child, IfBlock):
                # Tape record the condition value first
                control_tape_record = self.routine_trans.control_tape.record(
                    child.condition
                )
                recording = [control_tape_record]

                # Get the ArrayReference of the control tape element
                control_tape_ref = self.routine_trans.control_tape.restore(
                    child.condition
                )
                (rec, returning) = self.routine_trans.transform_body(
                    child, control_tape_ref, options
                )
                recording.extend(rec)
            elif isinstance(child, Loop):
                (recording, returning) = self.routine_trans.transform_loop(
                    child, options
                )
            else:
                raise NotImplementedError(
                    f"Transformations for "
                    f"'{type(child).__name__}' found in "
                    f"IfBlock body were not implemented "
                    f"yet."
                )

            # Add to recording motion in same order
            recording_body.extend(recording)

            # Add to returning motion in reversed order and before the existing
            # statements
            returning.reverse()
            returning_body.extend(returning)

        return recording_body, returning_body
