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
        :rtype: :py:class:`psyclone.psyir.nodes.OMPRegionDirective`, \
                :py:class:`psyclone.psyir.nodes.OMPRegionDirective`
        """
        self.validate(omp_region, options)

        clauses = omp_region.clauses
        recording_clauses = [clause.copy() for clause in clauses]
        returning_clauses = self.transform_clauses(clauses, options)

        body = omp_region.dir_body
        # Transform the if body to get both motions
        (recording_body, returning_body) = self.transform_body(
            body, options
        )
        returning_body.reverse()

        recording_children = recording_body + recording_clauses
        returning_children = returning_body + returning_clauses

        # This is either an OMPDoDirective or OMPParallelDoDirective
        if isinstance(omp_region, OMPDoDirective):
            recording = type(omp_region)(
                children=recording_children,
                omp_schedule=omp_region.omp_schedule,
                collapse=omp_region.collapse,
                reprod=omp_region.reprod,
            )
            returning = type(omp_region)(
                children=returning_children,
                omp_schedule=omp_region.omp_schedule,
                collapse=omp_region.collapse,
                reprod=omp_region.reprod,
            )
            return recording, returning

        # or an OMPParallelDirective
        # if type(omp_region) is OMPParallelDirective:
        recording = OMPParallelDirective(children=recording_children)
        returning = OMPParallelDirective(children=returning_children)
        return recording, returning

    def transform_clauses(self, clauses, options=None):
        """Transforms all OpenMP clauses of this OMPRegionDirective.
        In reverse-move this replicates the primal variable clause for the \
        associated derivative variable.

        :param clauses: list of OpenMP clauses.
        :type clauses: List[ \
                Union[:py:class:`psyclone.psyir.nodes.OMPPrivateClause`, \
                      :py:class:`psyclone.psyir.nodes.OMPSharedClause`, \
                      :py:class:`psyclone.psyir.nodes.OMPDefaultClause`]]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if clauses is of the wrong type.
        :raises TypeError: if an item of clauses is of the wrong type.
        :raises NotImplementedError: if an unsupported clause is included.

        :return: list of transformed clauses.
        :rtype: List[Union[\
                        :py:class:`psyclone.psyir.nodes.OMPPrivateClause`,\
                        :py:class:`psyclone.psyir.nodes.OMPSharedClause`,\
                        :py:class:`psyclone.psyir.nodes.OMPDefaultClause`]]
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
                    #OMPFirstprivateClause,
                    OMPSharedClause,
                    OMPDefaultClause,
                ),
            ):
                raise TypeError(
                    f"'clauses' argument should be a list of OMP clauses "
                    f"but found an item of type '{type(clause).__name__}'."
                )

        transformed_clauses = []
        for clause in clauses:
            if isinstance(clause, OMPPrivateClause):
                symbols = []
                for reference in clause.children:
                    symbol = reference.symbol
                    diff_symbol = (
                        self.routine_trans.data_symbol_differential_map[symbol]
                    )
                    symbols.append(symbol)
                    symbols.append(diff_symbol)
                transformed_clause = OMPPrivateClause.create(symbols)
                transformed_clauses.append(transformed_clause)
            elif isinstance(clause, OMPSharedClause):
                references = []
                for reference in clause.children:
                    symbol = reference.symbol
                    diff_symbol = (
                        self.routine_trans.data_symbol_differential_map[symbol]
                    )
                    references.append(reference.copy())
                    references.append(Reference(diff_symbol))
                transformed_clause = OMPSharedClause(children=references)
                transformed_clauses.append(transformed_clause)
            elif isinstance(clause, OMPDefaultClause):
                transformed_clauses.append(clause.copy())
            else:
                raise NotImplementedError(
                    "Only OMPPrivateClause, "
                    "OMPSharedClause "
                    "and OMPDefaultClause are "
                    "implemented but found "
                    f"{type(clause).__name__}."
                )

        return transformed_clauses

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
