# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2017-2022, Science and Technology Facilities Council.
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
# Authors R. W. Ford, A. R. Porter, S. Siso and N. Nobre, STFC Daresbury Lab
#         I. Kavcic, Met Office
#         J. Henrichs, Bureau of Meteorology
# -----------------------------------------------------------------------------

''' This module contains the GOcean-specific SymbolTable implementation.
It implements functions to access the iteration arguments and data arguments
for the GOcean API.
'''

from psyclone.errors import GenerationError
from psyclone.psyir.nodes import KernelSchedule
from psyclone.psyir.symbols import SymbolTable, ScalarType


class GOSymbolTable(SymbolTable):
    '''
    Sub-classes SymbolTable to provide a GOcean-specific implementation.
    '''

    def _check_gocean_conformity(self):
        '''
        Checks that the Symbol Table has at least 2 arguments which represent
        the iteration indices (are scalar integers).

        :raises GenerationError: if the Symbol Table does not conform to the \
                rules for a GOcean 1.0 kernel.
        '''
        # Get the kernel name if available for better error messages
        kname_str = ""
        if self._node and isinstance(self._node, KernelSchedule):
            kname_str = f" for kernel '{self._node.name}'"

        # Check that there are at least 2 arguments
        if len(self.argument_list) < 2:
            raise GenerationError(
                f"GOcean 1.0 API kernels should always have at least two "
                f"arguments representing the iteration indices but the "
                f"Symbol Table{kname_str} has only {len(self.argument_list)} "
                f"argument(s).")

        # Check that first 2 arguments are scalar integers
        for pos, posstr in [(0, "first"), (1, "second")]:
            dtype = self.argument_list[pos].datatype
            if not (isinstance(dtype, ScalarType) and
                    dtype.intrinsic == ScalarType.Intrinsic.INTEGER):
                raise GenerationError(
                    f"GOcean 1.0 API kernels {posstr} argument should be a "
                    f"scalar integer but got '{dtype}'{kname_str}.")

    @property
    def iteration_indices(self):
        '''In the GOcean API the two first kernel arguments are the iteration
        indices.

        :return: List of symbols representing the iteration indices.
        :rtype: list of :py:class:`psyclone.psyir.symbols.DataSymbol`
        '''
        self._check_gocean_conformity()
        return self.argument_list[:2]

    @property
    def data_arguments(self):
        '''In the GOcean API the data arguments start from the third item in
        the argument list.

        :return: List of symbols representing the data arguments.
        :rtype: list of :py:class:`psyclone.psyir.symbols.DataSymbol`
        '''
        self._check_gocean_conformity()
        return self.argument_list[2:]
