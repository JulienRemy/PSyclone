# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2022-2023, Science and Technology Facilities Council.
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
# Author: A. R. Porter, STFC Daresbury Lab
# Modified: R. W. Ford, STFC Daresbury Lab
#           J. Remy, Inria
# -----------------------------------------------------------------------------

''' This module contains the IntrinsicCall node implementation.'''

from collections import namedtuple
from enum import Enum

from psyclone.psyir.nodes.call import Call
from psyclone.psyir.nodes.datanode import DataNode
from psyclone.psyir.nodes.literal import Literal
from psyclone.psyir.nodes.reference import Reference
from psyclone.psyir.symbols import IntrinsicSymbol

from psyclone.psyir.symbols.datatypes import ScalarType, ArrayType


# pylint: disable=too-many-branches

class IntrinsicCall(Call):
    ''' Node representing a call to an intrinsic routine (function or
    subroutine). This can be found as a standalone statement
    or an expression.

    :param routine: the type of Intrinsic being created.
    :type routine: py:class:`psyclone.psyir.IntrinsicCall.Intrinsic`
    :param kwargs: additional keyword arguments provided to the PSyIR node.
    :type kwargs: unwrapped dict.

    :raises TypeError: if the routine argument is not an Intrinsic type.

    '''
    # Textual description of the node.
    _children_valid_format = "[DataNode]*"
    _text_name = "IntrinsicCall"
    _colour = "cyan"

    #: The type of Symbol this Call must refer to. Used for type checking in
    #: the constructor (of the parent class).
    _symbol_type = IntrinsicSymbol

    #: The intrinsics that can be represented by this node.
    Intrinsic = Enum('Intrinsic', [
        'ALLOCATE', 'DEALLOCATE', 'RANDOM_NUMBER', 'MINVAL', 'MAXVAL', "SUM",
        "TINY", "HUGE"])
    #: Named tuple for describing the properties of the required arguments to
    #: a particular intrinsic. If there's no limit on the number of arguments
    #: then `max_count` will be None.
    ArgDesc = namedtuple('ArgDesc', 'min_count max_count types')
    #: List of ArgDesc objects describing the required arguments, indexed
    #: by intrinsic name.
    _required_args = {}
    #: Dict of optional arguments, indexed by intrinsic name. Each optional
    #: argument is described by an entry in the Dict.
    _optional_args = {}
    # The positional arguments to allocate must all be References (or
    # ArrayReferences but they are a subclass of Reference).
    _required_args[Intrinsic.ALLOCATE] = ArgDesc(1, None, Reference)
    _optional_args[Intrinsic.ALLOCATE] = {
        # Argument used to specify the shape of the object being allocated.
        "mold": Reference,
        # Argument specifying both shape and initial value(s) for the object
        # being allocated.
        "source": Reference,
        # Integer variable given status value upon exit.
        "stat": Reference,
        # Variable in which message is stored upon error. (Requires that 'stat'
        # also be provided otherwise the program will just abort upon error.)
        "errmsg": Reference}
    _required_args[Intrinsic.DEALLOCATE] = ArgDesc(1, None, Reference)
    _optional_args[Intrinsic.DEALLOCATE] = {"stat": Reference}
    _required_args[Intrinsic.RANDOM_NUMBER] = ArgDesc(1, 1, Reference)
    _optional_args[Intrinsic.RANDOM_NUMBER] = {}
    _required_args[Intrinsic.MINVAL] = ArgDesc(1, 1, DataNode)
    _optional_args[Intrinsic.MINVAL] = {
        "dim": DataNode, "mask": DataNode}
    _required_args[Intrinsic.MAXVAL] = ArgDesc(1, 1, DataNode)
    _optional_args[Intrinsic.MAXVAL] = {
        "dim": DataNode, "mask": DataNode}
    _required_args[Intrinsic.SUM] = ArgDesc(1, 1, DataNode)
    _optional_args[Intrinsic.SUM] = {
        "dim": DataNode, "mask": DataNode}
    _required_args[Intrinsic.TINY] = ArgDesc(1, 1, (Reference, Literal))
    _optional_args[Intrinsic.TINY] = {}
    _required_args[Intrinsic.HUGE] = ArgDesc(1, 1, (Reference, Literal))
    _optional_args[Intrinsic.HUGE] = {}

    def __init__(self, routine, **kwargs):
        if not isinstance(routine, Enum) or routine not in self.Intrinsic:
            raise TypeError(
                f"IntrinsicCall 'routine' argument should be an "
                f"instance of IntrinsicCall.Intrinsic, but found "
                f"'{type(routine).__name__}'.")

        # A Call expects a symbol, so give it an intrinsic symbol.
        super().__init__(
            IntrinsicSymbol(
                routine.name,
                is_elemental=(routine in ELEMENTAL_INTRINSICS),
                is_pure=(routine in PURE_INTRINSICS)),
            **kwargs)
        self._intrinsic = routine

    @property
    def datatype(self):
        """Datatype of the intrinsic call result.

        :raises NotImplementedError: if the intrinsic is ALLOCATE or DEALLOCATE.
        :raises NotImplementedError: if the intrinsic is MINVAL, MAXVAL or SUM \
            and the DIM argument is not a literal.

        :return: Datatype of the intrinsic call result.
        :rtype: :py:class:`psyclone.psyir.symbols.DataType`
        """
        if self.intrinsic in (IntrinsicCall.Intrinsic.ALLOCATE,
                              IntrinsicCall.Intrinsic.DEALLOCATE):
            raise NotImplementedError("Datatypes of ALLOCATE and "
                                      "DEALLOCATE are not implemented yet.")
        
        if self.intrinsic in (IntrinsicCall.Intrinsic.RANDOM_NUMBER,
                              IntrinsicCall.Intrinsic.HUGE,
                              IntrinsicCall.Intrinsic.TINY):
            return self.children[0].datatype
        
        if self.intrinsic in (IntrinsicCall.Intrinsic.MINVAL,
                              IntrinsicCall.Intrinsic.MAXVAL,
                              IntrinsicCall.Intrinsic.SUM):
            # Array of rank 1 or one argument => scalar
            if (len(self.children[0].datatype.shape) == 1) \
                or (len(self.children) == 1):
                return ScalarType(self.children[0].datatype.intrinsic,
                                  self.children[0].datatype.precision)
            
            # Two or three arguments, the array and DIM and/or MASK
            # Second argument is DIM if it's scalar
            if isinstance(self.children[1].datatype, ScalarType):
                if not isinstance(self.children[1], Literal):
                    raise NotImplementedError("Calls to MINVAL, MAXVAL or SUM "
                                                "with a DIM argument other "
                                                "than a literal are not "
                                                "implemented yet.")
                scalar_type = ScalarType(self.children[0].datatype.intrinsic,
                                            self.children[0].datatype.precision)
                dim = int(self.children[1].value) # Fortran DIM
                shape = self.children[0].shape.copy().pop(dim - 1) # Python index
                return ArrayType(scalar_type, shape)

            # Second argument is MASK
            else:
                return ScalarType(self.children[0].datatype.intrinsic,
                                self.children[0].datatype.precision)

    @property
    def intrinsic(self):
        ''' Return the type of intrinsic.

        :returns: enumerated type capturing the type of intrinsic.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall.Intrinsic`

        '''
        return self._intrinsic

    @classmethod
    def create(cls, routine, arguments):
        '''Create an instance of this class given the type of routine and a
        list of nodes (or name-and-node tuples) for its arguments. Any
        named arguments *must* come after any required arguments.

        :param routine: the Intrinsic being called.
        :type routine: py:class:`psyclone.psyir.IntrinsicCall.Intrinsic`
        :param arguments: the arguments to this routine, and/or \
            2-tuples containing an argument name and the \
            argument. Arguments are added as child nodes.
        :type arguments: List[ \
            Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                  Tuple[str, :py:class:`psyclone.psyir.nodes.DataNode`]]]

        :returns: an instance of this class.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :raises TypeError: if any of the arguments are of the wrong type.
        :raises ValueError: if any optional arguments have incorrect names \
            or if a positional argument is listed *after* a named argument.
        :raises ValueError: if the number of supplied arguments is not valid \
            for the specified intrinsic routine.

        '''
        if not isinstance(routine, Enum) or routine not in cls.Intrinsic:
            raise TypeError(
                f"IntrinsicCall create() 'routine' argument should be an "
                f"instance of IntrinsicCall.Intrinsic but found "
                f"'{type(routine).__name__}'.")

        if not isinstance(arguments, list):
            raise TypeError(
                f"IntrinsicCall.create() 'arguments' argument should be a "
                f"list but found '{type(arguments).__name__}'")

        if cls._optional_args[routine]:
            optional_arg_names = sorted(list(
                cls._optional_args[routine].keys()))
        else:
            optional_arg_names = []

        # Validate the supplied arguments.
        reqd_args = cls._required_args[routine]
        opt_args = cls._optional_args[routine]
        last_named_arg = None
        pos_arg_count = 0
        for arg in arguments:
            if isinstance(arg, tuple):
                if not isinstance(arg[0], str):
                    raise TypeError(
                        f"Optional arguments to an IntrinsicCall must be "
                        f"specified by a (str, Reference) tuple but got "
                        f"a {type(arg[0]).__name__} instead of a str.")
                name = arg[0].lower()
                last_named_arg = name
                if not optional_arg_names:
                    raise ValueError(
                        f"The '{routine.name}' intrinsic does not support "
                        f"any optional arguments but got '{name}'.")
                if name not in optional_arg_names:
                    raise ValueError(
                        f"The '{routine.name}' intrinsic supports the "
                        f"optional arguments {optional_arg_names} but got "
                        f"'{name}'")
                if not isinstance(arg[1], opt_args[name]):
                    raise TypeError(
                        f"The optional argument '{name}' to intrinsic "
                        f"'{routine.name}' must be of type "
                        f"'{opt_args[name].__name__}' but got "
                        f"'{type(arg[1]).__name__}'")
            else:
                if last_named_arg:
                    raise ValueError(
                        f"Found a positional argument *after* a named "
                        f"argument ('{last_named_arg}'). This is invalid.'")
                if not isinstance(arg, reqd_args.types):
                    raise TypeError(
                        f"The '{routine.name}' intrinsic requires that "
                        f"positional arguments be of type '{reqd_args.types}' "
                        f"but got a '{type(arg).__name__}'")
                pos_arg_count += 1

        if ((reqd_args.max_count is not None and
             pos_arg_count > reqd_args.max_count)
                or pos_arg_count < reqd_args.min_count):
            msg = f"The '{routine.name}' intrinsic requires "
            if reqd_args.max_count is not None and reqd_args.max_count > 0:
                msg += (f"between {reqd_args.min_count} and "
                        f"{reqd_args.max_count} ")
            else:
                msg += f"at least {reqd_args.min_count} "
            msg += f"arguments but got {len(arguments)}."
            raise ValueError(msg)

        # Create an intrinsic call and add the arguments
        # afterwards. We can't call the parent create method as it
        # assumes the routine argument is a symbol and therefore tries
        # to create an intrinsic call with this symbol, rather than
        # the intrinsic enum.
        call = IntrinsicCall(routine)
        call._add_args(call, arguments)
        call._intrinsic = routine

        return call


# TODO #658 this can be removed once we have support for determining the
# type of a PSyIR expression.
# Intrinsics that perform a reduction on an array.
REDUCTION_INTRINSICS = [
    IntrinsicCall.Intrinsic.SUM, IntrinsicCall.Intrinsic.MINVAL,
    IntrinsicCall.Intrinsic.MAXVAL]

# Intrinsics that, provided with an input array, apply their operation
# individually to each of the array elements and return an array with
# the results.
ELEMENTAL_INTRINSICS = []

# Intrinsics that are 'pure' functions. Given the same input arguments, a pure
# function will always return the same value.
PURE_INTRINSICS = [IntrinsicCall.Intrinsic.SUM,
                   IntrinsicCall.Intrinsic.MINVAL,
                   IntrinsicCall.Intrinsic.MAXVAL,
                   IntrinsicCall.Intrinsic.TINY,
                   IntrinsicCall.Intrinsic.HUGE]
