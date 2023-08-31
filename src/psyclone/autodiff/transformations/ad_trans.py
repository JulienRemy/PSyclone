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

"""This module provides an abstract Transformation for reverse-mode automatic 
differentiation in `psyclone.autodiff`. This is the parent class of all 
`AD...Trans` classes."""

from abc import ABCMeta

from psyclone.psyGen import Transformation


class ADTrans(Transformation, metaclass=ABCMeta):
    """An abstract base class for all `psyclone.autodiff` transformations.
    Default options for `apply` are defined here, as a class attribute, \
    as well as option unpacking from a `dict`.
    """

    _default_options = {
        "verbose": False,
        "jacobian": False,
        "simplify": True,
        "simplify_n_times": 5,
        "inline_operation_adjoints": True,
    }

    @property
    def default_options(self):
        """The default options of this transformation.

        :return: dictionnary of options, with strings as keys.
        :rtype: dict[str]
        """
        return self._default_options

    @default_options.setter
    def default_options(self, default_options):
        self.typecheck_options(default_options)
        self._default_options = default_options

    @staticmethod
    def typecheck_options(options):
        """Check if an options dictionnary is of the right type, \
        that is it is a dict and its keys are all strings.

        :param options: options to check.
        :type options: object
        :raises TypeError: if options is neither a dict nor None.
        :raises TypeError: if a key of options is not a string.
        """
        if not isinstance(options, (dict, type(None))):
            raise TypeError(
                f"'options' argument should be of "
                f"type 'dict' or 'NoneType' but found "
                f"'{type(options).__name__}'."
            )
        if options is not None:
            for key in options:
                if not isinstance(key, str):
                    raise TypeError(
                        f"'options' argument should be a 'dict' "
                        f"with keys of type 'str' but found a key "
                        f"'{key}' of type '{type(key).__name__}'."
                    )

    def unpack_option(self, name, options):
        """Get the option with key 'name', from the 'options' dict \
        if it's in it, or then from the instance 'default_options' dict \
        if it's in it.

        :param name: name of the option to unpack.
        :type name: str
        :param options: dictionnary of options to search in.
        :type options: dict[str, object]

        :raises TypeError: if name is of the wrong type.
        :raises TypeError: if options is of the wrong type.
        :raises TypeError: if options is not None and one of its keys is of \
            the wrong type.
        :raises KeyError: if name is not a key of options and defaults is None.
        :raises KeyError: if name is neither a key of options not of defaults.
        :return: the value of the option.
        :rtype: object
        """
        if not isinstance(name, str):
            raise TypeError(
                f"'name' argument should be of "
                f"type 'str' but found "
                f"'{type(name).__name__}'."
            )
        self.typecheck_options(options)

        if (options is not None) and (name in options):
            return options[name]

        if name not in self.default_options:
            raise KeyError(
                f"'name' argument '{name}' "
                f"is neither a key of 'options' nor "
                f"of 'self.default_options'."
            )
        return self.default_options[name]

    def validate(self, node, options=None):
        """Validates the options arguments.

        :param node: node to transform.
        :type node: depends on the ADTrans subclass.
        :param options: a dictionary with options for transformations.
        :type options: Optional[Dict[Str, Any]]
        """
        super().validate(node, options)
        self.typecheck_options(options)
