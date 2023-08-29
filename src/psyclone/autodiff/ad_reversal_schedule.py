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

"""This module provides a class for specifying reversal schedules in 
reverse-mode automatic differentation using psyclone.autodiff."""

from abc import ABCMeta, abstractmethod


class ADReversalSchedule(object, metaclass=ABCMeta):
    """Abstract class describing a reversal schedule for reverse-mode
    automatic differentiation call trees.
    """

    # pylint: disable=useless-object-inheritance

    @abstractmethod
    def is_strong_link(self, parent_routine_name, child_routine_name):
        """Abstract method. Determines if the link between two routines \
        is strong or not.
        Strong links lead to split reversals.
        Weak links lead to joint reversals.

        :param parent_routine_name: name of the parent (calling) routine.
        :type parent_routine_name: Str
        :param child_routine_name: name of the child (called) routine.
        :type child_routine_name: Str
        """


class ADSplitReversalSchedule(ADReversalSchedule):
    """Class describing a split reversal schedule for reverse-mode
    automatic differentiation.
    This means that all links are strong.
    Children routines follow the recording or returning motion of their parents.
    """

    # pylint: disable=too-few-public-methods

    def is_strong_link(self, parent_routine_name, child_routine_name):
        """Determines if the link between two routines \
        is strong or not.
        Strong links lead to split reversals.
        Weak links lead to joint reversals.

        :param parent_routine_name: name of the parent (calling) routine.
        :type parent_routine_name: Str
        :param child_routine_name: name of the child (called) routine.
        :type child_routine_name: Str

        :raises TypeError: if parent_routine_name is of the wrong type.
        :raises TypeError: if child_routine_name is of the wrong type.

        :return: True, all links are strong in a split reversal.
        :rtype: Bool
        """
        if not isinstance(parent_routine_name, str):
            raise TypeError(
                f"'parent_routine_name' argument in ADReversalSchedule "
                f"is_strong_link method "
                f"should be of type 'str' but found "
                f"'{type(parent_routine_name).__name__}'."
            )
        if not isinstance(child_routine_name, str):
            raise TypeError(
                f"'child_routine_name' argument in ADReversalSchedule "
                f"is_strong_link method "
                f"should be of type 'str' but found "
                f"'{type(child_routine_name).__name__}'."
            )

        return True


class ADJointReversalSchedule(ADReversalSchedule):
    """Class describing a joint reversal schedule for reverse-mode
    automatic differentiation. 
    This means that all links are weak.
    Children routines advance without recording when their parent routine \
    is recording and reverse (record then immendiatly return) when their \
    parent routine is returning.
    """

    # pylint: disable=too-few-public-methods

    def is_strong_link(self, parent_routine_name, child_routine_name):
        """Determines if the link between two routines \
        is strong or not.
        Strong links lead to split reversals.
        Weak links lead to joint reversals.

        :param parent_routine_name: name of the parent (calling) routine.
        :type parent_routine_name: Str
        :param child_routine_name: name of the child (called) routine.
        :type child_routine_name: Str

        :raises TypeError: if parent_routine_name is of the wrong type.
        :raises TypeError: if child_routine_name is of the wrong type.

        :return: False, all links are weak in a joint reversal.
        :rtype: Bool
        """
        if not isinstance(parent_routine_name, str):
            raise TypeError(
                f"'parent_routine_name' argument in ADReversalSchedule "
                f"is_strong_link method "
                f"should be of type 'str' but found "
                f"'{type(parent_routine_name).__name__}'."
            )
        if not isinstance(child_routine_name, str):
            raise TypeError(
                f"'child_routine_name' argument in ADReversalSchedule "
                f"is_strong_link method "
                f"should be of type 'str' but found "
                f"'{type(child_routine_name).__name__}'."
            )

        return False


class ADLinkReversalSchedule(ADReversalSchedule):
    """Class describing a reversal schedule described by links between \
    parent and child routines for reverse-mode automatic differentiation. 
    Strong links mean that children routines follow the recording or returning \
    motion of their parents.
    Weak links that children routines advance without recording when their \
    parent routine is recording and reverse (record then immendiatly return) \
    when their parent routine is returning.

    :param strong_links: list of [parent_name, child_name] strong links,    \
                         defaults to None.
    :type strong_links: Optional[Union[List[List[Str]], NoneType]]
    :param weak_links: list of [parent_name, child_name] weak links, \
                       defaults to None.
    :type weak_links: Optional[Union[List[List[Str]], NoneType]]
    :param default_link: default link type, for links not found among the \
                         two preceding list, defaults to None.
    :type default_link: Optional[Union[Str, NoneType]]

    :raises TypeError: if an argument is of the wrong type.
    :raises ValueError: if a list in strong_links or weak_links is not of \
                        length 2.
    :raises ValueError: if default_link is not "strong", "weak" or None.
    :raises ValueError: if all three of strong_links, weak_links and \
                        default_link are None.
    :raises ValueError: if a link is present in both strong_links and \
                        weak_links.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, strong_links=None, weak_links=None, default_link=None):
        # pylint: disable=too-many-branches

        if not isinstance(strong_links, (list, type(None))):
            raise TypeError(
                f"'strong_links' argument in ADReversalSchedule constructor "
                f"should be of type 'list[list[str]]' or 'NoneType' but found "
                f"'{type(strong_links).__name__}'."
            )
        if strong_links is not None:
            for link in strong_links:
                if not isinstance(link, list):
                    raise TypeError(
                        f"'strong_links' argument in ADReversalSchedule "
                        f"constructor should be of type 'list[list[str]]' "
                        f"but found 'list[{type(link).__name__}]'."
                    )
                if len(link) != 2:
                    raise ValueError(
                        f"'strong_links' argument in ADReversalSchedule "
                        f"constructor should be of type "
                        f"'list[lists of 2 elements]' but found "
                        f"a list of length {len(link)}."
                    )
                for name in link:
                    if not isinstance(name, str):
                        raise TypeError(
                            f"'strong_links' argument in ADReversalSchedule "
                            f"constructor should be of type 'list[list[str]]' "
                            f"but found 'list[list[{type(name).__name__}]]'."
                        )
        if not isinstance(weak_links, (list, type(None))):
            raise TypeError(
                f"'weak_links' argument in ADReversalSchedule constructor "
                f"should be of type 'list[list[str]]' or 'NoneType' but found "
                f"'{type(weak_links).__name__}'."
            )
        if weak_links is not None:
            for link in weak_links:
                if not isinstance(link, list):
                    raise TypeError(
                        f"'weak_links' argument in ADReversalSchedule "
                        f"constructor should be of type 'list[list[str]]' "
                        f"but found 'list[{type(link).__name__}]'."
                    )
                if len(link) != 2:
                    raise ValueError(
                        f"'weak_links' argument in ADReversalSchedule "
                        f"constructor should be of type "
                        f"'list[lists of 2 elements]' but found "
                        f"a list of length {len(link)}."
                    )
                for name in link:
                    if not isinstance(name, str):
                        raise TypeError(
                            f"'weak_links' argument in ADReversalSchedule "
                            f"constructor should be of type 'list[list[str]]' "
                            f"but found 'list[list[{type(name).__name__}]]'."
                        )
        if not isinstance(default_link, (str, type(None))):
            raise TypeError(
                f"'default_link' argument in ADReversalSchedule constructor "
                f"should be of type 'str' or 'NoneType' but found "
                f"'{type(default_link).__name__}'."
            )
        if default_link is not None:
            if default_link not in ("strong", "weak"):
                raise ValueError(
                    f"'weak_links' argument in ADReversalSchedule constructor "
                    f"should be 'strong', 'weak' or None but found "
                    f"{default_link}."
                )
        if (
            (strong_links is None)
            and (weak_links is None)
            and (default_link is None)
        ):
            raise ValueError(
                "One at least of 'strong_links, 'weak_links' and 'default_link' "
                "arguments in ADReversalSchedule constructor "
                "must be different from 'None'."
            )
        if (strong_links is not None) and (weak_links is not None):
            for link in strong_links:
                if link in weak_links:
                    raise ValueError(
                        f"Both 'strong_links' and 'weak_links' arguments in "
                        f"ADReversalSchedule constructor "
                        f"contain link '{link}', this is not allowed."
                    )

        self.strong_links = strong_links if strong_links is not None else []
        self.weak_links = weak_links if weak_links is not None else []
        self.default_link = default_link

    def is_strong_link(self, parent_routine_name, child_routine_name):
        """Determines if the link between two routines \
        is strong or not.
        Strong links lead to split reversals.
        Weak links lead to joint reversals.

        :param parent_routine_name: name of the parent (calling) routine.
        :type parent_routine_name: Str
        :param child_routine_name: name of the child (called) routine.
        :type child_routine_name: Str

        :raises TypeError: if parent_routine_name is of the wrong type.
        :raises TypeError: if child_routine_name is of the wrong type.

        :return: True, if the link is strong. False otherwise.
        :rtype: Bool
        """
        if not isinstance(parent_routine_name, str):
            raise TypeError(
                f"'parent_routine_name' argument in ADReversalSchedule "
                f"is_strong_link method "
                f"should be of type 'str' but found "
                f"'{type(parent_routine_name).__name__}'."
            )
        if not isinstance(child_routine_name, str):
            raise TypeError(
                f"'child_routine_name' argument in ADReversalSchedule "
                f"is_strong_link method "
                f"should be of type 'str' but found "
                f"'{type(child_routine_name).__name__}'."
            )

        link = [parent_routine_name, child_routine_name]
        if link in self.strong_links:
            return True

        if link in self.weak_links:
            return False

        if self.default_link == "strong":
            return True

        if self.default_link == "weak":
            return False

        raise ValueError(
            "No link type could be determined for arguments "
            f"'parent_routine_name' = '{parent_routine_name}' "
            f"and 'child_routine_name' = '{child_routine_name}' "
            f"in ADLinkReversalSchedule is_strong_link method."
        )
