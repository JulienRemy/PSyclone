# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2019-2023, Science and Technology Facilities Council.
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
# Authors R. W. Ford, A. R. Porter and S. Siso, STFC Daresbury Lab
#         I. Kavcic, Met Office
#         J. Henrichs, Bureau of Meteorology
# -----------------------------------------------------------------------------

'''Performs pytest tests on the Operation PSyIR node and its
sub-classes.

'''
import pytest

from psyclone.core import VariablesAccessInfo
from psyclone.psyir.nodes import UnaryOperation, BinaryOperation, \
    NaryOperation, Literal, Reference, Return
from psyclone.psyir.symbols import DataSymbol, INTEGER_SINGLE_TYPE, \
    REAL_SINGLE_TYPE
from psyclone.errors import GenerationError
from psyclone.psyir.backend.fortran import FortranWriter
from psyclone.tests.utilities import check_links
from psyclone.psyir.nodes import Assignment, colored


# Test Operation class. These are mostly covered by the subclass tests.

def test_operation_named_arg_str():
    '''Check the output of str from the Operation class when there is a
    mixture of positional and named arguments. We use a
    BinaryOperation example to exercise this method.

    '''
    lhs = Reference(DataSymbol("tmp1", REAL_SINGLE_TYPE))
    rhs = Reference(DataSymbol("tmp2", REAL_SINGLE_TYPE))
    oper = BinaryOperation.Operator.DOT_PRODUCT
    binaryoperation = BinaryOperation.create(oper, lhs, ("named_arg", rhs))
    assert "named_arg=Reference[name:'tmp2']" in str(binaryoperation)


def test_operation_appendnamedarg():
    '''Test the append_named_arg method in the Operation class. Check
    it raises the expected exceptions if arguments are invalid and
    that it works as expected when the input is valid. We use the
    NaryOperation node to perform the tests.

    '''
    nary_operation = NaryOperation(NaryOperation.Operator.MAX)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("2", INTEGER_SINGLE_TYPE)
    op3 = Literal("3", INTEGER_SINGLE_TYPE)
    # name arg wrong type
    with pytest.raises(TypeError) as info:
        nary_operation.append_named_arg(1, op1)
    assert ("A name should be a string or None, but found int."
            in str(info.value))
    # name arg invalid
    with pytest.raises(ValueError) as info:
        nary_operation.append_named_arg("_", op2)
    assert "Invalid name '_' found." in str(info.value)
    # name arg already used
    nary_operation.append_named_arg("name1", op1)
    with pytest.raises(ValueError) as info:
        nary_operation.append_named_arg("name1", op2)
    assert ("The value of the name argument (name1) in 'append_named_arg' in "
            "the 'Operator' node is already used for a named argument."
            in str(info.value))
    # ok
    nary_operation.append_named_arg("name2", op2)
    nary_operation.append_named_arg(None, op3)
    assert nary_operation.children == [op1, op2, op3]
    assert nary_operation.argument_names == ["name1", "name2", None]
    # too many args
    binary_operation = BinaryOperation.create(
        BinaryOperation.Operator.DOT_PRODUCT, op1.copy(), op2.copy())
    with pytest.raises(GenerationError) as info:
        binary_operation.append_named_arg(None, op3.copy())
    assert ("Item 'Literal' can't be child 2 of 'BinaryOperation'. The valid "
            "format is: 'DataNode, DataNode'." in str(info.value))


def test_operation_insertnamedarg():
    '''Test the insert_named_arg method in the Operation class. Check
    it raises the expected exceptions if arguments are invalid and
    that it works as expected when the input is valid. We use the
    NaryOperation node to perform the tests.

    '''
    nary_operation = NaryOperation(NaryOperation.Operator.MAX)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("2", INTEGER_SINGLE_TYPE)
    op3 = Literal("3", INTEGER_SINGLE_TYPE)
    # name arg wrong type
    with pytest.raises(TypeError) as info:
        nary_operation.insert_named_arg(1, op1, 0)
    assert ("A name should be a string or None, but found int."
            in str(info.value))
    # name arg invalid
    with pytest.raises(ValueError) as info:
        nary_operation.append_named_arg(" a", op2)
    assert "Invalid name ' a' found." in str(info.value)
    # name arg already used
    nary_operation.insert_named_arg("name1", op1, 0)
    with pytest.raises(ValueError) as info:
        nary_operation.insert_named_arg("name1", op2, 0)
    assert ("The value of the name argument (name1) in 'insert_named_arg' in "
            "the 'Operator' node is already used for a named argument."
            in str(info.value))
    # invalid index type
    with pytest.raises(TypeError) as info:
        nary_operation.insert_named_arg("name2", op2, "hello")
    assert ("The 'index' argument in 'insert_named_arg' in the 'Operator' "
            "node should be an int but found str." in str(info.value))
    # ok
    assert nary_operation.children == [op1]
    assert nary_operation.argument_names == ["name1"]
    nary_operation.insert_named_arg("name2", op2, 0)
    assert nary_operation.children == [op2, op1]
    assert nary_operation.argument_names == ["name2", "name1"]
    nary_operation.insert_named_arg(None, op3, 0)
    assert nary_operation.children == [op3, op2, op1]
    assert nary_operation.argument_names == [None, "name2", "name1"]
    # invalid index value
    binary_operation = BinaryOperation.create(
        BinaryOperation.Operator.DOT_PRODUCT, op1.copy(), op2.copy())
    with pytest.raises(GenerationError) as info:
        binary_operation.insert_named_arg("name2", op2.copy(), 2)
    assert ("Item 'Literal' can't be child 2 of 'BinaryOperation'. The valid "
            "format is: 'DataNode, DataNode'." in str(info.value))


def test_operation_replacenamedarg():
    '''Test the replace_named_arg method in the Operation class. Check
    it raises the expected exceptions if arguments are invalid and
    that it works as expected when the input is valid. We use the
    BinaryOperation node to perform the tests.

    '''
    binary_operation = BinaryOperation(BinaryOperation.Operator.DOT_PRODUCT)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("2", INTEGER_SINGLE_TYPE)
    op3 = Literal("3", INTEGER_SINGLE_TYPE)
    binary_operation.append_named_arg("name1", op1)
    binary_operation.append_named_arg("name2", op2)

    # name arg wrong type
    with pytest.raises(TypeError) as info:
        binary_operation.replace_named_arg(1, op3)
    assert ("The 'name' argument in 'replace_named_arg' in the 'Operation' "
            "node should be a string, but found int."
            in str(info.value))
    # name arg is not found
    with pytest.raises(ValueError) as info:
        binary_operation.replace_named_arg("new_name", op3)
    assert ("The value of the existing_name argument (new_name) in "
            "'replace_named_arg' in the 'Operation' node was not found in the "
            "existing arguments." in str(info.value))
    # ok
    assert binary_operation.children == [op1, op2]
    assert binary_operation.argument_names == ["name1", "name2"]
    assert binary_operation._argument_names[0][0] == id(op1)
    assert binary_operation._argument_names[1][0] == id(op2)
    binary_operation.replace_named_arg("name1", op3)
    assert binary_operation.children == [op3, op2]
    assert binary_operation.argument_names == ["name1", "name2"]
    assert binary_operation._argument_names[0][0] == id(op3)
    assert binary_operation._argument_names[1][0] == id(op2)


def test_operation_argumentnames_after_removearg():
    '''Test the argument_names property makes things consistent if a child
    argument is removed. This is used transparently by the class to
    keep things consistent. We use the BinaryOperation node to perform
    the tests.

    '''
    binary_operation = BinaryOperation(BinaryOperation.Operator.DOT_PRODUCT)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("1", INTEGER_SINGLE_TYPE)
    binary_operation.append_named_arg("name1", op1)
    binary_operation.append_named_arg("name2", op2)
    assert len(binary_operation.children) == 2
    assert len(binary_operation._argument_names) == 2
    assert binary_operation.argument_names == ["name1", "name2"]
    binary_operation.children.pop(0)
    assert len(binary_operation.children) == 1
    assert len(binary_operation._argument_names) == 2
    # argument_names property makes _argument_names list consistent.
    assert binary_operation.argument_names == ["name2"]
    assert len(binary_operation._argument_names) == 1


def test_operation_argumentnames_after_addarg():
    '''Test the argument_names property makes things consistent if a child
    argument is added. This is used transparently by the class to
    keep things consistent. We use the NaryOperation node to perform
    the tests (as it allows an arbitrary number of arguments.

    '''
    nary_operation = NaryOperation(NaryOperation.Operator.MAX)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("1", INTEGER_SINGLE_TYPE)
    op3 = Literal("1", INTEGER_SINGLE_TYPE)
    nary_operation.append_named_arg("name1", op1)
    nary_operation.append_named_arg("name2", op2)
    assert len(nary_operation.children) == 2
    assert len(nary_operation._argument_names) == 2
    assert nary_operation.argument_names == ["name1", "name2"]
    nary_operation.children.append(op3)
    assert len(nary_operation.children) == 3
    assert len(nary_operation._argument_names) == 2
    # argument_names property makes _argument_names list consistent.
    assert nary_operation.argument_names == ["name1", "name2", None]
    assert len(nary_operation._argument_names) == 3


def test_operation_argumentnames_after_replacearg():
    '''Test the argument_names property makes things consistent if a child
    argument is replaced with another. This is used transparently by
    the class to keep things consistent. We use the BinaryOperation
    node to perform the tests.

    '''
    binary_operation = BinaryOperation(BinaryOperation.Operator.DOT_PRODUCT)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("1", INTEGER_SINGLE_TYPE)
    op3 = Literal("1", INTEGER_SINGLE_TYPE)
    binary_operation.append_named_arg("name1", op1)
    binary_operation.append_named_arg("name2", op2)
    assert len(binary_operation.children) == 2
    assert len(binary_operation._argument_names) == 2
    assert binary_operation.argument_names == ["name1", "name2"]
    binary_operation.children[0] = op3
    assert len(binary_operation.children) == 2
    assert len(binary_operation._argument_names) == 2
    # argument_names property makes _argument_names list consistent.
    assert binary_operation.argument_names == [None, "name2"]
    assert len(binary_operation._argument_names) == 2


def test_operation_argumentnames_after_reorderearg():
    '''Test the argument_names property makes things consistent if child
    arguments are re-order. This is used transparently by the class to
    keep things consistent. We use the BinaryOperation node to perform
    the tests.

    '''
    binary_operation = BinaryOperation(BinaryOperation.Operator.DOT_PRODUCT)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("1", INTEGER_SINGLE_TYPE)
    binary_operation.append_named_arg("name1", op1)
    binary_operation.append_named_arg("name2", op2)
    assert len(binary_operation.children) == 2
    assert len(binary_operation._argument_names) == 2
    assert binary_operation.argument_names == ["name1", "name2"]
    tmp0 = binary_operation.children[0]
    tmp1 = binary_operation.children[1]
    tmp0.detach()
    tmp1.detach()
    binary_operation.children.extend([tmp1, tmp0])
    assert len(binary_operation.children) == 2
    assert len(binary_operation._argument_names) == 2
    # argument_names property makes _argument_names list consistent.
    assert binary_operation.argument_names == ["name2", "name1"]
    assert len(binary_operation._argument_names) == 2


def test_operation_reconcile_add():
    '''Test that the reconcile method behaves as expected. Use an
    NaryOperation example where we add a new arg.

    '''
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("1", INTEGER_SINGLE_TYPE)
    op3 = Literal("1", INTEGER_SINGLE_TYPE)
    oper = NaryOperation.create(
        NaryOperation.Operator.MAX, [("name1", op1), ("name2", op2)])
    # consistent
    assert len(oper._argument_names) == 2
    assert oper._argument_names[0] == (id(oper.children[0]), "name1")
    assert oper._argument_names[1] == (id(oper.children[1]), "name2")
    oper.children.append(op3)
    # inconsistent
    assert len(oper._argument_names) == 2
    assert oper._argument_names[0] == (id(oper.children[0]), "name1")
    assert oper._argument_names[1] == (id(oper.children[1]), "name2")
    oper._reconcile()
    # consistent
    assert len(oper._argument_names) == 3
    assert oper._argument_names[0] == (id(oper.children[0]), "name1")
    assert oper._argument_names[1] == (id(oper.children[1]), "name2")
    assert oper._argument_names[2] == (id(oper.children[2]), None)


def test_operation_reconcile_reorder():
    '''Test that the reconcile method behaves as expected. Use a
    BinaryOperation example where we reorder the arguments.

    '''
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("2", INTEGER_SINGLE_TYPE)
    oper = BinaryOperation.create(
        BinaryOperation.Operator.DOT_PRODUCT, ("name1", op1), ("name2", op2))
    # consistent
    assert len(oper._argument_names) == 2
    assert oper._argument_names[0] == (id(oper.children[0]), "name1")
    assert oper._argument_names[1] == (id(oper.children[1]), "name2")
    oper.children = [op2.detach(), op1.detach()]
    # inconsistent
    assert len(oper._argument_names) == 2
    assert oper._argument_names[0] != (id(oper.children[0]), "name1")
    assert oper._argument_names[1] != (id(oper.children[1]), "name2")
    oper._reconcile()
    # consistent
    assert len(oper._argument_names) == 2
    assert oper._argument_names[0] == (id(oper.children[0]), "name2")
    assert oper._argument_names[1] == (id(oper.children[1]), "name1")


# Test BinaryOperation class
def test_binaryoperation_initialization():
    ''' Check the initialization method of the BinaryOperation class works
    as expected.'''

    with pytest.raises(TypeError) as err:
        _ = BinaryOperation("not an operator")
    assert "BinaryOperation operator argument must be of type " \
           "BinaryOperation.Operator but found" in str(err.value)
    bop = BinaryOperation(BinaryOperation.Operator.ADD)
    assert bop._operator is BinaryOperation.Operator.ADD


def test_binaryoperation_operator():
    '''Test that the operator property returns the binaryoperator in the
    binaryoperation.

    '''
    binary_operation = BinaryOperation(BinaryOperation.Operator.ADD)
    assert binary_operation.operator == BinaryOperation.Operator.ADD


def test_binaryoperation_node_str():
    ''' Check the node_str method of the Binary Operation class.'''
    binary_operation = BinaryOperation(BinaryOperation.Operator.ADD)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("1", INTEGER_SINGLE_TYPE)
    binary_operation.addchild(op1)
    binary_operation.addchild(op2)
    coloredtext = colored("BinaryOperation", BinaryOperation._colour)
    assert coloredtext+"[operator:'ADD']" in binary_operation.node_str()


def test_binaryoperation_can_be_printed():
    '''Test that a Binary Operation instance can always be printed (i.e. is
    initialised fully)'''
    binary_operation = BinaryOperation(BinaryOperation.Operator.ADD)
    assert "BinaryOperation[operator:'ADD']" in str(binary_operation)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("2", INTEGER_SINGLE_TYPE)
    binary_operation.addchild(op1)
    binary_operation.addchild(op2)
    # Check the node children are also printed
    assert ("Literal[value:'1', Scalar<INTEGER, SINGLE>]\n"
            in str(binary_operation))
    assert ("Literal[value:'2', Scalar<INTEGER, SINGLE>]"
            in str(binary_operation))


def test_binaryoperation_create():
    '''Test that the create method in the BinaryOperation class correctly
    creates a BinaryOperation instance.

    '''
    lhs = Reference(DataSymbol("tmp1", REAL_SINGLE_TYPE))
    rhs = Reference(DataSymbol("tmp2", REAL_SINGLE_TYPE))
    oper = BinaryOperation.Operator.ADD
    binaryoperation = BinaryOperation.create(oper, lhs, rhs)
    check_links(binaryoperation, [lhs, rhs])
    result = FortranWriter().binaryoperation_node(binaryoperation)
    assert result == "tmp1 + tmp2"


def test_binaryoperation_named_create():
    '''Test that the create method in the BinaryOperation class correctly
    creates a BinaryOperation instance when one or more of the
    arguments is a named argument.

    '''
    lhs = Reference(DataSymbol("tmp1", REAL_SINGLE_TYPE))
    rhs = Reference(DataSymbol("tmp2", REAL_SINGLE_TYPE))
    oper = BinaryOperation.Operator.DOT_PRODUCT
    binaryoperation = BinaryOperation.create(oper, lhs, ("dim", rhs))
    check_links(binaryoperation, [lhs, rhs])
    result = FortranWriter().binaryoperation_node(binaryoperation)
    assert result == "DOT_PRODUCT(tmp1, dim=tmp2)"
    binaryoperation = BinaryOperation.create(
        oper, ("dummy", lhs.detach()), ("dim", rhs.detach()))
    check_links(binaryoperation, [lhs, rhs])
    result = FortranWriter().binaryoperation_node(binaryoperation)
    assert result == "DOT_PRODUCT(dummy=tmp1, dim=tmp2)"


def test_binaryoperation_create_invalid():
    '''Test that the create method in a BinaryOperation class raises the
    expected exception if the provided input is invalid.

    '''
    ref1 = Reference(DataSymbol("tmp1", REAL_SINGLE_TYPE))
    ref2 = Reference(DataSymbol("tmp2", REAL_SINGLE_TYPE))
    add = BinaryOperation.Operator.ADD

    # oper not a BinaryOperation.Operator.
    with pytest.raises(GenerationError) as excinfo:
        _ = BinaryOperation.create("invalid", ref1, ref2)
    assert ("operator argument in create method of BinaryOperation class "
            "should be a PSyIR BinaryOperation Operator but found 'str'."
            in str(excinfo.value))

    # lhs not a Node.
    with pytest.raises(GenerationError) as excinfo:
        _ = BinaryOperation.create(add, "invalid", ref2)
    assert ("Item 'str' can't be child 0 of 'BinaryOperation'. The valid "
            "format is: 'DataNode, DataNode'.") in str(excinfo.value)

    # rhs not a Node.
    with pytest.raises(GenerationError) as excinfo:
        _ = BinaryOperation.create(add, ref1, "invalid")
    assert ("Item 'str' can't be child 1 of 'BinaryOperation'. The valid "
            "format is: 'DataNode, DataNode'.") in str(excinfo.value)

    # rhs is an invalid tuple (too many elements)
    oper = BinaryOperation.Operator.DOT_PRODUCT
    with pytest.raises(GenerationError) as excinfo:
        _ = BinaryOperation.create(oper, ref1, (1, 2, 3))
    assert ("If the rhs argument in create method of BinaryOperation class "
            "is a tuple, it's length should be 2, but it is 3."
            in str(excinfo.value))

    # rhs is an invalid tuple (1st element not str)
    oper = BinaryOperation.Operator.DOT_PRODUCT
    with pytest.raises(GenerationError) as excinfo:
        _ = BinaryOperation.create(oper, ref1, (1, 2))
    assert ("If the rhs argument in create method of BinaryOperation class "
            "is a tuple, its first argument should be a str, but found "
            "int." in str(excinfo.value))

    # rhs has an invalid name (1st element invalid value)
    oper = BinaryOperation.Operator.DOT_PRODUCT
    with pytest.raises(ValueError) as info:
        _ = BinaryOperation.create(oper, ref1.copy(), ("_", 2))
    assert "Invalid name '_' found." in str(info.value)


def test_binaryoperation_children_validation():
    '''Test that children added to BinaryOperation are validated.
    BinaryOperations accept 2 DataNodes as children.

    '''
    operation = BinaryOperation(BinaryOperation.Operator.ADD)
    literal1 = Literal("1", INTEGER_SINGLE_TYPE)
    literal2 = Literal("2", INTEGER_SINGLE_TYPE)
    literal3 = Literal("3", INTEGER_SINGLE_TYPE)
    statement = Return()

    # Statements are not valid
    with pytest.raises(GenerationError) as excinfo:
        operation.addchild(statement)
    assert ("Item 'Return' can't be child 0 of 'BinaryOperation'. The valid "
            "format is: 'DataNode, DataNode'.") in str(excinfo.value)

    # First DataNodes is valid, but not subsequent ones
    operation.addchild(literal1)
    operation.addchild(literal2)
    with pytest.raises(GenerationError) as excinfo:
        operation.addchild(literal3)
    assert ("Item 'Literal' can't be child 2 of 'BinaryOperation'. The valid "
            "format is: 'DataNode, DataNode'.") in str(excinfo.value)


def test_binaryoperation_is_elemental():
    '''Test that the is_elemental method properly returns if an operation is
    elemental in each BinaryOperation.

    '''
    # MATMUL, SIZE, LBOUND, UBOUND and DOT_PRODUCT are not
    # elemental
    not_elemental = [
        BinaryOperation.Operator.SIZE,
        BinaryOperation.Operator.MATMUL,
        BinaryOperation.Operator.LBOUND,
        BinaryOperation.Operator.UBOUND,
        BinaryOperation.Operator.DOT_PRODUCT
    ]

    for binary_operator in BinaryOperation.Operator:
        operation = BinaryOperation(binary_operator)
        if binary_operator in not_elemental:
            assert operation.is_elemental is False
        else:
            assert operation.is_elemental is True


# Test UnaryOperation class
def test_unaryoperation_initialization():
    ''' Check the initialization method of the UnaryOperation class works
    as expected.'''

    with pytest.raises(TypeError) as err:
        _ = UnaryOperation("not an operator")
    assert "UnaryOperation operator argument must be of type " \
           "UnaryOperation.Operator but found" in str(err.value)
    uop = UnaryOperation(UnaryOperation.Operator.MINUS)
    assert uop._operator is UnaryOperation.Operator.MINUS


@pytest.mark.parametrize("operator_name", ['MINUS', 'MINUS', 'PLUS', 'SQRT',
                                           'EXP', 'LOG', 'LOG10', 'NOT',
                                           'COS', 'SIN', 'TAN', 'ACOS',
                                           'ASIN', 'ATAN', 'ABS', 'CEIL',
                                           'FLOOR', 'REAL', 'INT', 'NINT'])
def test_unaryoperation_operator(operator_name):
    '''Test that the operator property returns the unaryoperator in the
    unaryoperation.

    '''
    operator = getattr(UnaryOperation.Operator, operator_name)
    unary_operation = UnaryOperation(operator)
    assert unary_operation.operator == operator


def test_unaryoperation_node_str():
    ''' Check the view method of the UnaryOperation class.'''
    ref1 = Reference(DataSymbol("a", REAL_SINGLE_TYPE))
    unary_operation = UnaryOperation.create(UnaryOperation.Operator.MINUS,
                                            ref1)
    coloredtext = colored("UnaryOperation", UnaryOperation._colour)
    assert coloredtext+"[operator:'MINUS']" in unary_operation.node_str()


def test_unaryoperation_can_be_printed():
    '''Test that a UnaryOperation instance can always be printed (i.e. is
    initialised fully)'''
    unary_operation = UnaryOperation(UnaryOperation.Operator.MINUS)
    assert "UnaryOperation[operator:'MINUS']" in str(unary_operation)
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    unary_operation.addchild(op1)
    # Check the node children are also printed
    assert ("Literal[value:'1', Scalar<INTEGER, SINGLE>]"
            in str(unary_operation))


def test_unaryoperation_create():
    '''Test that the create method in the UnaryOperation class correctly
    creates a UnaryOperation instance.

    '''
    child = Reference(DataSymbol("tmp", REAL_SINGLE_TYPE))
    oper = UnaryOperation.Operator.SIN
    unaryoperation = UnaryOperation.create(oper, child)
    check_links(unaryoperation, [child])
    result = FortranWriter().unaryoperation_node(unaryoperation)
    assert result == "SIN(tmp)"


def test_unaryoperation_named_create():
    '''Test that the create method in the UnaryOperation class correctly
    creates a UnaryOperation instance when there is a named argument.

    '''
    child = Reference(DataSymbol("tmp", REAL_SINGLE_TYPE))
    oper = UnaryOperation.Operator.SIN
    unaryoperation = UnaryOperation.create(oper, ("name", child))
    assert unaryoperation.argument_names == ["name"]
    check_links(unaryoperation, [child])
    result = FortranWriter().unaryoperation_node(unaryoperation)
    assert result == "SIN(name=tmp)"


def test_unaryoperation_create_invalid1():
    '''Test that the create method in a UnaryOperation class raises the
    expected exception if the provided argument is a tuple that does
    not have 2 elements.

    '''
    # oper not a UnaryOperator.Operator.
    oper = UnaryOperation.Operator.SIN
    with pytest.raises(GenerationError) as excinfo:
        _ = UnaryOperation.create(
            oper, (1, 2, 3))
    assert ("If the argument in the create method of UnaryOperation class is "
            "a tuple, it's length should be 2, but it is 3."
            in str(excinfo.value))


def test_unaryoperation_create_invalid2():
    '''Test that the create method in a UnaryOperation class raises the
    expected exception if the provided argument is a tuple and the
    first element of the tuple is not a string.

    '''
    # oper not a UnaryOperator.Operator.
    oper = UnaryOperation.Operator.SIN
    with pytest.raises(GenerationError) as excinfo:
        _ = UnaryOperation.create(
            oper, (1, 2))
    assert ("If the argument in the create method of UnaryOperation class "
            "is a tuple, its first argument should be a str, but found int."
            in str(excinfo.value))


def test_unaryoperation_create_invalid3():
    '''Test that the create method in a UnaryOperation class raises the
    expected exception a named argument is provided with an invalid name.

    '''
    oper = UnaryOperation.Operator.SIN
    with pytest.raises(ValueError) as info:
        _ = UnaryOperation.create(oper, ("1", 2))
    assert "Invalid name '1' found." in str(info.value)


def test_unaryoperation_create_invalid4():
    '''Test that the create method in a UnaryOperation class raises the
    expected exception if the provided argument is a tuple with the
    wrong number of arguments.

    '''
    # oper not a UnaryOperator.Operator.
    with pytest.raises(GenerationError) as excinfo:
        _ = UnaryOperation.create(
            "invalid",
            Reference(DataSymbol("tmp", REAL_SINGLE_TYPE)))
    assert ("operator argument in create method of UnaryOperation class "
            "should be a PSyIR UnaryOperation Operator but found 'str'."
            in str(excinfo.value))


def test_unaryoperation_children_validation():
    '''Test that children added to unaryOperation are validated.
    UnaryOperations accept just 1 DataNode as child.

    '''
    operation = UnaryOperation(UnaryOperation.Operator.SIN)
    literal1 = Literal("1", INTEGER_SINGLE_TYPE)
    literal2 = Literal("2", INTEGER_SINGLE_TYPE)
    statement = Return()

    # Statements are not valid
    with pytest.raises(GenerationError) as excinfo:
        operation.addchild(statement)
    assert ("Item 'Return' can't be child 0 of 'UnaryOperation'. The valid "
            "format is: 'DataNode'.") in str(excinfo.value)

    # First DataNodes is valid, but not subsequent ones
    operation.addchild(literal1)
    with pytest.raises(GenerationError) as excinfo:
        operation.addchild(literal2)
    assert ("Item 'Literal' can't be child 1 of 'UnaryOperation'. The valid "
            "format is: 'DataNode'.") in str(excinfo.value)


def test_unaryoperation_is_elemental():
    '''Test that the is_elemental method properly returns if an operation is
    elemental in each UnaryOperation.

    '''
    # All unary operators are elemental
    for unary_operator in UnaryOperation.Operator:
        operation = UnaryOperation(unary_operator)
        assert operation.is_elemental is True


# Test NaryOperation class
def test_naryoperation_node_str():
    ''' Check the node_str method of the Nary Operation class.'''
    nary_operation = NaryOperation(NaryOperation.Operator.MAX)
    nary_operation.addchild(Literal("1", INTEGER_SINGLE_TYPE))
    nary_operation.addchild(Literal("1", INTEGER_SINGLE_TYPE))
    nary_operation.addchild(Literal("1", INTEGER_SINGLE_TYPE))

    coloredtext = colored("NaryOperation", NaryOperation._colour)
    assert coloredtext+"[operator:'MAX']" in nary_operation.node_str()


def test_naryoperation_can_be_printed():
    '''Test that an Nary Operation instance can always be printed (i.e. is
    initialised fully)'''
    nary_operation = NaryOperation(NaryOperation.Operator.MAX)
    assert "NaryOperation[operator:'MAX']" in str(nary_operation)
    nary_operation.addchild(Literal("1", INTEGER_SINGLE_TYPE))
    nary_operation.addchild(Literal("2", INTEGER_SINGLE_TYPE))
    nary_operation.addchild(Literal("3", INTEGER_SINGLE_TYPE))
    # Check the node children are also printed
    assert ("Literal[value:'1', Scalar<INTEGER, SINGLE>]\n"
            in str(nary_operation))
    assert ("Literal[value:'2', Scalar<INTEGER, SINGLE>]\n"
            in str(nary_operation))
    assert ("Literal[value:'3', Scalar<INTEGER, SINGLE>]"
            in str(nary_operation))


def test_naryoperation_create():
    '''Test that the create method in the NaryOperation class correctly
    creates an NaryOperation instance.

    '''
    children = [Reference(DataSymbol("tmp1", REAL_SINGLE_TYPE)),
                Reference(DataSymbol("tmp2", REAL_SINGLE_TYPE)),
                Reference(DataSymbol("tmp3", REAL_SINGLE_TYPE))]
    oper = NaryOperation.Operator.MAX
    naryoperation = NaryOperation.create(oper, children)
    check_links(naryoperation, children)
    result = FortranWriter().naryoperation_node(naryoperation)
    assert result == "MAX(tmp1, tmp2, tmp3)"


def test_naryoperation_named_create():
    '''Test that the create method in the NaryOperation class correctly
    creates a NaryOperation instance when one of the arguments is a
    named argument.

    '''
    children = [Reference(DataSymbol("tmp1", REAL_SINGLE_TYPE)),
                Reference(DataSymbol("tmp2", REAL_SINGLE_TYPE)),
                Reference(DataSymbol("tmp3", REAL_SINGLE_TYPE))]
    oper = NaryOperation.Operator.MAX
    naryoperation = NaryOperation.create(
        oper, [children[0], children[1], ("name", children[2])])
    check_links(naryoperation, children)
    result = FortranWriter().naryoperation_node(naryoperation)
    assert result == "MAX(tmp1, tmp2, name=tmp3)"


def test_naryoperation_create_invalid():
    '''Test that the create method in an NaryOperation class raises the
    expected exception if the provided input is invalid.

    '''
    # oper not an NaryOperation.Operator
    with pytest.raises(GenerationError) as excinfo:
        _ = NaryOperation.create("invalid", [])
    assert ("operator argument in create method of NaryOperation class should "
            "be a PSyIR NaryOperation Operator but found 'str'."
            in str(excinfo.value))

    oper = NaryOperation.Operator.MAX

    # children not a list
    with pytest.raises(GenerationError) as excinfo:
        _ = NaryOperation.create(oper, "invalid")
    assert ("operands argument in create method of NaryOperation class should "
            "be a list but found 'str'." in str(excinfo.value))

    ref1 = Reference(DataSymbol("tmp1", REAL_SINGLE_TYPE))
    ref2 = Reference(DataSymbol("tmp2", REAL_SINGLE_TYPE))

    # rhs is an invalid tuple (too many elements)
    with pytest.raises(GenerationError) as excinfo:
        _ = NaryOperation.create(oper, [ref1, ref2, (1, 2, 3)])
    assert ("If an element of the operands argument in create method of "
            "NaryOperation class is a tuple, it's length should be 2, "
            "but found 3." in str(excinfo.value))

    # rhs is an invalid tuple (1st element not str)
    with pytest.raises(GenerationError) as excinfo:
        _ = NaryOperation.create(oper, [ref1.copy(), ref2.copy(), (1, 2)])
    assert ("If an element of the operands argument in create method of "
            "NaryOperation class is a tuple, its first argument should "
            "be a str, but found int." in str(excinfo.value))


def test_naryoperation_children_validation():
    '''Test that children added to NaryOperation are validated. NaryOperations
    accepts DataNodes nodes as children.

    '''
    nary = NaryOperation(NaryOperation.Operator.MAX)
    literal1 = Literal("1", INTEGER_SINGLE_TYPE)
    literal2 = Literal("2", INTEGER_SINGLE_TYPE)
    literal3 = Literal("3", INTEGER_SINGLE_TYPE)
    statement = Return()

    # DataNodes are valid
    nary.addchild(literal1)
    nary.addchild(literal2)
    nary.addchild(literal3)

    # Statements are not valid
    with pytest.raises(GenerationError) as excinfo:
        nary.addchild(statement)
    assert ("Item 'Return' can't be child 3 of 'NaryOperation'. The valid "
            "format is: '[DataNode]+'.") in str(excinfo.value)


def test_naryoperation_is_elemental():
    '''Test that the is_elemental method properly returns if an operation is
    elemental in each NaryOperation.

    '''
    # All nary operations are elemental
    for nary_operator in NaryOperation.Operator:
        operation = NaryOperation(nary_operator)
        assert operation.is_elemental is True


def test_operations_can_be_copied():
    ''' Test that an operation can be copied. '''

    operands = [Reference(DataSymbol("tmp1", REAL_SINGLE_TYPE)),
                Reference(DataSymbol("tmp2", REAL_SINGLE_TYPE)),
                Reference(DataSymbol("tmp3", REAL_SINGLE_TYPE))]
    operation = NaryOperation.create(NaryOperation.Operator.MAX, operands)

    operation1 = operation.copy()
    assert isinstance(operation1, NaryOperation)
    assert operation1 is not operation
    assert operation1.operator is NaryOperation.Operator.MAX
    assert operation1.children[0].symbol.name == "tmp1"
    assert operation1.children[0] is not operands[0]
    assert operation1.children[0].parent is operation1
    assert operation1.children[1].symbol.name == "tmp2"
    assert operation1.children[1] is not operands[1]
    assert operation1.children[1].parent is operation1
    assert operation1.children[2].symbol.name == "tmp3"
    assert operation1.children[2] is not operands[2]
    assert operation1.children[2].parent is operation1
    assert len(operation1.children) == 3
    assert len(operation.children) == 3

    # Modifying the new operation does not affect the original
    operation1._operator = NaryOperation.Operator.MIN
    operation1.children.pop()
    assert len(operation1.children) == 2
    assert len(operation.children) == 3
    assert operation1.operator is NaryOperation.Operator.MIN
    assert operation.operator is NaryOperation.Operator.MAX


def test_copy():
    '''Test that the copy() method behaves as expected when there are
    named arguments.

    '''
    op1 = Literal("1", INTEGER_SINGLE_TYPE)
    op2 = Literal("2", INTEGER_SINGLE_TYPE)
    oper = BinaryOperation.create(
        BinaryOperation.Operator.DOT_PRODUCT, ("name1", op1), ("name2", op2))
    # consistent operation
    oper_copy = oper.copy()
    assert oper._argument_names[0] == (id(oper.children[0]), "name1")
    assert oper._argument_names[1] == (id(oper.children[1]), "name2")
    assert oper_copy._argument_names[0] == (id(oper_copy.children[0]), "name1")
    assert oper_copy._argument_names[1] == (id(oper_copy.children[1]), "name2")
    assert oper._argument_names != oper_copy._argument_names

    oper.children = [op2.detach(), op1.detach()]
    assert oper._argument_names[0] != (id(oper.children[0]), "name2")
    assert oper._argument_names[1] != (id(oper.children[1]), "name1")
    # inconsistent operation
    oper_copy = oper.copy()
    assert oper._argument_names[0] == (id(oper.children[0]), "name2")
    assert oper._argument_names[1] == (id(oper.children[1]), "name1")
    assert oper_copy._argument_names[0] == (id(oper_copy.children[0]), "name2")
    assert oper_copy._argument_names[1] == (id(oper_copy.children[1]), "name1")
    assert oper._argument_names != oper_copy._argument_names


def test_operation_equality():
    ''' Test the __eq__ method of Operation'''
    tmp1 = DataSymbol("tmp1", REAL_SINGLE_TYPE)
    tmp2 = DataSymbol("tmp2", REAL_SINGLE_TYPE)
    lhs = Reference(tmp1)
    rhs = Reference(tmp2)
    oper = BinaryOperation.Operator.ADD
    binaryoperation1 = BinaryOperation.create(oper, lhs, rhs)

    oper = BinaryOperation.Operator.ADD
    binaryoperation2 = BinaryOperation.create(oper, lhs.copy(), rhs.copy())

    assert binaryoperation1 == binaryoperation2

    # change the operator
    binaryoperation2._operator = BinaryOperation.Operator.SUB
    assert binaryoperation1 != binaryoperation2

    # Check with arguments names
    binaryoperation3 = BinaryOperation.create(
        oper, ("name1", lhs.copy()), rhs.copy())
    binaryoperation4 = BinaryOperation.create(
        oper, ("name1", lhs.copy()), rhs.copy())
    assert binaryoperation3 == binaryoperation4

    # Check with argument name and no argument name
    assert binaryoperation3 != binaryoperation1

    # Check with different argument names
    binaryoperation5 = BinaryOperation.create(
        oper, ("new_name", lhs.copy()), rhs.copy())
    assert binaryoperation3 != binaryoperation5


@pytest.mark.parametrize("operator", ["lbound", "ubound", "size"])
def test_reference_accesses_bounds(operator, fortran_reader):
    '''Test that the reference_accesses method behaves as expected when
    the reference is the first argument to either the lbound or ubound
    intrinsic as that is simply looking up the array bounds (therefore
    var_access_info should be empty) and when the reference is the
    second argument of either the lbound or ubound intrinsic (in which
    case the access should be a read).

    '''
    code = f'''module test
        contains
        subroutine tmp()
          real, dimension(:,:), allocatable:: a, b
          integer :: n
          n = {operator}(a, b(1,1))
        end subroutine tmp
        end module test'''
    psyir = fortran_reader.psyir_from_source(code)
    schedule = psyir.walk(Assignment)[0]

    # By default, the access to 'a' should not be reported as read,
    # but the access to b must be reported:
    vai = VariablesAccessInfo(schedule)
    assert str(vai) == "b: READ, n: WRITE"

    # When explicitly requested, the access to 'a' should be reported:
    vai = VariablesAccessInfo(schedule,
                              options={"COLLECT-ARRAY-SHAPE-READS": True})
    assert str(vai) == "a: READ, b: READ, n: WRITE"


from itertools import product
from subprocess import run
from psyclone.psyir.nodes import Routine, Reference, FileContainer, Call
from psyclone.psyir.symbols import DataSymbol, ScalarType, ArrayType
from psyclone.psyir.symbols.interfaces import ArgumentInterface, StaticInterface

def _initialize_scalar_kinds():
    intrinsics = ['INTEGER', 'BOOLEAN', 'REAL']
    precisions = ['SINGLE', 'DOUBLE', 'UNDEFINED', 4, 8]

    integer_kinds = []
    boolean_kinds = []
    real_kinds = []
    for t in intrinsics:
        for k in precisions:
            if isinstance(k, str):
                scalar_type = ScalarType(ScalarType.Intrinsic[t], ScalarType.Precision[k])
            else:
                scalar_type = ScalarType(ScalarType.Intrinsic[t], k)

            if t == 'INTEGER':
                integer_kinds.append(scalar_type)
            elif t == 'REAL':
                real_kinds.append(scalar_type)
            else:
                boolean_kinds.append(scalar_type)

    return integer_kinds, real_kinds, boolean_kinds

def _initialize_type_checkers(container, array_shape, arrays_only=False):
    integer_kinds, real_kinds, boolean_kinds = _initialize_scalar_kinds()
    scalar_datatypes = integer_kinds + real_kinds + boolean_kinds

    array_datatypes = [ArrayType(scalar_type, array_shape) for scalar_type in scalar_datatypes]

    routine_symbols = {}

    if arrays_only:
        datatypes = array_datatypes
    else:
        datatypes = scalar_datatypes + array_datatypes

    for datatype in datatypes:
        arg = DataSymbol("arg", datatype)
        arg.interface = ArgumentInterface(ArgumentInterface.Access.READ)

        if isinstance(datatype.precision, int):
            name = f"{datatype.intrinsic.name}_{datatype.precision}"
        else:
            name = f"{datatype.intrinsic.name}_{datatype.precision.name}"

        if datatype in array_datatypes:
            dim_str = f"{array_shape[0]}"
            if len(array_shape) == 2:
                dim_str += f"x{array_shape[1]}"
            name += "_" + dim_str

        routine = Routine(name)
        routine.symbol_table._argument_list.append(arg)
        routine.symbol_table.add(arg)
    
        container.addchild(routine)
        
        if datatype in scalar_datatypes:
            key = "scalar"
        else:
            key = dim_str
        routine_symbols[(datatype.intrinsic, datatype.precision, key)] = routine.symbol_table.lookup_with_tag("own_routine_symbol")

    return container, routine_symbols

def _initialize_datasymbol(datatype):
    if datatype.intrinsic is ScalarType.Intrinsic.BOOLEAN:
        initial_value = True
    elif datatype.intrinsic is ScalarType.Intrinsic.INTEGER:
        initial_value = 1
    else:
        initial_value = 1.1

    if isinstance(datatype.precision, int):
        name = f"ref_{datatype.intrinsic.name}_{datatype.precision}"
    else:
        name = f"ref_{datatype.intrinsic.name}_{datatype.precision.name}"
    
    if isinstance(datatype, ArrayType):
        name += f"_{datatype.shape[0].upper.value}"
        if len(datatype.shape) == 2:
            name += f"x{datatype.shape[1].upper.value}"

    return DataSymbol(name, datatype, initial_value=initial_value, interface=StaticInterface())

    
def _write_and_compile(fortran_writer, container, compiler_cmd = "gfortran"):
    with open(f"{container.name}.f90", 'w') as file:
        fortran = fortran_writer(container)
        #print(fortran)
        file.write(fortran)
        
    run([compiler_cmd, f"{container.name}.f90", "-o", container.name], check=True)
    run(f"./{container.name}", check=True)

def test_unaryoperation_datatypes(fortran_writer, compiler_cmd = "gfortran"):
    container = FileContainer("unaryoperation")

    array_shape = [2,3]

    container, routine_symbols = _initialize_type_checkers(container, array_shape)

    integer_kinds, real_kinds, boolean_kinds = _initialize_scalar_kinds()
    scalar_datatypes = integer_kinds + real_kinds + boolean_kinds

    array_datatypes = [ArrayType(scalar_type, array_shape) for scalar_type in scalar_datatypes]

    program = Routine("unaryoperation", is_program=True)
    container.addchild(program)

    for datatype in scalar_datatypes + array_datatypes:
        sym = _initialize_datasymbol(datatype)
        
        program.symbol_table.add(sym)

        if datatype in scalar_datatypes:
            key = "scalar"
        else:
            key = f"{array_shape[0]}x{array_shape[1]}"

        # CEIL, FLOOT, NINT : REAL => default INTEGER
        if datatype.intrinsic is ScalarType.Intrinsic.REAL:
            for op in (UnaryOperation.Operator.CEIL, 
                       UnaryOperation.Operator.FLOOR, 
                       UnaryOperation.Operator.NINT):

                operation = UnaryOperation.create(op, Reference(sym))

                assert (operation.datatype.intrinsic is ScalarType.Intrinsic.INTEGER)
                assert (operation.datatype.precision is ScalarType.Precision.UNDEFINED)
                assert (type(operation.datatype) is type(datatype))
                if datatype in array_datatypes:
                    assert (operation.datatype.shape == datatype.shape)

                check = Call.create(routine_symbols[(operation.datatype.intrinsic, 
                                                     operation.datatype.precision,
                                                     key)], 
                                    [operation])
                
                program.addchild(check)
        
        # INT :
        # - INTEGER, REAL => default INTEGER
        if datatype.intrinsic in (ScalarType.Intrinsic.INTEGER, ScalarType.Intrinsic.REAL):
            operation = UnaryOperation.create(UnaryOperation.Operator.INT, 
                                              Reference(sym))
            
            assert (operation.datatype.intrinsic is ScalarType.Intrinsic.INTEGER)
            assert (operation.datatype.precision is ScalarType.Precision.UNDEFINED)
            assert (type(operation.datatype) is type(datatype))
            if datatype in array_datatypes:
                    assert (operation.datatype.shape == datatype.shape)

            check = Call.create(routine_symbols[(operation.datatype.intrinsic, 
                                                 operation.datatype.precision,
                                                 key)], 
                                [operation])
            program.addchild(check)

        # MINUS, PLUS, ABS:
        # - INTEGER*k => INTEGER*k
        # - REAL*k => REAL*k
        if datatype.intrinsic in (ScalarType.Intrinsic.INTEGER, ScalarType.Intrinsic.REAL):
            for op in (UnaryOperation.Operator.MINUS, 
                       UnaryOperation.Operator.PLUS,
                       UnaryOperation.Operator.ABS):

                operation = UnaryOperation.create(op, Reference(sym))

                assert (operation.datatype.intrinsic is datatype.intrinsic)
                assert (operation.datatype.precision is datatype.precision)
                assert (type(operation.datatype) is type(datatype))
                if datatype in array_datatypes:
                    assert (operation.datatype.shape == datatype.shape)

                check = Call.create(routine_symbols[(operation.datatype.intrinsic, 
                                                     operation.datatype.precision,
                                                     key)], 
                                    [operation])
                
                program.addchild(check)

        # SQRT, EXP, LOG, LOG10, COS, SIN, TAN, ACOS, ASIN, ATAN : REAL*k => REAL*k
        if datatype.intrinsic is ScalarType.Intrinsic.REAL:
            for op in (UnaryOperation.Operator.SQRT, 
                       UnaryOperation.Operator.EXP,
                       UnaryOperation.Operator.LOG,
                       UnaryOperation.Operator.LOG10,
                       UnaryOperation.Operator.COS,
                       UnaryOperation.Operator.SIN,
                       UnaryOperation.Operator.TAN,
                       UnaryOperation.Operator.ACOS,
                       UnaryOperation.Operator.ASIN,
                       UnaryOperation.Operator.ATAN):

                operation = UnaryOperation.create(op, Reference(sym))

                assert (operation.datatype.intrinsic is datatype.intrinsic)
                assert (operation.datatype.precision is datatype.precision)
                assert (type(operation.datatype) is type(datatype))
                if datatype in array_datatypes:
                    assert (operation.datatype.shape == datatype.shape)

                check = Call.create(routine_symbols[(operation.datatype.intrinsic, 
                                                     operation.datatype.precision,
                                                     key)], 
                                    [operation])
                
                program.addchild(check)

        # NOT : LOGICAL*K => LOGICAL*K
        if datatype.intrinsic is ScalarType.Intrinsic.BOOLEAN:
            operation = UnaryOperation.create(UnaryOperation.Operator.NOT, 
                                              Reference(sym))

            assert (operation.datatype.intrinsic is datatype.intrinsic)
            assert (operation.datatype.precision is datatype.precision)
            assert (type(operation.datatype) is type(datatype))
            if datatype in array_datatypes:
                assert (operation.datatype.shape == datatype.shape)

            check = Call.create(routine_symbols[(operation.datatype.intrinsic, 
                                                 operation.datatype.precision,
                                                 key)], 
                                [operation])
            
            program.addchild(check)

        # TRANSPOSE : type_kind_ixj => type_kind_jxi
        if datatype in array_datatypes:
            transpose_datatype = ArrayType(ScalarType(datatype.intrinsic, datatype.precision), datatype.shape[::-1])
            transpose_sym = _initialize_datasymbol(transpose_datatype)

            new_sym = DataSymbol(transpose_sym.name + "_T", INTEGER_SINGLE_TYPE)
            new_sym.copy_properties(transpose_sym)

            program.symbol_table.add(new_sym)

            operation = UnaryOperation.create(UnaryOperation.Operator.TRANSPOSE, 
                                              Reference(new_sym))
            
            assert (operation.datatype.intrinsic is datatype.intrinsic)
            assert (operation.datatype.precision is datatype.precision)
            assert (type(operation.datatype) is type(datatype))
            assert (operation.datatype.shape == datatype.shape)

            check = Call.create(routine_symbols[(operation.datatype.intrinsic, 
                                                 operation.datatype.precision,
                                                 key)], 
                                [operation])
            
            program.addchild(check)

    _write_and_compile(fortran_writer, container, compiler_cmd)

def test_binaryoperation_datatypes(fortran_writer, compiler_cmd = "gfortran"):
    container = FileContainer("binaryoperation")

    array_shape = [2,3]
    vector_shape = [3]

    container, routine_symbols = _initialize_type_checkers(container, array_shape)
    # For 2x3 @ 3x2
    container, other_routine_symbols = _initialize_type_checkers(container, [array_shape[0], array_shape[0]], arrays_only=True)
    routine_symbols = {**routine_symbols, **other_routine_symbols}
    # For 2x3 @ 3
    container, other_routine_symbols = _initialize_type_checkers(container, [array_shape[0]], arrays_only=True)
    routine_symbols = {**routine_symbols, **other_routine_symbols}

    integer_kinds, real_kinds, boolean_kinds = _initialize_scalar_kinds()
    scalar_datatypes = integer_kinds + real_kinds + boolean_kinds

    array_datatypes1 = [ArrayType(scalar_type, array_shape) for scalar_type in scalar_datatypes]
    vector_datatypes1 = [ArrayType(scalar_type, vector_shape) for scalar_type in scalar_datatypes]
    # Transposed matrices
    array_datatypes2 = [ArrayType(scalar_type, array_shape[::-1]) for scalar_type in scalar_datatypes]
    vector_datatypes2 = [ArrayType(scalar_type, vector_shape) for scalar_type in scalar_datatypes]

    datatypes1 = scalar_datatypes + array_datatypes1 + vector_datatypes1
    datatypes2 = scalar_datatypes + array_datatypes2 + vector_datatypes2

    program = Routine("binaryoperation", is_program=True)
    container.addchild(program)

    for datatype1, datatype2 in product(datatypes1, datatypes2):
        sym1 = _initialize_datasymbol(datatype1)
        sym2 = _initialize_datasymbol(datatype2)
        
        for sym in (sym1, sym2):
            if sym.name not in program.symbol_table:
                program.symbol_table.add(sym)

        if datatype1 in scalar_datatypes:
            key1 = "scalar"
        elif datatype1 in array_datatypes1:
            key1 = f"{array_shape[0]}x{array_shape[1]}"
        else:
            key1 = f"{vector_shape[0]}"

        if datatype2 in scalar_datatypes:
            key2 = "scalar"
        elif datatype2 in array_datatypes2:
            key2 = f"{array_shape[1]}x{array_shape[0]}"
        else:
            key2 = f"{vector_shape[0]}"

        # SIZE, LBOUND, UBOUND : (array, INTEGER) => default INTEGER
        if datatype1 in array_datatypes1 \
            and (datatype2 in scalar_datatypes \
                 and datatype2.intrinsic is ScalarType.Intrinsic.INTEGER):
            for op in (BinaryOperation.Operator.SIZE, 
                       BinaryOperation.Operator.LBOUND, 
                       BinaryOperation.Operator.UBOUND):
                
                operation = BinaryOperation.create(op, Reference(sym1), Reference(sym2))

                assert (operation.datatype.intrinsic is ScalarType.Intrinsic.INTEGER)
                assert (operation.datatype.precision is ScalarType.Precision.UNDEFINED)
                assert isinstance(operation.datatype, ScalarType)

                check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                     operation.datatype.precision,
                                                     "scalar")],
                                    [operation])
                
                program.addchild(check)

        # CAST : (arg, scalar_type_kind) => scalar_type_kind
        if datatype2 in scalar_datatypes:
            operation = BinaryOperation.create(BinaryOperation.Operator.CAST,
                                               Reference(sym1), Reference(sym2))
            
            assert (operation.datatype.intrinsic is datatype2.intrinsic)
            assert (operation.datatype.precision is datatype2.precision)
            assert isinstance(operation.datatype, ScalarType)

            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                 operation.datatype.precision,
                                                 "scalar")],
                                [operation])
            
            program.addchild(check)

        # MATMUL : (array_ixj, array_jxk) => array_ixk
        if datatype1 in array_datatypes1 and datatype2 in array_datatypes2 \
            and not ((datatype1.intrinsic is ScalarType.Intrinsic.BOOLEAN) \
                     ^ (datatype2.intrinsic is ScalarType.Intrinsic.BOOLEAN)):
            operation = BinaryOperation.create(BinaryOperation.Operator.MATMUL,
                                               Reference(sym1), Reference(sym2))
            
            assert isinstance(operation.datatype, ArrayType)
            assert (operation.datatype.shape == [datatype1.shape[0], datatype2.shape[1]])

            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                 operation.datatype.precision,
                                                 f"{operation.datatype.shape[0].upper.value}x{operation.datatype.shape[1].upper.value}")],
                                [operation])
            
            program.addchild(check)

        # MATMUL : (array_ixj, vector_j) => vector_i
        if datatype1 in array_datatypes1 and datatype2 in vector_datatypes2 \
            and not ((datatype1.intrinsic is ScalarType.Intrinsic.BOOLEAN) \
                     ^ (datatype2.intrinsic is ScalarType.Intrinsic.BOOLEAN)):
            operation = BinaryOperation.create(BinaryOperation.Operator.MATMUL,
                                               Reference(sym1), Reference(sym2))
            
            assert isinstance(operation.datatype, ArrayType)
            assert (operation.datatype.shape == [datatype1.shape[0]])

            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                 operation.datatype.precision,
                                                 f"{operation.datatype.shape[0].upper.value}")],
                                [operation])
            
            program.addchild(check)
        
        # DOT_PRODUCT : (vector_i, vector_i) => scalar
        if datatype1 in vector_datatypes1 and datatype2 in vector_datatypes2 \
            and not ((datatype1.intrinsic is ScalarType.Intrinsic.BOOLEAN) \
                     ^ (datatype2.intrinsic is ScalarType.Intrinsic.BOOLEAN)):
            operation = BinaryOperation.create(BinaryOperation.Operator.DOT_PRODUCT,
                                               Reference(sym1), Reference(sym2))
            
            assert isinstance(operation.datatype, ScalarType)

            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                 operation.datatype.precision,
                                                 "scalar")],
                                [operation])
            
            program.addchild(check)

        # REAL, INT : (arg, k) => REAL*k/INT*k
        # NOTE: no typecheck functions for vectors
        if datatype1 in scalar_datatypes + array_datatypes1 \
            and (datatype1.intrinsic is not ScalarType.Intrinsic.BOOLEAN):
            operation = BinaryOperation.create(BinaryOperation.Operator.REAL,
                                            Reference(sym1), Literal("4", INTEGER_SINGLE_TYPE))
            assert (operation.datatype.intrinsic is ScalarType.Intrinsic.REAL)
            assert (operation.datatype.precision == 4)
            assert (type(operation.datatype) is type(datatype1))
            if datatype1 in array_datatypes1:
                    assert (operation.datatype.shape == datatype1.shape)
            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                operation.datatype.precision,
                                                key1)],
                                    [operation])
            program.addchild(check)

            operation = BinaryOperation.create(BinaryOperation.Operator.REAL,
                                            Reference(sym1), Literal("8", INTEGER_SINGLE_TYPE))
            assert (operation.datatype.intrinsic is ScalarType.Intrinsic.REAL)
            assert (operation.datatype.precision == 8)
            assert (type(operation.datatype) is type(datatype1))
            if datatype1 in array_datatypes1:
                    assert (operation.datatype.shape == datatype1.shape)
            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                operation.datatype.precision,
                                                key1)],
                                    [operation])
            program.addchild(check)

            operation = BinaryOperation.create(BinaryOperation.Operator.INT,
                                            Reference(sym1), Literal("4", INTEGER_SINGLE_TYPE))
            assert (operation.datatype.intrinsic is ScalarType.Intrinsic.INTEGER)
            assert (operation.datatype.precision == 4)
            assert (type(operation.datatype) is type(datatype1))
            if datatype1 in array_datatypes1:
                    assert (operation.datatype.shape == datatype1.shape)
            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                operation.datatype.precision,
                                                key1)],
                                    [operation])
            program.addchild(check)

            operation = BinaryOperation.create(BinaryOperation.Operator.INT,
                                            Reference(sym1), Literal("8", INTEGER_SINGLE_TYPE))
            assert (operation.datatype.intrinsic is ScalarType.Intrinsic.INTEGER)
            assert (operation.datatype.precision == 8)
            assert (type(operation.datatype) is type(datatype1))
            if datatype1 in array_datatypes1:
                    assert (operation.datatype.shape == datatype1.shape)
            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                operation.datatype.precision,
                                                key1)],
                                    [operation])
            program.addchild(check)

        # ADD, SUB, MUL, DIV, POW : (arg1, arg2) => promote
        if (datatype1 in scalar_datatypes and datatype2 in scalar_datatypes) \
            and (datatype1.intrinsic is not ScalarType.Intrinsic.BOOLEAN) \
            and (datatype2.intrinsic is not ScalarType.Intrinsic.BOOLEAN):
            for op in (BinaryOperation.Operator.ADD,
                       BinaryOperation.Operator.SUB,
                       BinaryOperation.Operator.MUL,
                       BinaryOperation.Operator.DIV,
                       BinaryOperation.Operator.POW):
                
                operation = BinaryOperation.create(op, Reference(sym1), Reference(sym2))

                assert isinstance(operation.datatype, ScalarType)

                check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                     operation.datatype.precision,
                                                     "scalar")],
                                    [operation])
                
                program.addchild(check)

        # REM = MOD, very messy promotion rules (in gfortran)
        if (datatype1 in scalar_datatypes and datatype2 in scalar_datatypes) \
            and (datatype1.intrinsic is not ScalarType.Intrinsic.BOOLEAN) \
            and (datatype2.intrinsic is not ScalarType.Intrinsic.BOOLEAN) \
            and (datatype1.intrinsic is datatype2.intrinsic):
            operation = BinaryOperation.create(BinaryOperation.Operator.REM, Reference(sym1), Reference(sym2))

            assert isinstance(operation.datatype, ScalarType)

            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                    operation.datatype.precision,
                                                    "scalar")],
                                [operation])
            
            program.addchild(check)

        # EQ, NE, GT, LT, GE, LE => default BOOLEAN
        if (datatype1 in scalar_datatypes and datatype2 in scalar_datatypes) \
            and (datatype1.intrinsic is not ScalarType.Intrinsic.BOOLEAN) \
            and (datatype2.intrinsic is not ScalarType.Intrinsic.BOOLEAN): \
            #and (datatype1.intrinsic is datatype2.intrinsic):
            for op in (BinaryOperation.Operator.EQ,
                       BinaryOperation.Operator.NE,
                       BinaryOperation.Operator.GT,
                       BinaryOperation.Operator.LT,
                       BinaryOperation.Operator.GE,
                       BinaryOperation.Operator.LE):
                operation = BinaryOperation.create(op, Reference(sym1), Reference(sym2))

                assert isinstance(operation.datatype, ScalarType)

                check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                     operation.datatype.precision,
                                                     "scalar")],
                                    [operation])
                
                program.addchild(check)

        # AND, OR, EQV, NEQV => default BOOLEAN
        if (datatype1 in scalar_datatypes and datatype2 in scalar_datatypes) \
            and (datatype1.intrinsic is ScalarType.Intrinsic.BOOLEAN) \
            and (datatype2.intrinsic is ScalarType.Intrinsic.BOOLEAN):
            for op in (BinaryOperation.Operator.AND,
                       BinaryOperation.Operator.OR,
                       BinaryOperation.Operator.EQV,
                       BinaryOperation.Operator.NEQV):
                operation = BinaryOperation.create(op, Reference(sym1), Reference(sym2))

                assert isinstance(operation.datatype, ScalarType)

                check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                     operation.datatype.precision,
                                                     "scalar")],
                                    [operation])
                
                program.addchild(check)

    _write_and_compile(fortran_writer, container, compiler_cmd)

def test_naryoperation_datatypes(fortran_writer, compiler_cmd = "gfortran"):
    container = FileContainer("naryoperation")

    container, routine_symbols = _initialize_type_checkers(container, [2,3])

    # No booleans
    integer_kinds, real_kinds, _ = _initialize_scalar_kinds()
    scalar_datatypes = integer_kinds + real_kinds

    datatypes1 = scalar_datatypes
    datatypes2 = scalar_datatypes

    program = Routine("naryoperation", is_program=True)
    container.addchild(program)

    for datatype1, datatype2 in product(datatypes1, datatypes2):
        sym1 = _initialize_datasymbol(datatype1)
        sym2 = _initialize_datasymbol(datatype2)
        
        for sym in (sym1, sym2):
            if sym.name not in program.symbol_table:
                program.symbol_table.add(sym)

        if datatype1.intrinsic is datatype2.intrinsic:
            for op in (NaryOperation.Operator.MAX, 
                       NaryOperation.Operator.MIN):
                
                operation = NaryOperation.create(op, [Reference(sym1), Reference(sym2)])

                check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                     operation.datatype.precision,
                                                     "scalar")],
                                    [operation])
                
                program.addchild(check)

    _write_and_compile(fortran_writer, container, compiler_cmd)

if __name__ == "__main__":
    fortran_writer = FortranWriter()
    test_unaryoperation_datatypes(fortran_writer, "gfortran")
    test_binaryoperation_datatypes(fortran_writer, "gfortran")
    test_naryoperation_datatypes(fortran_writer, "gfortran")
            