import pytest
from psyclone.psyir.dataflow.dataflow import DataFlowNode, AccessType
from psyclone.psyir.dataflow.dataflow import DataFlowDAG
from psyclone.psyir.nodes import (
    DataNode,
    Reference,
    Literal,
    IntrinsicCall,
    Call,
    UnaryOperation,
    Schedule,
    Routine,
    BinaryOperation,
    Assignment,
    ArrayReference,
    Loop,
    OMPParallelDirective,
    OMPPrivateClause,
    OMPFirstprivateClause,
    ACCParallelDirective,
    ACCDataDirective,
    ACCCopyClause,
    ACCCopyOutClause,
    ACCCopyInClause,
)
from psyclone.psyir.symbols import (
    DataSymbol,
    ArgumentInterface,
    REAL_TYPE,
    RoutineSymbol,
    ArrayType,
    INTEGER_TYPE,
)


def test_data_flow_node():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    operation = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("1.0", REAL_TYPE)
    )
    access_type = AccessType.READ

    # Test valid initialization
    node = DataFlowNode(dag, reference, access_type)
    assert node._dag == dag
    assert node._psyir == reference
    assert node._access_type == access_type
    assert node._forward_dependences == []
    assert node._backward_dependences == []

    # Test invalid initialization
    with pytest.raises(TypeError):
        DataFlowNode(False, reference, access_type)
    with pytest.raises(TypeError):
        DataFlowNode(dag, None, access_type)
    with pytest.raises(TypeError):
        DataFlowNode(dag, reference, None)
    with pytest.raises(TypeError):
        DataFlowNode(dag, reference, "invalid_access_type")
    with pytest.raises(ValueError):
        DataFlowNode(dag, reference, AccessType.UNKNOWN)
    with pytest.raises(ValueError):
        DataFlowNode(dag, operation, AccessType.WRITE)


def test_data_flow_node_properties():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    access_type = AccessType.READ

    node = DataFlowNode(dag, reference, access_type)

    # Test properties
    assert node.dag == dag
    assert node.psyir == reference
    assert node.access_type == access_type


def test_data_flow_node_add_forward_dependence():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference1 = Reference(datasymbol)
    reference2 = Reference(datasymbol)
    read = AccessType.READ
    write = AccessType.WRITE

    node1 = DataFlowNode(dag, reference1, read)
    node2 = DataFlowNode(dag, reference2, write)

    # Test valid addition
    node1.add_forward_dependence(node2)
    assert node1._forward_dependences == [node2]
    assert node2._backward_dependences == [node1]

    # Test adding the same dependence twice does not change the list
    node1.add_forward_dependence(node2)
    assert node1._forward_dependences == [node2]
    assert node2._backward_dependences == [node1]

    # Test invalid addition
    with pytest.raises(TypeError):
        node1.add_forward_dependence(False)
    with pytest.raises(TypeError):
        node1.add_forward_dependence("invalid_node")


def test_data_flow_node_add_backward_dependence():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference1 = Reference(datasymbol)
    reference2 = Reference(datasymbol)
    read = AccessType.READ
    write = AccessType.WRITE

    node1 = DataFlowNode(dag, reference1, read)
    node2 = DataFlowNode(dag, reference2, write)

    # Test valid addition
    node2.add_backward_dependence(node1)
    assert node2._backward_dependences == [node1]
    assert node1._forward_dependences == [node2]

    # Test adding the same dependence twice does not change the list
    node2.add_backward_dependence(node1)
    assert node2._backward_dependences == [node1]
    assert node1._forward_dependences == [node2]

    # Test invalid addition
    with pytest.raises(TypeError):
        node1.add_backward_dependence(False)
    with pytest.raises(TypeError):
        node1.add_backward_dependence("invalid_node")


def test_data_flow_node_create():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    operation = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("1.0", REAL_TYPE)
    )
    read = AccessType.READ
    write = AccessType.WRITE
    unknown = AccessType.UNKNOWN

    # Test valid creation
    node1 = DataFlowNode.create(dag, reference, write)
    assert node1._psyir == reference
    assert node1._access_type == write
    assert dag.dag_nodes == [node1]

    node2 = DataFlowNode.create(dag, operation, unknown)
    assert node2._psyir == operation
    assert node2._access_type == unknown
    assert len(dag.dag_nodes) == 3

    node3 = DataFlowNode.create(dag, reference, read)
    assert node3._psyir == reference
    assert node3._access_type == read
    assert node3._forward_dependences == []
    assert node3._backward_dependences == []

    binary_operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("1.0", REAL_TYPE),
        Literal("2.0", REAL_TYPE),
    )
    other_dag = DataFlowDAG()
    other_node = DataFlowNode.create(other_dag, binary_operation, unknown)
    assert len(other_dag.dag_nodes) == 3
    assert other_node._psyir == binary_operation
    assert other_node._access_type == unknown
    assert other_node._forward_dependences == []
    assert len(other_node._backward_dependences) == 2

    intrinsic_call = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.EXP, [Literal("1.0", REAL_TYPE)]
    )
    other_dag2 = DataFlowDAG()
    other_node2 = DataFlowNode.create(other_dag2, intrinsic_call, unknown)
    assert len(other_dag2.dag_nodes) == 2
    assert other_node2._psyir == intrinsic_call
    assert other_node2._access_type == unknown
    assert other_node2._forward_dependences == []
    assert len(other_node2._backward_dependences) == 1

    # Test invalid creation
    with pytest.raises(TypeError):
        DataFlowNode.create(dag, None, read)
    with pytest.raises(TypeError):
        DataFlowNode.create(dag, reference, None)
    with pytest.raises(TypeError):
        DataFlowNode.create(dag, reference, "invalid_access_type")
    with pytest.raises(ValueError):
        DataFlowNode.create(dag, operation, AccessType.READ)
    with pytest.raises(ValueError):
        DataFlowNode.create(dag, reference, AccessType.UNKNOWN)


def test_data_flow_node_create_or_get():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    operation = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("1.0", REAL_TYPE)
    )
    read = AccessType.READ
    write = AccessType.WRITE
    unknown = AccessType.UNKNOWN

    # Test valid creation
    node1 = DataFlowNode.create_or_get(dag, reference, write)
    assert node1._psyir == reference
    assert node1._access_type == write
    assert dag.dag_nodes == [node1]

    node2 = DataFlowNode.create_or_get(dag, operation, unknown)
    assert node2._psyir == operation
    assert node2._access_type == unknown
    assert len(dag.dag_nodes) == 3

    node3 = DataFlowNode.create_or_get(dag, reference, read)
    assert node3._psyir == reference
    assert node3._access_type == read
    assert node3._forward_dependences == []
    assert node3._backward_dependences == []
    assert len(dag.dag_nodes) == 4

    # Test valid get
    node4 = DataFlowNode.create_or_get(dag, reference, write)
    assert node4 is node1
    assert len(dag.dag_nodes) == 4

    # Test invalid creation
    with pytest.raises(TypeError):
        DataFlowNode.create_or_get(dag, None, read)
    with pytest.raises(TypeError):
        DataFlowNode.create_or_get(dag, reference, None)
    with pytest.raises(TypeError):
        DataFlowNode.create_or_get(dag, reference, "invalid_access_type")
    with pytest.raises(ValueError):
        DataFlowNode.create_or_get(dag, operation, AccessType.READ)
    with pytest.raises(ValueError):
        DataFlowNode.create_or_get(dag, reference, AccessType.UNKNOWN)


def test_data_flow_node_copy_single_node_to():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    operation = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, reference.copy()
    )
    read = AccessType.READ
    unknown = AccessType.UNKNOWN

    new_dag = DataFlowDAG()

    node1 = DataFlowNode(dag, reference, read)
    nodeA = DataFlowNode(dag, operation, unknown)

    # Test valid copy
    node2 = node1.copy_single_node_to(new_dag)
    assert node2._dag == new_dag
    assert node2._psyir == reference
    assert node2._access_type == read
    assert node2._forward_dependences == []
    assert node2._backward_dependences == []

    # Test invalid copy
    with pytest.raises(TypeError):
        node1.copy_single_node_to(False)
    with pytest.raises(TypeError):
        node1.copy_single_node_to("invalid_node")


def test_data_flow_node_copy_or_get_single_node_to():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    read = AccessType.READ

    new_dag = DataFlowDAG()

    node1 = DataFlowNode(dag, reference, read)

    # Test valid copy
    node2 = node1.copy_or_get_single_node_to(new_dag)
    assert node2._dag == new_dag
    assert node2._psyir == reference
    assert node2._access_type == read
    assert node2._forward_dependences == []
    assert node2._backward_dependences == []

    # Test valid get
    node3 = node1.copy_or_get_single_node_to(new_dag)
    assert node3 is node2

    # Test invalid copy
    with pytest.raises(TypeError):
        node1.copy_or_get_single_node_to(False)
    with pytest.raises(TypeError):
        node1.copy_or_get_single_node_to("invalid_node")


def test_data_flow_node_is_call_argument_reference():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol)
    call = Call.create(RoutineSymbol("a"), [reference])
    read = AccessType.READ

    node1 = DataFlowNode(dag, reference, read)
    node2 = DataFlowNode(dag, reference2, read)

    # Test valid call argument reference
    assert node1.is_call_argument_reference is True
    assert node2.is_call_argument_reference is False


def test_data_flow_node_copy_forward():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    operation = UnaryOperation.create(UnaryOperation.Operator.MINUS, reference)
    operation2 = UnaryOperation.create(UnaryOperation.Operator.MINUS, operation)
    read = AccessType.READ
    write = AccessType.WRITE
    unknown = AccessType.UNKNOWN

    op2_node = DataFlowNode.create(dag, operation2, unknown)
    assert len(dag.dag_nodes) == 3
    ref_node = dag.get_dag_node_for(reference, read)
    op_node = dag.get_dag_node_for(operation, unknown)

    # Test valid copy forward

    dag_copy1 = DataFlowDAG()
    copy_ref_node = ref_node.copy_forward(dag_copy1)
    assert len(dag_copy1.dag_nodes) == 3

    dag_copy2 = DataFlowDAG()
    copy_op_node = op_node.copy_forward(dag_copy2)
    assert len(op_node._forward_dependences) == 1
    assert len(copy_op_node._forward_dependences) == 1
    # assert len(dag_copy2.dag_nodes) == 2

    dag_copy3 = DataFlowDAG()
    copy_op2_node = op2_node.copy_forward(dag_copy3)
    assert len(dag_copy3.dag_nodes) == 1

    # Test invalid copy forward
    with pytest.raises(TypeError):
        ref_node.copy_forward(False)
    with pytest.raises(TypeError):
        ref_node.copy_forward(dag_copy1, False)
    with pytest.raises(TypeError):
        ref_node.copy_forward(dag_copy1, [], False)
    with pytest.raises(ValueError):
        ref_node.copy_forward(dag_copy1, [False], [False, False])
    with pytest.raises(ValueError):
        ref_node.copy_forward(dag_copy1, [ref_node], [])
    with pytest.raises(TypeError):
        ref_node.copy_forward(dag_copy1, [False], [copy_ref_node])
    with pytest.raises(TypeError):
        ref_node.copy_forward(dag_copy1, [ref_node], [False])


def test_data_flow_node_copy_backward():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    operation = UnaryOperation.create(UnaryOperation.Operator.MINUS, reference)
    operation2 = UnaryOperation.create(UnaryOperation.Operator.MINUS, operation)
    read = AccessType.READ
    write = AccessType.WRITE
    unknown = AccessType.UNKNOWN

    op2_node = DataFlowNode.create(dag, operation2, unknown)
    assert len(dag.dag_nodes) == 3
    ref_node = dag.get_dag_node_for(reference, read)
    op_node = dag.get_dag_node_for(operation, unknown)

    # Test valid copy backward

    dag_copy1 = DataFlowDAG()
    copy_op2_node = op2_node.copy_backward(dag_copy1)
    assert len(dag_copy1.dag_nodes) == 3

    dag_copy2 = DataFlowDAG()
    copy_op_node = op_node.copy_backward(dag_copy2)
    assert len(op_node._backward_dependences) == 1
    assert len(copy_op_node._backward_dependences) == 1
    # assert len(dag_copy2.dag_nodes) == 2

    dag_copy3 = DataFlowDAG()
    copy_ref_node = ref_node.copy_backward(dag_copy3)
    assert len(dag_copy3.dag_nodes) == 1

    # Test invalid copy backward
    with pytest.raises(TypeError):
        ref_node.copy_backward(False)
    with pytest.raises(TypeError):
        ref_node.copy_backward(dag_copy1, False)
    with pytest.raises(TypeError):
        ref_node.copy_backward(dag_copy1, [], False)
    with pytest.raises(ValueError):
        ref_node.copy_backward(dag_copy1, [False], [False, False])
    with pytest.raises(ValueError):
        ref_node.copy_backward(dag_copy1, [ref_node], [])
    with pytest.raises(TypeError):
        ref_node.copy_backward(dag_copy1, [False], [copy_ref_node])
    with pytest.raises(TypeError):
        ref_node.copy_backward(dag_copy1, [ref_node], [False])


def test_data_flow_node_to_psyir_list_forward():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    operation = UnaryOperation.create(UnaryOperation.Operator.MINUS, reference)
    operation2 = UnaryOperation.create(UnaryOperation.Operator.MINUS, operation)
    read = AccessType.READ
    write = AccessType.WRITE
    unknown = AccessType.UNKNOWN

    op2_node = DataFlowNode.create(dag, operation2, unknown)
    ref_node = dag.get_dag_node_for(reference, read)
    op_node = dag.get_dag_node_for(operation, unknown)

    # Test valid to_psyir_list_forward
    psyir_list = ref_node.to_psyir_list_forward()
    assert len(psyir_list) == 3
    assert (
        reference in psyir_list
        and operation in psyir_list
        and operation2 in psyir_list
    )

    psyir_list = op_node.to_psyir_list_forward()
    assert len(psyir_list) == 2
    assert operation in psyir_list and operation2 in psyir_list

    psyir_list = op2_node.to_psyir_list_forward()
    assert len(psyir_list) == 1
    assert operation2 in psyir_list


def test_data_flow_node_to_psyir_list_backward():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    operation = UnaryOperation.create(UnaryOperation.Operator.MINUS, reference)
    operation2 = UnaryOperation.create(UnaryOperation.Operator.MINUS, operation)
    read = AccessType.READ
    write = AccessType.WRITE
    unknown = AccessType.UNKNOWN

    op2_node = DataFlowNode.create(dag, operation2, unknown)
    ref_node = dag.get_dag_node_for(reference, read)
    op_node = dag.get_dag_node_for(operation, unknown)

    # Test valid to_psyir_list_backward
    psyir_list = op2_node.to_psyir_list_backward()
    assert len(psyir_list) == 3
    assert (
        reference in psyir_list
        and operation in psyir_list
        and operation2 in psyir_list
    )

    psyir_list = op_node.to_psyir_list_backward()
    assert len(psyir_list) == 2
    assert operation in psyir_list and reference in psyir_list

    psyir_list = ref_node.to_psyir_list_backward()
    assert len(psyir_list) == 1
    assert reference in psyir_list


def test_data_flow_dag():
    # Test valid initialization
    dag = DataFlowDAG()
    assert dag.dag_nodes == []
    assert dag.schedule is None


def test_data_flow_dag_create_from_schedule():
    schedule = Schedule()
    symbol_a = DataSymbol("a", REAL_TYPE)
    symbol_b = DataSymbol("b", REAL_TYPE)
    symbol_c = DataSymbol("c", REAL_TYPE)
    sum = BinaryOperation.create(
        BinaryOperation.Operator.ADD, Reference(symbol_b), Reference(symbol_c)
    )
    assignment = Assignment.create(Reference(symbol_a), sum)
    schedule.addchild(assignment)

    # Test valid creation
    dag = DataFlowDAG.create_from_schedule(schedule)
    assert len(dag.dag_nodes) == 4

    new_assignment = Assignment.create(Reference(symbol_a), Reference(symbol_b))
    new_schedule = Schedule()
    new_schedule.addchild(assignment.copy())
    new_schedule.addchild(new_assignment)
    new_dag = DataFlowDAG.create_from_schedule(new_schedule)
    assert len(new_dag.dag_nodes) == 6

    symbol_a = DataSymbol(
        "a",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.WRITE),
    )
    symbol_b = DataSymbol(
        "b",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.READ),
    )
    symbol_c = DataSymbol(
        "c",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.READWRITE),
    )
    sum = BinaryOperation.create(
        BinaryOperation.Operator.ADD, Reference(symbol_b), Reference(symbol_c)
    )
    assignment = Assignment.create(Reference(symbol_a), sum)
    new_assignment = Assignment.create(Reference(symbol_a), Reference(symbol_b))
    routine = Routine("test_routine")
    routine.symbol_table._argument_list.append(symbol_a)
    routine.symbol_table._argument_list.append(symbol_b)
    routine.symbol_table._argument_list.append(symbol_c)
    routine.symbol_table.add(symbol_a)
    routine.symbol_table.add(symbol_b)
    routine.symbol_table.add(symbol_c)

    routine.addchild(assignment.copy())
    routine.addchild(new_assignment.copy())
    new_dag = DataFlowDAG.create_from_schedule(routine)
    assert len(new_dag.dag_nodes) == 10

    # Test invalid creation
    with pytest.raises(TypeError):
        DataFlowDAG.create_from_schedule(False)
    with pytest.raises(TypeError):
        DataFlowDAG.create_from_schedule(1)
    with pytest.raises(TypeError):
        DataFlowDAG.create_from_schedule(None)


def test_data_flow_dag_get_dag_node_for():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    datasymbol2 = DataSymbol("b", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)
    read = AccessType.READ
    write = AccessType.WRITE

    node = DataFlowNode.create(dag, reference, read)

    # Test valid get
    assert dag.get_dag_node_for(reference, read) is node
    assert dag.get_dag_node_for(reference, write) is None
    assert dag.get_dag_node_for(reference2, read) is None

    node2 = DataFlowNode.create(dag, reference, read)
    with pytest.raises(ValueError):
        dag.get_dag_node_for(reference, read)

    # Test invalid get
    with pytest.raises(TypeError):
        dag.get_dag_node_for(False, read)
    with pytest.raises(TypeError):
        dag.get_dag_node_for(reference, None)
    with pytest.raises(TypeError):
        dag.get_dag_node_for(reference, "invalid_access_type")


def test_data_flow_dag_to_psyir_list():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    datasymbol2 = DataSymbol("b", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, reference, reference2
    )
    read = AccessType.READ
    write = AccessType.WRITE

    node = DataFlowNode.create(dag, operation, AccessType.UNKNOWN)

    # Test valid to_psyir_list
    psyir_list = dag.to_psyir_list()
    assert len(psyir_list) == 3
    assert (
        reference in psyir_list
        and reference2 in psyir_list
        and operation in psyir_list
    )


def test_data_flow_dag_dataflow_tree_from():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    datasymbol2 = DataSymbol("b", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, reference, reference2
    )
    read = AccessType.READ
    write = AccessType.WRITE

    node = DataFlowNode.create(dag, operation, AccessType.UNKNOWN)
    assert len(node.backward_dependences) == 2
    assert len(dag.backward_leaves) == 2
    assert dag.get_dag_node_for(reference, read) in dag.backward_leaves
    assert dag.get_dag_node_for(reference2, read) in dag.backward_leaves
    assert node in dag.forward_leaves
    assert len(dag.forward_leaves) == 1
    assert len(dag.get_dag_node_for(reference, read).forward_dependences) == 1
    assert len(dag.get_dag_node_for(reference2, read).forward_dependences) == 1

    # Test valid dataflow_tree_from
    tree = dag.dataflow_tree_from(reference)
    assert len(tree.dag_nodes) == 2
    assert reference in [dag_node.psyir for dag_node in tree.dag_nodes]
    assert operation in [dag_node.psyir for dag_node in tree.dag_nodes]

    tree2 = dag.dataflow_tree_from(reference2)
    assert tree2 is not tree
    assert len(tree2.dag_nodes) == 2
    assert reference2 in [dag_node.psyir for dag_node in tree2.dag_nodes]
    assert operation in [dag_node.psyir for dag_node in tree2.dag_nodes]

    tree3 = dag.dataflow_tree_from(operation)
    assert [operation] == [dag_node.psyir for dag_node in tree3.dag_nodes]

    # Test invalid dataflow_tree_from
    with pytest.raises(TypeError):
        dag.dataflow_tree_from(False)
    with pytest.raises(TypeError):
        dag.dataflow_tree_from(None)
    datasymbol3 = DataSymbol("c", REAL_TYPE, interface=ArgumentInterface())
    reference3 = Reference(datasymbol3)
    with pytest.raises(ValueError):
        dag.dataflow_tree_from(reference3)


def test_data_flow_dag_dataflow_tree_to():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    datasymbol2 = DataSymbol("b", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, reference, reference2
    )
    read = AccessType.READ
    write = AccessType.WRITE

    node = DataFlowNode.create(dag, operation, AccessType.UNKNOWN)
    assert len(node.backward_dependences) == 2
    assert len(dag.backward_leaves) == 2
    assert dag.get_dag_node_for(reference, read) in dag.backward_leaves
    assert dag.get_dag_node_for(reference2, read) in dag.backward_leaves
    assert node in dag.forward_leaves
    assert len(dag.forward_leaves) == 1
    assert len(dag.get_dag_node_for(reference, read).forward_dependences) == 1
    assert len(dag.get_dag_node_for(reference2, read).forward_dependences) == 1

    # Test valid dataflow_tree_to
    tree = dag.dataflow_tree_to(operation)
    assert len(tree.dag_nodes) == 3
    assert operation in [dag_node.psyir for dag_node in tree.dag_nodes]
    assert reference in [dag_node.psyir for dag_node in tree.dag_nodes]
    assert reference2 in [dag_node.psyir for dag_node in tree.dag_nodes]

    tree2 = dag.dataflow_tree_to(reference)
    assert tree2 is not tree
    assert [reference] == [dag_node.psyir for dag_node in tree2.dag_nodes]

    tree3 = dag.dataflow_tree_to(reference2)
    assert [reference2] == [dag_node.psyir for dag_node in tree3.dag_nodes]

    # Test invalid dataflow_tree_to
    with pytest.raises(TypeError):
        dag.dataflow_tree_to(False)
    with pytest.raises(TypeError):
        dag.dataflow_tree_to(None)
    datasymbol3 = DataSymbol("c", REAL_TYPE, interface=ArgumentInterface())
    reference3 = Reference(datasymbol3)
    with pytest.raises(ValueError):
        dag.dataflow_tree_to(reference3)


def test_data_flow_dag_all_reads():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    datasymbol2 = DataSymbol("b", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, reference, reference2
    )
    read = AccessType.READ
    write = AccessType.WRITE

    node = DataFlowNode.create(dag, operation, AccessType.UNKNOWN)

    # Test valid all_reads
    reads = dag.all_reads
    assert len(reads) == 2
    assert reference in [read.psyir for read in reads]
    assert reference2 in [read.psyir for read in reads]


def test_data_flow_dag_all_writes():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    datasymbol2 = DataSymbol("b", REAL_TYPE, interface=ArgumentInterface())
    datasymbol3 = DataSymbol("c", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)
    reference3 = Reference(datasymbol3)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, reference, reference2
    )
    read = AccessType.READ
    write = AccessType.WRITE

    schedule = Schedule()

    # Test valid all_writes
    writes = dag.all_writes
    assert len(writes) == 0

    assignment = Assignment.create(reference3, operation)
    schedule.addchild(assignment)
    other_dag = DataFlowDAG.create_from_schedule(schedule)

    writes = other_dag.all_writes
    assert len(writes) == 1
    assert reference3 in [write.psyir for write in writes]


def test_data_flow_all_reads_from():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    datasymbol3 = DataSymbol("c", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference3 = Reference(datasymbol3)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        reference.copy(),
        BinaryOperation.create(
            BinaryOperation.Operator.ADD, reference.copy(), reference3.copy()
        ),
    )
    read = AccessType.READ
    write = AccessType.WRITE

    schedule = Schedule()

    # Test valid all_reads_from
    reads = dag.all_reads_from(reference)
    assert len(reads) == 0

    assignment = Assignment.create(reference3, operation)
    schedule.addchild(assignment)
    other_dag = DataFlowDAG.create_from_schedule(schedule)

    reads = other_dag.all_reads_from(reference)
    assert len(reads) == 2
    reads = other_dag.all_reads_from(reference3)
    assert len(reads) == 1

    assignment2 = Assignment.create(reference3.copy(), reference.copy())
    schedule.addchild(assignment2)
    other_dag2 = DataFlowDAG.create_from_schedule(schedule)

    reads = other_dag2.all_reads_from(reference)
    assert len(reads) == 3
    reads = other_dag2.all_reads_from(reference3)
    assert len(reads) == 1


def test_data_flow_all_writes_to():
    dag = DataFlowDAG()
    datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
    datasymbol3 = DataSymbol("c", REAL_TYPE, interface=ArgumentInterface())
    reference = Reference(datasymbol)
    reference3 = Reference(datasymbol3)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        reference.copy(),
        BinaryOperation.create(
            BinaryOperation.Operator.ADD, reference.copy(), reference3.copy()
        ),
    )
    read = AccessType.READ
    write = AccessType.WRITE

    schedule = Schedule()

    # Test valid all_writes_to
    writes = dag.all_writes_to(reference)
    assert len(writes) == 0

    assignment = Assignment.create(reference3, operation)
    schedule.addchild(assignment)
    other_dag = DataFlowDAG.create_from_schedule(schedule)

    writes = other_dag.all_writes_to(reference3)
    assert len(writes) == 1

    assignment2 = Assignment.create(reference3.copy(), reference.copy())
    schedule.addchild(assignment2)
    other_dag2 = DataFlowDAG.create_from_schedule(schedule)

    writes = other_dag2.all_writes_to(reference3)
    assert len(writes) == 2


def test_data_flow_dag_node_position():
    datasymbol = DataSymbol(
        "a",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.READ),
    )
    datasymbol2 = DataSymbol(
        "b",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.WRITE),
    )
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, reference, reference2
    )
    read = AccessType.READ
    write = AccessType.WRITE

    routine = Routine("test_routine")
    routine.symbol_table._argument_list.append(datasymbol)
    routine.symbol_table._argument_list.append(datasymbol2)
    routine.symbol_table.add(datasymbol)
    routine.symbol_table.add(datasymbol2)

    assignment = Assignment.create(reference2.copy(), operation)
    routine.addchild(assignment)

    dag = DataFlowDAG.create_from_schedule(routine)
    node = dag.get_dag_node_for(operation, AccessType.UNKNOWN)
    node2 = dag.get_dag_node_for(reference, read)
    node3 = dag.get_dag_node_for(reference2, read)
    node4 = dag.get_dag_node_for(datasymbol, write)
    node5 = dag.get_dag_node_for(datasymbol2, read)

    # Test valid node_position
    assert dag.node_position(node) == node.psyir.abs_position
    assert dag.node_position(node2) == node2.psyir.abs_position
    assert dag.node_position(node3) == node3.psyir.abs_position
    assert dag.node_position(node4) == -1
    assert dag.node_position(node5) == 1000000

    # Test errors
    with pytest.raises(TypeError):
        dag.node_position(False)
    with pytest.raises(ValueError):
        node6 = DataFlowNode.create(dag, datasymbol, AccessType.UNKNOWN)
        dag.node_position(node6)


def test_data_flow_dag_last_write_before_scalar_case():
    # Scalar case
    datasymbol = DataSymbol(
        "a",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.READ),
    )
    datasymbol2 = DataSymbol(
        "b",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.WRITE),
    )
    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, reference, reference2
    )
    read = AccessType.READ
    write = AccessType.WRITE

    routine = Routine("test_routine")
    routine.symbol_table._argument_list.append(datasymbol)
    routine.symbol_table._argument_list.append(datasymbol2)
    routine.symbol_table.add(datasymbol)
    routine.symbol_table.add(datasymbol2)

    reference2_copy = reference2.copy()
    assignment = Assignment.create(reference2_copy, operation)
    routine.addchild(assignment)

    dag = DataFlowDAG.create_from_schedule(routine)
    node = dag.get_dag_node_for(operation, AccessType.UNKNOWN)
    node2 = dag.get_dag_node_for(reference, read)
    node3 = dag.get_dag_node_for(reference2, read)
    node4 = dag.get_dag_node_for(reference2_copy, write)  # LHS of b = a + b
    node5 = dag.get_dag_node_for(datasymbol, write)
    node6 = dag.get_dag_node_for(datasymbol2, read)

    # Test valid last_write_before
    assert dag.last_write_before(node2) is node5
    assert dag.last_write_before(node3) is None
    assert dag.last_write_before(node5) is None
    assert dag.last_write_before(node6) is node4

    # Test invalid last_write_before
    with pytest.raises(TypeError):
        dag.last_write_before(False)
    with pytest.raises(TypeError):
        dag.last_write_before(1)
    with pytest.raises(TypeError):
        dag.last_write_before(None)


def test_data_flow_dag_last_write_before_array_case():
    #####
    # Array case
    datasymbol_a = DataSymbol(
        "a",
        ArrayType(REAL_TYPE, shape=[10]),
        interface=ArgumentInterface(access=ArgumentInterface.Access.READ),
    )
    datasymbol_b = DataSymbol(
        "b",
        ArrayType(REAL_TYPE, shape=[10]),
        interface=ArgumentInterface(access=ArgumentInterface.Access.WRITE),
    )
    datasymbol_c = DataSymbol(
        "c",
        ArrayType(REAL_TYPE, shape=[10]),
    )
    datasymbol_i = DataSymbol("i", INTEGER_TYPE)

    reference_a = Reference(datasymbol_a)
    reference_b = Reference(datasymbol_b)
    reference_c = Reference(datasymbol_c)
    reference_i = Reference(datasymbol_i)

    array_reference_a1 = ArrayReference.create(
        datasymbol_a, [Literal("1", INTEGER_TYPE)]
    )
    array_reference_a2 = ArrayReference.create(
        datasymbol_a, [Literal("2", INTEGER_TYPE)]
    )

    array_reference_b1 = ArrayReference.create(
        datasymbol_b, [Literal("1", INTEGER_TYPE)]
    )
    array_reference_b2 = ArrayReference.create(
        datasymbol_b, [Literal("2", INTEGER_TYPE)]
    )

    array_reference_c1 = ArrayReference.create(
        datasymbol_c, [Literal("1", INTEGER_TYPE)]
    )
    array_reference_c2 = ArrayReference.create(
        datasymbol_c, [Literal("2", INTEGER_TYPE)]
    )

    # subroutine test_routine(a:in, b:out)
    routine = Routine("test_routine")
    routine.symbol_table._argument_list.append(datasymbol_a)
    routine.symbol_table._argument_list.append(datasymbol_b)
    routine.symbol_table.add(datasymbol_a)
    routine.symbol_table.add(datasymbol_b)
    routine.symbol_table.add(datasymbol_c)
    routine.symbol_table.add(datasymbol_i)

    # b = a
    assignment1 = Assignment.create(reference_b, reference_a)
    routine.addchild(assignment1)

    dag = DataFlowDAG.create_from_schedule(routine)
    node_arg_a = dag.get_dag_node_for(datasymbol_a, AccessType.WRITE)
    node_arg_b = dag.get_dag_node_for(datasymbol_b, AccessType.READ)
    node_ref_a = dag.get_dag_node_for(reference_a, AccessType.READ)
    node_ref_b = dag.get_dag_node_for(reference_b, AccessType.WRITE)

    # Test valid last_write_before
    assert dag.last_write_before(node_ref_a) is node_arg_a
    assert dag.last_write_before(node_ref_b) is None
    assert dag.last_write_before(node_arg_a) is None
    assert dag.last_write_before(node_arg_b) is node_ref_b

    # b = a
    # c = b
    reference_b_copy = reference_b.copy()
    assignment2 = Assignment.create(reference_c, reference_b_copy)
    routine.addchild(assignment2)

    dag = DataFlowDAG.create_from_schedule(routine)
    node_arg_a = dag.get_dag_node_for(datasymbol_a, AccessType.WRITE)
    node_arg_b = dag.get_dag_node_for(datasymbol_b, AccessType.READ)
    node_ref_a = dag.get_dag_node_for(reference_a, AccessType.READ)
    node_ref_b_write = dag.get_dag_node_for(reference_b, AccessType.WRITE)
    node_ref_b_read = dag.get_dag_node_for(reference_b_copy, AccessType.READ)
    node_ref_c = dag.get_dag_node_for(reference_c, AccessType.WRITE)

    # Test valid last_write_before
    assert dag.last_write_before(node_ref_a) is node_arg_a
    assert dag.last_write_before(node_ref_b_write) is None
    assert dag.last_write_before(node_ref_b_read) is node_ref_b_write
    assert dag.last_write_before(node_ref_c) is None

    # b = a
    # c = b
    # b(1) = c(1)
    # c(2) = b(2)
    array_assignment1 = Assignment.create(
        array_reference_b1, array_reference_c1
    )
    routine.addchild(array_assignment1)
    array_assignment2 = Assignment.create(
        array_reference_c2, array_reference_b2
    )
    routine.addchild(array_assignment2)

    dag = DataFlowDAG.create_from_schedule(routine)
    node_arg_b = dag.get_dag_node_for(datasymbol_b, AccessType.READ)
    node_ref_a = dag.get_dag_node_for(reference_a, AccessType.READ)
    node_ref_b_read = dag.get_dag_node_for(reference_b_copy, AccessType.READ)
    node_ref_b_write = dag.get_dag_node_for(reference_b, AccessType.WRITE)
    node_ref_c = dag.get_dag_node_for(reference_c, AccessType.WRITE)
    node_ref_b1 = dag.get_dag_node_for(array_reference_b1, AccessType.WRITE)
    node_ref_c1 = dag.get_dag_node_for(array_reference_c1, AccessType.READ)
    node_ref_b2 = dag.get_dag_node_for(array_reference_b2, AccessType.READ)
    node_ref_c2 = dag.get_dag_node_for(array_reference_c2, AccessType.WRITE)

    # Test valid last_write_before
    assert dag.last_write_before(node_ref_c2) is node_ref_c
    assert dag.last_write_before(node_ref_b1) is node_ref_b_write
    assert dag.last_write_before(node_ref_c) is None
    assert dag.last_write_before(node_ref_b_read) is node_ref_b_write
    assert dag.last_write_before(node_arg_b) is node_ref_b1

    # Test array indexing within same loop
    # subroutine test_routine(a:in, b:out)
    routine = Routine("test_routine")
    routine.symbol_table._argument_list.append(datasymbol_a)
    routine.symbol_table._argument_list.append(datasymbol_b)
    routine.symbol_table.add(datasymbol_a)
    routine.symbol_table.add(datasymbol_b)
    routine.symbol_table.add(datasymbol_c)
    routine.symbol_table.add(datasymbol_i)
    # do i = 1, 9, 1
    #   b(i) = b(i) + a(i)
    #   b(i+1) = b(i+1) + c(i+1)
    # end do
    array_reference_bi_write = ArrayReference.create(
        datasymbol_b, [reference_i]
    )
    array_reference_bi_read = array_reference_bi_write.copy()
    array_reference_bi1_write = ArrayReference.create(
        datasymbol_b,
        [
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                reference_i.copy(),
                Literal("1", INTEGER_TYPE),
            )
        ],
    )
    array_reference_bi1_read = array_reference_bi1_write.copy()
    array_reference_ai_read = ArrayReference.create(
        datasymbol_a, [reference_i.copy()]
    )
    array_reference_ci1_read = ArrayReference.create(
        datasymbol_c,
        [
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                reference_i.copy(),
                Literal("1", INTEGER_TYPE),
            )
        ],
    )
    loop = Loop.create(
        datasymbol_i,
        Literal("1", INTEGER_TYPE),
        Literal("9", INTEGER_TYPE),
        Literal("1", INTEGER_TYPE),
        [],
    )
    assignment1 = Assignment.create(
        array_reference_bi_write,
        BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            array_reference_bi_read,
            array_reference_ai_read,
        ),
    )
    assignment2 = Assignment.create(
        array_reference_bi1_write,
        BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            array_reference_bi1_read,
            array_reference_ci1_read,
        ),
    )
    loop.loop_body.addchild(assignment1)
    loop.loop_body.addchild(assignment2)
    routine.addchild(loop)

    dag = DataFlowDAG.create_from_schedule(routine)
    node_arg_b = dag.get_dag_node_for(datasymbol_b, AccessType.READ)
    node_ref_bi_write = dag.get_dag_node_for(
        array_reference_bi_write, AccessType.WRITE
    )
    node_ref_bi1_write = dag.get_dag_node_for(
        array_reference_bi1_write, AccessType.WRITE
    )
    node_ref_bi_read = dag.get_dag_node_for(
        array_reference_bi_read, AccessType.READ
    )
    node_ref_bi1_read = dag.get_dag_node_for(
        array_reference_bi1_read, AccessType.READ
    )
    node_ref_ai_read = dag.get_dag_node_for(
        array_reference_ai_read, AccessType.READ
    )
    node_ref_ci1_read = dag.get_dag_node_for(
        array_reference_ci1_read, AccessType.READ
    )

    # Test valid last_write_before
    assert dag.last_write_before(node_ref_bi_write) is None
    assert dag.last_write_before(node_ref_bi1_write) is None
    assert dag.last_write_before(node_arg_b) is node_ref_bi1_write
    assert node_ref_bi_read.backward_dependences == []
    assert node_ref_bi1_read.backward_dependences == []
    assert len(node_arg_b.backward_dependences) == 2

    # Test array indexing within loop and outside loop
    # subroutine test_routine(a:in, b:out)
    routine = Routine("test_routine")
    routine.symbol_table._argument_list.append(datasymbol_a)
    routine.symbol_table._argument_list.append(datasymbol_b)
    routine.symbol_table.add(datasymbol_a)
    routine.symbol_table.add(datasymbol_b)
    routine.symbol_table.add(datasymbol_c)
    routine.symbol_table.add(datasymbol_i)
    # i = 1
    # b(i+1) = c(i+1)
    # do i = 1, 9, 1
    #   b(i) = b(i) + a(i)
    #   b(i+1) = b(i+1) + 2.0
    # end do
    # i = 1
    # c(i) = b(i)
    array_reference_bi_write = ArrayReference.create(
        datasymbol_b, [reference_i.copy()]
    )
    array_reference_bi_read = array_reference_bi_write.copy()
    array_reference_bi1_write = ArrayReference.create(
        datasymbol_b,
        [
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                reference_i.copy(),
                Literal("1", INTEGER_TYPE),
            )
        ],
    )
    array_reference_bi1_read = array_reference_bi1_write.copy()
    array_reference_ai_read = ArrayReference.create(
        datasymbol_a, [reference_i.copy()]
    )
    loop = Loop.create(
        datasymbol_i,
        Literal("1", INTEGER_TYPE),
        Literal("9", INTEGER_TYPE),
        Literal("1", INTEGER_TYPE),
        [],
    )
    assignment1 = Assignment.create(
        array_reference_bi_write,
        BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            array_reference_bi_read,
            array_reference_ai_read,
        ),
    )
    assignment2 = Assignment.create(
        array_reference_bi1_write,
        BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            array_reference_bi1_read,
            Literal("2.0", REAL_TYPE),
        ),
    )
    loop.loop_body.addchild(assignment1)
    loop.loop_body.addchild(assignment2)
    array_reference_bi_read2 = array_reference_bi_write.copy()
    array_reference_bi1_write2 = array_reference_bi1_write.copy()
    array_reference_ci1_read = ArrayReference.create(
        datasymbol_c,
        [
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                reference_i.copy(),
                Literal("1", INTEGER_TYPE),
            )
        ],
    )
    array_reference_ci_write = ArrayReference.create(
        datasymbol_c, [reference_i.copy()]
    )
    assignment3 = Assignment.create(
        array_reference_bi1_write2, array_reference_ci1_read
    )
    assignment4 = Assignment.create(
        array_reference_ci_write, array_reference_bi_read2
    )

    routine.addchild(assignment3)
    routine.addchild(loop)
    routine.addchild(assignment4)

    dag = DataFlowDAG.create_from_schedule(routine)
    node_ref_bi1_write = dag.get_dag_node_for(
        array_reference_bi1_write, AccessType.WRITE
    )
    node_ref_bi_read2 = dag.get_dag_node_for(
        array_reference_bi_read2, AccessType.READ
    )
    node_ref_bi_read = dag.get_dag_node_for(
        array_reference_bi_read, AccessType.READ
    )
    node_ref_bi_write = dag.get_dag_node_for(
        array_reference_bi_write, AccessType.WRITE
    )
    node_ref_bi1_write = dag.get_dag_node_for(
        array_reference_bi1_write, AccessType.WRITE
    )
    node_ref_bi1_write2 = dag.get_dag_node_for(
        array_reference_bi1_write2, AccessType.WRITE
    )
    node_datasymbol_b = dag.get_dag_node_for(datasymbol_b, AccessType.READ)

    assert len(node_datasymbol_b.backward_dependences) == 3
    assert node_ref_bi1_write in node_datasymbol_b.backward_dependences
    assert node_ref_bi1_write2 in node_datasymbol_b.backward_dependences
    assert node_ref_bi_write in node_datasymbol_b.backward_dependences

    # Test array indexing within two different loops
    # subroutine test_routine(a:in, b:out)
    routine = Routine("test_routine")
    routine.symbol_table._argument_list.append(datasymbol_a)
    routine.symbol_table._argument_list.append(datasymbol_b)
    routine.symbol_table.add(datasymbol_a)
    routine.symbol_table.add(datasymbol_b)
    routine.symbol_table.add(datasymbol_c)
    routine.symbol_table.add(datasymbol_i)

    # do i = 1, 10, 1
    #   b(i) = b(i) + a(i)
    # end do
    # do i = 1, 9, 1
    #   b(i+1) = b(i+1) + a(i+1)
    # end do
    loop1 = Loop.create(
        datasymbol_i,
        Literal("1", INTEGER_TYPE),
        Literal("10", INTEGER_TYPE),
        Literal("1", INTEGER_TYPE),
        [],
    )
    loop2 = Loop.create(
        datasymbol_i,
        Literal("1", INTEGER_TYPE),
        Literal("9", INTEGER_TYPE),
        Literal("1", INTEGER_TYPE),
        [],
    )
    array_reference_bi_write = ArrayReference.create(
        datasymbol_b, [reference_i.copy()]
    )
    array_reference_bi_read = array_reference_bi_write.copy()
    array_reference_bi1_write = ArrayReference.create(
        datasymbol_b,
        [
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                reference_i.copy(),
                Literal("1", INTEGER_TYPE),
            )
        ],
    )
    array_reference_bi1_read = array_reference_bi1_write.copy()
    array_reference_ai_read = ArrayReference.create(
        datasymbol_a, [reference_i.copy()]
    )
    array_reference_ai1_read = ArrayReference.create(
        datasymbol_a,
        [
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                reference_i.copy(),
                Literal("1", INTEGER_TYPE),
            )
        ],
    )
    assignment1 = Assignment.create(
        array_reference_bi_write,
        BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            array_reference_bi_read,
            array_reference_ai_read,
        ),
    )
    loop1.loop_body.addchild(assignment1)
    assignment2 = Assignment.create(
        array_reference_bi1_write,
        BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            array_reference_bi1_read,
            array_reference_ai1_read,
        ),
    )
    loop2.loop_body.addchild(assignment2)
    routine.addchild(loop1)
    routine.addchild(loop2)

    dag = DataFlowDAG.create_from_schedule(routine)
    node_ref_bi_write = dag.get_dag_node_for(
        array_reference_bi_write, AccessType.WRITE
    )
    node_ref_bi_read = dag.get_dag_node_for(
        array_reference_bi_read, AccessType.READ
    )
    node_ref_bi1_write = dag.get_dag_node_for(
        array_reference_bi1_write, AccessType.WRITE
    )
    node_ref_bi1_read = dag.get_dag_node_for(
        array_reference_bi1_read, AccessType.READ
    )
    node_datasymbol_b = dag.get_dag_node_for(datasymbol_b, AccessType.READ)

    # Test valid last_write_before
    assert dag.last_write_before(node_ref_bi_write) is None
    assert dag.last_write_before(node_ref_bi1_read) is node_ref_bi_write
    assert dag.last_write_before(node_ref_bi1_write) is node_ref_bi_write
    assert dag.last_write_before(node_datasymbol_b) is node_ref_bi1_write
    assert len(node_datasymbol_b.backward_dependences) == 2


def test_data_flow_dag_last_write_before_directives_and_clauses():
    datasymbol_a = DataSymbol(
        "a",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.READWRITE),
    )
    datasymbol_b = DataSymbol(
        "b",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.READWRITE),
    )
    datasymbol_c = DataSymbol(
        "c",
        REAL_TYPE,
        interface=ArgumentInterface(access=ArgumentInterface.Access.READWRITE),
    )
    ref_a_read1 = Reference(datasymbol_a)
    ref_b_read1 = Reference(datasymbol_b)
    ref_c_read1 = Reference(datasymbol_c)
    ref_a_read2 = Reference(datasymbol_a)
    ref_b_read2 = Reference(datasymbol_b)
    ref_c_read2 = Reference(datasymbol_c)
    ref_a_read3 = Reference(datasymbol_a)
    ref_b_read3 = Reference(datasymbol_b)
    ref_c_read3 = Reference(datasymbol_c)
    ref_a_write1 = Reference(datasymbol_a)
    ref_b_write1 = Reference(datasymbol_b)
    ref_c_write1 = Reference(datasymbol_c)
    ref_a_write2 = Reference(datasymbol_a)
    ref_b_write2 = Reference(datasymbol_b)
    ref_c_write2 = Reference(datasymbol_c)
    ref_a_write3 = Reference(datasymbol_a)
    ref_b_write3 = Reference(datasymbol_b)
    ref_c_write3 = Reference(datasymbol_c)

    read = AccessType.READ
    write = AccessType.WRITE

    def build_routine():
        routine = Routine("test_routine")
        routine.symbol_table._argument_list.append(datasymbol_a)
        routine.symbol_table._argument_list.append(datasymbol_b)
        routine.symbol_table._argument_list.append(datasymbol_c)
        routine.symbol_table.add(datasymbol_a)
        routine.symbol_table.add(datasymbol_b)
        routine.symbol_table.add(datasymbol_c)
        return routine

    # OpenMP directives
    # a: shared (default), b: private, c: firstprivate
    routine = build_routine()
    directive = OMPParallelDirective.create([])
    private_clause = OMPPrivateClause.create([datasymbol_b])
    # ref_b_private = private_clause.children[0]
    firstprivate_clause = OMPFirstprivateClause.create([datasymbol_c])
    # ref_c_firstprivate = firstprivate_clause.children[0]
    directive.children[2] = private_clause
    directive.children[3] = firstprivate_clause

    # b = 0.0
    assignment0 = Assignment.create(ref_b_write1, Literal("0.0", REAL_TYPE))

    # a = b + c
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, ref_b_read1, ref_c_read1
    )
    assignment1 = Assignment.create(ref_a_write1, operation)

    # b = 3.0
    assignment2 = Assignment.create(ref_b_write2, Literal("3.0", REAL_TYPE))

    # c = 4.0
    assignment3 = Assignment.create(ref_c_write1, Literal("4.0", REAL_TYPE))

    directive.dir_body.addchild(assignment0)
    directive.dir_body.addchild(assignment1)
    directive.dir_body.addchild(assignment2)
    directive.dir_body.addchild(assignment3)

    # subroutine test_routine(a:inout, b:inout, c:inout)
    # !$omp parallel private(b) firstprivate(c)
    # a = b + c
    # b = 3.0
    # c = 4.0
    # !$omp end parallel
    routine.addchild(directive)

    dag = DataFlowDAG.create_from_schedule(routine)
    node_ref_a_write1 = dag.get_dag_node_for(ref_a_write1, write)
    node_ref_b_write1 = dag.get_dag_node_for(ref_b_write1, write)
    node_ref_b_write2 = dag.get_dag_node_for(ref_b_write2, write)
    node_ref_c_write1 = dag.get_dag_node_for(ref_c_write1, write)
    # node_ref_b_private = dag.get_dag_node_for(ref_b_private, write)
    # node_ref_c_firstprivate = dag.get_dag_node_for(ref_c_firstprivate, read)
    node_ref_b_read1 = dag.get_dag_node_for(ref_b_read1, read)
    node_ref_c_read1 = dag.get_dag_node_for(ref_c_read1, read)
    node_arg_a_in = dag.get_dag_node_for(datasymbol_a, write)
    node_arg_b_in = dag.get_dag_node_for(datasymbol_b, write)
    node_arg_c_in = dag.get_dag_node_for(datasymbol_c, write)
    node_arg_a_out = dag.get_dag_node_for(datasymbol_a, read)
    node_arg_b_out = dag.get_dag_node_for(datasymbol_b, read)
    node_arg_c_out = dag.get_dag_node_for(datasymbol_c, read)
    node_operation = dag.get_dag_node_for(operation, AccessType.UNKNOWN)

    # Test valid last_write_before
    assert dag.last_write_before(node_ref_b_write1) is node_arg_b_in
    assert dag.last_write_before(node_ref_a_write1) is node_arg_a_in
    assert node_ref_a_write1.backward_dependences == [node_operation]
    assert dag.last_write_before(node_ref_b_read1) is node_ref_b_write1
    assert node_ref_b_read1.backward_dependences == [node_ref_b_write1]
    assert dag.last_write_before(node_ref_c_read1) is node_arg_c_in
    assert node_ref_c_read1.backward_dependences == [node_arg_c_in]
    assert dag.last_write_before(node_arg_a_out) is node_ref_a_write1
    assert node_arg_a_out.backward_dependences == [node_ref_a_write1]
    assert dag.last_write_before(node_arg_b_out) is node_arg_b_in
    assert node_arg_b_out.backward_dependences == [node_arg_b_in]
    assert dag.last_write_before(node_arg_c_out) is node_arg_c_in
    assert node_arg_c_out.backward_dependences == [node_arg_c_in]

    # OpenACC directives
    # OpenMP directives
    # a: copy, b: copyout, c: copyin
    routine = build_routine()
    parallel_directive = ACCParallelDirective()
    data_directive = ACCDataDirective(
        parent=routine, children=[parallel_directive]
    )
    copy_clause = ACCCopyClause(children=[Reference(datasymbol_a)])
    copyout_clause = ACCCopyOutClause(children=[Reference(datasymbol_b)])
    copyin_clause = ACCCopyInClause(children=[Reference(datasymbol_c)])

    ref_a_read1 = Reference(datasymbol_a)
    ref_b_read1 = Reference(datasymbol_b)
    ref_c_read1 = Reference(datasymbol_c)
    ref_a_read2 = Reference(datasymbol_a)
    ref_b_read2 = Reference(datasymbol_b)
    ref_c_read2 = Reference(datasymbol_c)
    ref_a_read3 = Reference(datasymbol_a)
    ref_b_read3 = Reference(datasymbol_b)
    ref_c_read3 = Reference(datasymbol_c)
    ref_a_write1 = Reference(datasymbol_a)
    ref_b_write1 = Reference(datasymbol_b)
    ref_c_write1 = Reference(datasymbol_c)
    ref_a_write2 = Reference(datasymbol_a)
    ref_b_write2 = Reference(datasymbol_b)
    ref_c_write2 = Reference(datasymbol_c)
    ref_a_write3 = Reference(datasymbol_a)
    ref_b_write3 = Reference(datasymbol_b)
    ref_c_write3 = Reference(datasymbol_c)

    # b = 0.0
    assignment0 = Assignment.create(ref_b_write1, Literal("0.0", REAL_TYPE))

    # a = b + c
    operation = BinaryOperation.create(
        BinaryOperation.Operator.ADD, ref_b_read1, ref_c_read1
    )
    assignment1 = Assignment.create(ref_a_write1, operation)

    # b = 3.0
    assignment2 = Assignment.create(ref_b_write2, Literal("3.0", REAL_TYPE))

    # c = 4.0
    assignment3 = Assignment.create(ref_c_write1, Literal("4.0", REAL_TYPE))

    parallel_directive.dir_body.addchild(assignment0)
    parallel_directive.dir_body.addchild(assignment1)
    parallel_directive.dir_body.addchild(assignment2)
    parallel_directive.dir_body.addchild(assignment3)

    # FIXME: forcibly replacing _children to circumvent internals
    copy_clause = ACCCopyClause(children=[Reference(datasymbol_a)])
    copyout_clause = ACCCopyOutClause(children=[Reference(datasymbol_b)])
    copyin_clause = ACCCopyInClause(children=[Reference(datasymbol_c)])
    data_directive._children = [
        data_directive.children[0],
        copy_clause,
        copyout_clause,
        copyin_clause,
    ]

    # subroutine test_routine(a:inout, b:inout, c:inout)
    # !$acc data copy(a) copyout(b) copyin(c)
    # !$acc parallel
    # a = b + c
    # b = 3.0
    # c = 4.0
    # !$acc end parallel
    # !$acc end data
    routine.addchild(data_directive)

    dag = DataFlowDAG.create_from_schedule(routine)
    node_ref_a_write1 = dag.get_dag_node_for(ref_a_write1, write)
    node_ref_b_write1 = dag.get_dag_node_for(ref_b_write1, write)
    node_ref_b_write2 = dag.get_dag_node_for(ref_b_write2, write)
    node_ref_c_write1 = dag.get_dag_node_for(ref_c_write1, write)
    # node_ref_b_private = dag.get_dag_node_for(ref_b_private, write)
    # node_ref_c_firstprivate = dag.get_dag_node_for(ref_c_firstprivate, read)
    node_ref_b_read1 = dag.get_dag_node_for(ref_b_read1, read)
    node_ref_c_read1 = dag.get_dag_node_for(ref_c_read1, read)
    node_arg_a_in = dag.get_dag_node_for(datasymbol_a, write)
    node_arg_b_in = dag.get_dag_node_for(datasymbol_b, write)
    node_arg_c_in = dag.get_dag_node_for(datasymbol_c, write)
    node_arg_a_out = dag.get_dag_node_for(datasymbol_a, read)
    node_arg_b_out = dag.get_dag_node_for(datasymbol_b, read)
    node_arg_c_out = dag.get_dag_node_for(datasymbol_c, read)
    node_operation = dag.get_dag_node_for(operation, AccessType.UNKNOWN)

    # Test valid last_write_before
    assert dag.last_write_before(node_ref_b_write1) is node_arg_b_in
    assert dag.last_write_before(node_ref_a_write1) is node_arg_a_in
    assert node_ref_a_write1.backward_dependences == [node_operation]
    assert dag.last_write_before(node_ref_b_read1) is node_ref_b_write1
    assert node_ref_b_read1.backward_dependences == [node_ref_b_write1]
    assert dag.last_write_before(node_ref_c_read1) is node_arg_c_in
    assert node_ref_c_read1.backward_dependences == [node_arg_c_in]
    assert dag.last_write_before(node_arg_a_out) is node_ref_a_write1
    assert node_arg_a_out.backward_dependences == [node_ref_a_write1]
    assert dag.last_write_before(node_arg_b_out) is node_ref_b_write2
    assert node_arg_b_out.backward_dependences == [node_ref_b_write2]
    assert dag.last_write_before(node_arg_c_out) is node_arg_c_in
    assert node_arg_c_out.backward_dependences == [node_arg_c_in]
