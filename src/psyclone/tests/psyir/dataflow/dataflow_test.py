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
    assert node3._backward_dependences == [node1]

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
    assert node3._backward_dependences == [node1]
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


def test_data_flow_dag_last_write_to():
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
    node4 = dag.get_dag_node_for(reference2_copy, write)
    node5 = dag.get_dag_node_for(datasymbol, write)
    node6 = dag.get_dag_node_for(datasymbol2, read)

    # Test valid last_write_to
    assert dag.last_write_to(reference) is node5
    assert dag.last_write_to(reference2) is node4

    #####
    # Array case
    datasymbol = DataSymbol(
        "a",
        ArrayType(REAL_TYPE, shape=[10]),
        interface=ArgumentInterface(access=ArgumentInterface.Access.READ),
    )
    datasymbol2 = DataSymbol(
        "b",
        ArrayType(REAL_TYPE, shape=[10]),
        interface=ArgumentInterface(access=ArgumentInterface.Access.WRITE),
    )

    reference = Reference(datasymbol)
    reference2 = Reference(datasymbol2)

    array_reference_a1 = ArrayReference.create(
        datasymbol, [Literal("1", INTEGER_TYPE)]
    )
    array_reference_a2 = ArrayReference.create(
        datasymbol, [Literal("2", INTEGER_TYPE)]
    )

    array_reference_b1 = ArrayReference.create(
        datasymbol2, [Literal("1", INTEGER_TYPE)]
    )
    array_reference_b2 = ArrayReference.create(
        datasymbol2, [Literal("2", INTEGER_TYPE)]
    )

    # TODO: Add test for array case, within loop, outside, etc.
    # TODO Add tests for directives

    # Test invalid last_write_to
    with pytest.raises(TypeError):
        dag.last_write_to(False)
    with pytest.raises(TypeError):
        dag.last_write_to(1)
    with pytest.raises(TypeError):
        dag.last_write_to(None)
