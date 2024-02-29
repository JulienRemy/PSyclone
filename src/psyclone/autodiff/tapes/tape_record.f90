subroutine record_real(array, index, value)
    implicit none
    real, intent(inout), allocatable :: array(:)
    real, allocatable :: tmp(:)
    integer, intent(in) :: index
    real, intent(in) :: value

    if (index <= size(array)) then
        array(index) = value
    else
        allocate(tmp(2*size(array)))
        tmp(:size(array)) = array(:)
        call move_alloc(tmp, array)
        array(index) = value
    end if
end subroutine record_real

subroutine record_real_array(array, start_index, stop_index, values)
    implicit none
    real, intent(inout), allocatable :: array(:)
    real, allocatable :: tmp(:)
    integer, intent(in) :: start_index, stop_index
    real, intent(in) :: values(stop_index - start_index + 1)

    if (stop_index <= size(array)) then
        array(start_index : stop_index) = values(:)
    else
        allocate(tmp(2*size(array)))
        tmp(:size(array)) = array(:)
        call move_alloc(tmp, array)
        array(start_index : stop_index) = values(:)
    end if
end subroutine record_real_array

subroutine record_logical(array, index, value)
    implicit none
    logical, intent(inout), allocatable :: array(:)
    logical, allocatable :: tmp(:)
    integer, intent(in) :: index
    logical, intent(in) :: value

    if (index <= size(array)) then
        array(index) = value
    else
        allocate(tmp(2*size(array)))
        tmp(:size(array)) = array(:)
        call move_alloc(tmp, array)
        array(index) = value
    end if
end subroutine record_logical

subroutine record_logical_array(array, start_index, stop_index, values)
    implicit none
    logical, intent(inout), allocatable :: array(:)
    logical, allocatable :: tmp(:)
    integer, intent(in) :: start_index, stop_index
    logical, intent(in) :: values(stop_index - start_index + 1)

    if (stop_index <= size(array)) then
        array(start_index : stop_index) = values(:)
    else
        allocate(tmp(2*size(array)))
        tmp(:size(array)) = array(:)
        call move_alloc(tmp, array)
        array(start_index : stop_index) = values(:)
    end if
end subroutine record_logical_array
