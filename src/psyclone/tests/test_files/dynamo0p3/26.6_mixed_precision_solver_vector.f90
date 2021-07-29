!-------------------------------------------------------------------------------
! BSD 3-Clause License
!
! Copyright (c) 2021, Science and Technology Facilities Council
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! * Redistributions of source code must retain the above copyright notice, this
!   list of conditions and the following disclaimer.
!
! * Redistributions in binary form must reproduce the above copyright notice,
!   this list of conditions and the following disclaimer in the documentation
!   and/or other materials provided with the distribution.
!
! * Neither the name of the copyright holder nor the names of its
!   contributors may be used to endorse or promote products derived from
!   this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
! -----------------------------------------------------------------------------
! Author: R. W. Ford STFC Daresbury Lab
!
! Example where the field is dereferenced from an abstract_vector_type
! and therefore has no type information. The LFRic rules state that an
! abstract_vector_type can only be of type field_vector_type in LFRic
! code and all fields within a field_vector_type are of type
! field_type. This is checked at runtime by the algorithm code using a
! select statement.

module vector_type

  use constants_mod,    only : r_def
  use vector_mod,       only : abstract_vector_type
  use field_vector_mod, only : field_vector_type
  use field_mod,        only : field_type
  use testkern_mod,     only : testkern_type

contains

  subroutine my_sub(x, y, m1, m2)
    class (abstract_vector_type), intent(inout) :: x,y
    type(field_type), intent(inout) :: m1, m2
    real(r_def) :: a
    select type (x)
    type is (field_vector_type)
      call invoke(testkern_type(a, x%vector(1), y%vector(1), m1, m2))
    class default
      print *,"Error"
    end select
  end subroutine my_sub

end module vector_type
