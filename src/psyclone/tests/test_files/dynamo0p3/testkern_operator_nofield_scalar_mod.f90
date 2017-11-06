!-------------------------------------------------------------------------------
! BSD 3-Clause License
!
! Copyright (c) 2017, Science and Technology Facilities Council
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
! Authors: A. R. Porter and  R. W. Ford, STFC Daresbury Lab

module testkern_operator_nofield_scalar_mod
  use argument_mod
  use kernel_mod
  use constants_mod
  type, extends(kernel_type) :: testkern_operator_nofield_scalar_type
     type(arg_type), dimension(2) :: meta_args =    &
          (/ arg_type(gh_operator,gh_write,w2,w2),  &
             arg_type(gh_integer, gh_read)          &
          /)
     type(func_type) :: meta_funcs(1) =             &
          (/ func_type(w2, gh_basis)                &
          /)
     integer :: iterates_over = cells
     integer :: gh_shape = gh_quadrature_XYoZ
   contains
     procedure, nopass :: code => testkern_operator_code
  end type testkern_operator_nofield_scalar_type
contains
  subroutine testkern_operator_code(cell, nlayers, ncell_3d, local_stencil, &
       box_b, ndf_w2, basis_w2, np_xy, np_z, weights_xy, weights_z)
    implicit none
    integer :: cell, nlayers, ncell_3d, box_b, ndf_w2, np_xy, np_z
    real(kind=r_def), dimension(:,:,:) :: local_stencil
    real(kind=r_def), dimension(:,:,:,:) :: basis_w2
    real(kind=r_def), dimension(:) :: weights_xy, weights_z
  end subroutine testkern_operator_code
end module testkern_operator_nofield_scalar_mod
