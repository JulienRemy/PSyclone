# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2022, Science and Technology Facilities Council.
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
# Modified I. Kavcic and A. Coughtrie, Met Office
#          C.M. Maynard, Met Office / University of Reading
# Modified J. Henrichs, Bureau of Meteorology
# Modified A. B. G. Chalk, STFC Daresbury Lab

''' Tests of the KernelModuleInlineTrans PSyIR transformation. '''

import pytest
from psyclone.configuration import Config
from psyclone.psyGen import CodedKern, Kern
from psyclone.psyir.nodes import Container, Routine
from psyclone.psyir.symbols import DataSymbol, RoutineSymbol, REAL_TYPE, \
    SymbolError
from psyclone.psyir.transformations import TransformationError
from psyclone.tests.gocean_build import GOceanBuild
from psyclone.tests.utilities import count_lines, get_invoke
from psyclone.domain.common.transformations import KernelModuleInlineTrans


def test_module_inline_constructor_and_str():
    ''' Test that the transformation can be created and stringified. '''
    inline_trans = KernelModuleInlineTrans()
    assert str(inline_trans) == \
        "Inline a kernel subroutine into the PSy module"


def test_validate_inline_error_if_not_kernel():
    ''' Test that the inline transformation fails if the object being
    passed is not a kernel'''
    _, invoke = get_invoke("single_invoke_three_kernels.f90", "gocean1.0",
                           idx=0, dist_mem=False)
    schedule = invoke.schedule
    kern_call = schedule.children[0].loop_body[0]
    inline_trans = KernelModuleInlineTrans()
    with pytest.raises(TransformationError) as err:
        inline_trans.apply(kern_call)
    assert ("Target of a KernelModuleInline must be a sub-class of "
            "psyGen.CodedKern but got 'GOLoop'" in str(err.value))


def test_validate_invalid_get_kernel_schedule(monkeypatch):
    '''Check that the validate method in the class KernelTrans raises an
    exception if the kernel code can not be retrieved.

    '''
    kernel_trans = KernelModuleInlineTrans()
    _, invoke = get_invoke("single_invoke_kern_with_global.f90",
                           api="gocean1.0", idx=0)
    sched = invoke.schedule
    kernels = sched.walk(Kern)
    kernel = kernels[0]

    def raise_symbol_error():
        '''Simple function that raises SymbolError.'''
        raise SymbolError("error")
    monkeypatch.setattr(kernel, "get_kernel_schedule", raise_symbol_error)
    with pytest.raises(TransformationError) as err:
        kernel_trans.apply(kernel)
    assert ("KernelModuleInline failed to retrieve PSyIR for kernel "
            "'kernel_with_global_code' using the 'get_kernel_schedule' "
            "method." in str(err.value))


def test_validate_no_inline_global_var():
    ''' Check that we refuse to in-line a kernel that accesses a global
    variable. '''
    inline_trans = KernelModuleInlineTrans()
    _, invoke = get_invoke("single_invoke_kern_with_global.f90",
                           api="gocean1.0", idx=0)
    sched = invoke.schedule
    kernels = sched.walk(Kern)
    with pytest.raises(TransformationError) as err:
        inline_trans.apply(kernels[0])
    assert ("'kernel_with_global_code' contains accesses to data (variable "
            "'alpha') that are not present in the Symbol Table(s) "
            "within KernelSchedule scope." in str(err.value))


def test_validate_name_clashes():
    ''' Test that if the module-inline transformation finds the kernel name
    already used in the Container scope, it ...'''
    # Use LFRic example with a repeated CodedKern
    psy, _ = get_invoke("4.6_multikernel_invokes.f90", "dynamo0.3", idx=0,
                        dist_mem=False)
    schedule = psy.invokes.invoke_list[0].schedule
    coded_kern = schedule.children[0].loop_body[0]
    inline_trans = KernelModuleInlineTrans()

    # Check that name clashes which are not subroutines are detected
    schedule.symbol_table.add(DataSymbol("ru_code", REAL_TYPE))
    with pytest.raises(TransformationError) as err:
        inline_trans.apply(coded_kern)
    assert ("Cannot module-inline subroutine 'ru_code' because symbol "
            "'ru_code: DataSymbol<Scalar<REAL, UNDEFINED>, Local>' with the "
            "same name already exists and changing the name of module-inlined "
            "subroutines is not supported yet." in str(err.value))

    # TODO # 898. Manually force removal of previous imported symbol
    # symbol_table.remove() is not implemented yet.
    schedule.symbol_table._symbols.pop("ru_code")

    # Check that if a subroutine with the same name already exists and it is
    # not identical, it fails.
    new_symbol = RoutineSymbol("ru_code")
    schedule.parent.symbol_table.add(new_symbol)
    schedule.parent.addchild(Routine(new_symbol.name))
    with pytest.raises(TransformationError) as err:
        inline_trans.apply(coded_kern)
    assert ("Cannot inline subroutine 'ru_code' because another, different, "
            "subroutine with the same name already exists and versioning of "
            "module-inlined subroutines is not implemented "
            "yet.") in str(err.value)


def test_module_inline_apply_transformation(tmpdir, fortran_writer):
    ''' Test that we can succesfully inline a basic kernel subroutine
    routine into the PSy layer module using a transformation '''
    psy, invoke = get_invoke("single_invoke_three_kernels.f90", "gocean1.0",
                             idx=0, dist_mem=False)
    schedule = invoke.schedule

    # Apply the inline transformation
    kern_call = schedule.children[1].loop_body[0].loop_body[0]
    inline_trans = KernelModuleInlineTrans()
    inline_trans.apply(kern_call)

    # The new inlined routine must now exist
    assert kern_call.ancestor(Container).symbol_table.lookup("compute_cv_code")
    assert kern_call.ancestor(Container).children[1].name == "compute_cv_code"

    # We should see it in the output of both:
    # - the backend
    code = fortran_writer(schedule.root)
    assert 'subroutine compute_cv_code(i, j, cv, p, v)' in code

    # - the gen_code
    gen = str(psy.gen)
    assert 'SUBROUTINE compute_cv_code(i, j, cv, p, v)' in gen

    # And the import has been remove from both
    # check that the associated use no longer exists
    assert 'use compute_cv_mod, only: compute_cv_code' not in code
    assert 'USE compute_cv_mod, ONLY: compute_cv_code' not in gen

    # Do the gen_code check again because repeating the call resets some
    # aspects and we need to see if the second call still works as expected
    gen = str(psy.gen)
    assert 'SUBROUTINE compute_cv_code(i, j, cv, p, v)' in gen
    assert 'USE compute_cv_mod, ONLY: compute_cv_code' not in gen
    assert gen.count("SUBROUTINE compute_cv_code(") == 1

    # And it is valid code
    assert GOceanBuild(tmpdir).code_compiles(psy)


def test_module_inline_apply_kernel_in_multiple_invokes():
    ''' Check that module-inline works as expected when the same kernel
    is provided in different invokes'''
    # Use LFRic example with the kernel 'testkern_qr' repeated once in
    # the first invoke and 3 times in the second invoke.
    psy, _ = get_invoke("3.1_multi_functions_multi_invokes.f90", "dynamo0.3",
                        idx=0, dist_mem=False)

    # By default the kernel is imported once per invoke
    gen = str(psy.gen)
    assert gen.count("USE testkern_qr, ONLY: testkern_qr_code") == 2
    assert gen.count("END SUBROUTINE testkern_qr_code") == 0

    # Module inline kernel in invoke 1
    inline_trans = KernelModuleInlineTrans()
    schedule1 = psy.invokes.invoke_list[0].schedule
    for coded_kern in schedule1.walk(CodedKern):
        if coded_kern.name == "testkern_qr_code":
            inline_trans.apply(coded_kern)
    gen = str(psy.gen)

    # After this, one invoke uses the inlined top-level subroutine
    # and the other imports it (shadowing the top-level symbol)
    assert gen.count("USE testkern_qr, ONLY: testkern_qr_code") == 1
    assert gen.count("END SUBROUTINE testkern_qr_code") == 1

    # Module inline kernel in invoke 2
    schedule1 = psy.invokes.invoke_list[1].schedule
    for coded_kern in schedule1.walk(CodedKern):
        if coded_kern.name == "testkern_qr_code":
            inline_trans.apply(coded_kern)
    gen = str(psy.gen)
    # After this, no imports are remaining and both use the same
    # top-level implementation
    assert gen.count("USE testkern_qr, ONLY: testkern_qr_code") == 0
    assert gen.count("END SUBROUTINE testkern_qr_code") == 1


def test_module_inline_apply_with_sub_use(tmpdir):
    ''' Test that we can module inline a kernel subroutine which
    contains a use statement'''
    psy, invoke = get_invoke("single_invoke_scalar_int_arg.f90", "gocean1.0",
                             idx=0, dist_mem=False)
    schedule = invoke.schedule
    kern_call = schedule.children[0].loop_body[0].loop_body[0]
    inline_trans = KernelModuleInlineTrans()
    inline_trans.apply(kern_call)
    gen = str(psy.gen)
    # check that the subroutine has been inlined
    assert 'SUBROUTINE bc_ssh_code(ji, jj, istep, ssha, tmask)' in gen
    # check that the use within the subroutine exists
    assert 'USE grid_mod' in gen
    # check that the associated psy use does not exist
    assert 'USE bc_ssh_mod, ONLY: bc_ssh_code' not in gen
    assert GOceanBuild(tmpdir).code_compiles(psy)


def test_module_inline_apply_same_kernel(tmpdir):
    '''Tests that correct results are obtained when an invoke that uses
    the same kernel subroutine more than once has that kernel
    inlined'''
    psy, invoke = get_invoke("test14_module_inline_same_kernel.f90",
                             "gocean1.0", idx=0)
    schedule = invoke.schedule
    kern_call = schedule.coded_kernels()[0]
    inline_trans = KernelModuleInlineTrans()
    inline_trans.apply(kern_call)
    gen = str(psy.gen)
    # check that the subroutine has been inlined
    assert 'SUBROUTINE compute_cu_code(' in gen
    # check that the associated psy "use" does not exist
    assert 'USE compute_cu_mod, ONLY: compute_cu_code' not in gen
    # check that the subroutine has only been inlined once
    count = count_lines(gen, "SUBROUTINE compute_cu_code(")
    assert count == 1, "Expecting subroutine to be inlined once"
    assert GOceanBuild(tmpdir).code_compiles(psy)


def test_module_inline_dynamo(monkeypatch, annexed, dist_mem):
    '''Tests that correct results are obtained when a kernel is inlined
    into the psy-layer in the dynamo0.3 API. All previous tests use GOcean
    for testing.

    We also test when annexed is False and True as it affects how many halo
    exchanges are generated.

    '''
    config = Config.get()
    dyn_config = config.api_conf("dynamo0.3")
    monkeypatch.setattr(dyn_config, "_compute_annexed_dofs", annexed)
    psy, invoke = get_invoke("4.6_multikernel_invokes.f90", "dynamo0.3",
                             name="invoke_0", dist_mem=dist_mem)
    schedule = invoke.schedule
    if dist_mem:
        if annexed:
            index = 6
        else:
            index = 8
    else:
        index = 1
    kern_call = schedule.children[index].loop_body[0]
    inline_trans = KernelModuleInlineTrans()
    inline_trans.apply(kern_call)
    gen = str(psy.gen)
    # check that the subroutine has been inlined
    assert 'SUBROUTINE ru_code(' in gen
    # check that the associated psy "use" does not exist
    assert 'USE ru_kernel_mod, only : ru_code' not in gen
