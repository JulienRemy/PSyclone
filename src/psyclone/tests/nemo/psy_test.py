# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2017-2018, Science and Technology Facilities Council.
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
# Authors: R. W. Ford and A. R. Porter, STFC Daresbury Lab

''' Module containing py.test tests for the construction of a PSy
    representation of NEMO code '''


from __future__ import print_function, absolute_import
import os
import pytest
from fparser.common.readfortran import FortranStringReader
from fparser.two import Fortran2003
from fparser.two.parser import ParserFactory
import psyclone
from psyclone.parse import parse, ParseError
from psyclone.psyGen import PSyFactory, InternalError, GenerationError
from psyclone import nemo


# Constants
API = "nemo"
# Location of the Fortran files associated with these tests
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_files")
# Fortran parser
_PARSER = ParserFactory().create()

def test_explicit_do_sched():
    ''' Check that we generate a correct schedule for a triply-nested,
    explicit do loop '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "explicit_do.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    assert isinstance(psy, nemo.NemoPSy)
    invoke = psy.invokes.invoke_list[0]
    sched = invoke.schedule
    # The schedule should contain 3 loop objects
    loops = sched.walk(sched.children, nemo.NemoLoop)
    assert len(loops) == 3
    # The schedule should contain just 1 kernel
    assert isinstance(loops[2].children[0], nemo.NemoKern)


def test_implicit_loop_sched1():
    ''' Check that we get the correct schedule for an implicit loop '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "implicit_do.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    assert isinstance(psy, nemo.NemoPSy)
    print(len(psy.invokes.invoke_list))
    sched = psy.invokes.invoke_list[0].schedule
    sched.view()
    loops = sched.walk(sched.children, nemo.NemoLoop)
    assert len(loops) == 3
    kerns = sched.walk(sched.children, nemo.NemoKern)
    assert len(kerns) == 1


def test_implicit_loop_sched2():
    ''' Check that we get the correct schedule for an explicit loop over
    levels containing an implicit loop over the i-j slab '''
    ast, invoke_info = parse(os.path.join(BASE_PATH,
                                          "explicit_over_implicit.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    sched = psy.invokes.invoke_list[0].schedule
    sched.view()
    # We should have 3 loops (one from the explicit loop over levels and
    # the other two from the implicit loops over ji and jj).
    loops = sched.walk(sched.children, nemo.NemoLoop)
    assert len(loops) == 3
    kerns = sched.walk(sched.children, nemo.NemoKern)
    assert len(kerns) == 1


@pytest.mark.xfail(reason="Do not currently check for previous variable"
                   "declarations when adding loop variables")
def test_implicit_loop_assign():
    ''' Check that we only identify an implicit loop when array syntax
    is used as part of an assignment statement. '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "array_syntax.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    sched = psy.invokes.invoke_list[0].schedule
    loops = sched.walk(sched.children, nemo.NemoLoop)
    sched.view()
    gen = str(ast).lower()
    print(gen)
    # Our implicit loops gives us 5 explicit loops
    assert len(loops) == 5
    assert isinstance(sched.children[0], nemo.NemoLoop)
    # The other statements (that use array syntax) are not assignments
    # and therefore are not implicit loops
    assert isinstance(sched.children[1], nemo.NemoCodeBlock)
    # Check that the loop variables have been declared just once
    for var in ["psy_ji", "psy_jj", "psy_jk"]:
        assert gen.count("integer :: {0}".format(var)) == 1


def test_unrecognised_implicit():
    ''' Check that we raise the expected error if we encounter an
    unrecognised form of implicit loop. '''
    from psyclone.nemo import NemoImplicitLoop, NemoInvoke
    from fparser.two.utils import walk_ast
    reader = FortranStringReader("umask(:, :, :, :) = 0.0D0")
    assign = Fortran2003.Assignment_Stmt(reader)
    with pytest.raises(GenerationError) as err:
        NemoImplicitLoop(assign)
    assert ("Array section in unsupported dimension (4) for code "
            "'umask(:, :, :, :) = 0.0D0'" in str(err))
    # and now for the case where the Program unit doesn't have a
    # specification section to modify. This is hard to trigger
    # so we manually construct some objects and put them together
    # to create an artificial example...
    reader = FortranStringReader("umask(:, :, :) = 0.0D0")
    assign = Fortran2003.Assignment_Stmt(reader)
    reader = FortranStringReader("program atest\nreal :: umask(1,1,1,1)\n"
                                 "umask(:, :, :) = 0.0\nend program atest")
    prog = Fortran2003.Program_Unit(reader)
    invoke = NemoInvoke(prog, name="atest")
    loop = NemoImplicitLoop.__new__(NemoImplicitLoop)
    loop._parent = None
    loop.invoke = invoke
    loop.root.invoke._ast = prog
    spec = walk_ast(prog.content, [Fortran2003.Specification_Part])
    prog.content.remove(spec[0])
    with pytest.raises(InternalError) as err:
        loop.__init__(assign)
    assert "No specification part found for routine atest" in str(err)


def test_codeblock():
    ''' Check that we get the right schedule when the code contains
    some unrecognised statements as well as both an explict and an
    implicit loop. '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "code_block.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    sched = psy.invokes.invoke_list[0].schedule
    loops = sched.walk(sched.children, nemo.NemoLoop)
    assert len(loops) == 5
    cblocks = sched.walk(sched.children, nemo.NemoCodeBlock)
    assert len(cblocks) == 4
    kerns = sched.walk(sched.children, nemo.NemoKern)
    assert len(kerns) == 2
    # The last loop does not contain a kernel
    assert loops[-1].kernel is None


def test_io_not_kernel():
    ''' Check that we reject a kernel candidate if a loop body contains
    a write/read statement '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "io_in_loop.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    sched = psy.invokes.invoke_list[0].schedule
    # We should have only 1 actual kernel and 2 code blocks
    cblocks = sched.walk(sched.children, nemo.NemoCodeBlock)
    assert len(cblocks) == 2
    kerns = sched.walk(sched.children, nemo.NemoKern)
    assert len(kerns) == 1


def test_schedule_view(capsys):
    ''' Check the schedule view/str methods work as expected '''
    from psyclone.psyGen import colored
    from psyclone.nemo import NEMO_SCHEDULE_COLOUR_MAP
    ast, invoke_info = parse(os.path.join(BASE_PATH, "io_in_loop.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    sched = psy.invokes.invoke_list[0].schedule
    sched_str = str(sched)
    assert "CodeBlock[2 statements]" in sched_str
    assert "NemoLoop[levels]: jk=1,jpk,1" in sched_str
    assert "NemoLoop[lat]: jj=1,jpj,1" in sched_str
    assert "NemoLoop[lon]: ji=1,jpi,1" in sched_str
    sched.view()
    output, _ = capsys.readouterr()

    # Have to allow for colouring of output text
    loop_str = colored("Loop", NEMO_SCHEDULE_COLOUR_MAP["Loop"])
    cb_str = colored("NemoCodeBlock", NEMO_SCHEDULE_COLOUR_MAP["CodeBlock"])
    kern_str = colored("KernCall", NEMO_SCHEDULE_COLOUR_MAP["KernCall"])
    sched_str = colored("Schedule", NEMO_SCHEDULE_COLOUR_MAP["Schedule"])

    expected_sched = (
        sched_str + "[]\n"
        "    " + loop_str + "[type='levels',field_space='None',"
        "it_space='None']\n"
        "        " + loop_str + "[type='lat',field_space='None',"
        "it_space='None']\n"
        "            " + loop_str + "[type='lon',field_space='None',"
        "it_space='None']\n"
        "                " + kern_str + "[]\n"
        "    " + loop_str + "[type='levels',field_space='None',"
        "it_space='None']\n"
        "        " + loop_str + "[type='lat',field_space='None',"
        "it_space='None']\n"
        "            " + loop_str + "[type='lon',field_space='None',"
        "it_space='None']\n"
        "                " + cb_str + "[<class 'fparser.two.Fortran2003."
        "Assignment_Stmt'>]\n"
        "    " + loop_str + "[type='levels',field_space='None',"
        "it_space='None']\n"
        "        " + loop_str + "[type='lat',field_space='None',"
        "it_space='None']\n"
        "            " + loop_str + "[type='lon',field_space='None',"
        "it_space='None']\n"
        "                " + cb_str + "[<class 'fparser.two.Fortran2003."
        "Assignment_Stmt'>]")
    assert expected_sched in output


def test_kern_inside_if():
    ''' Check that we identify kernels when they are within an if block. '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "imperfect_nest.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    sched = psy.invokes.invoke_list[0].schedule
    kerns = sched.walk(sched.children, nemo.NemoKern)
    assert len(kerns) == 6
    ifblock = sched.children[0].children[1]
    assert isinstance(ifblock, nemo.NemoIfBlock)
    assert str(ifblock) == "If-block: jk == 1"
    assert isinstance(ifblock.children[1], nemo.NemoIfClause)
    assert isinstance(ifblock.children[2], nemo.NemoIfClause)


def test_invalid_if_clause():
    ''' Check that we raise the expected error if the NemoIfClause
    is passed something that isn't an if-clause. '''
    from psyclone.nemo import NemoIfClause
    reader = FortranStringReader("umask(:, :, :, :) = 0")
    assign = Fortran2003.Assignment_Stmt(reader)
    with pytest.raises(InternalError) as err:
        _ = NemoIfClause([assign])
    assert "Unrecognised member of if block: " in str(err)


def test_kern_load_errors(monkeypatch):
    ''' Check that the various load methods of the NemoKern class raise
    the expected errors. '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "explicit_do.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    invoke = psy.invokes.invoke_list[0]
    sched = invoke.schedule
    # The schedule should contain 3 loop objects
    kerns = sched.walk(sched.children, nemo.NemoKern)
    with pytest.raises(InternalError) as err:
        kerns[0].load("Not an fparser2 AST node")
    assert ("internal error: Expecting either Block_Nonlabel_Do_Construct "
            "or Assignment_Stmt but got " in str(err))
    # TODO why hasn't the Kernel or Loop objects got a valid _ast?
    loop = sched.children[0].children[0].children[0]._ast
    monkeypatch.setattr(loop, "content", ["not_a_loop"])
    with pytest.raises(InternalError) as err:
        kerns[0]._load_from_loop(loop)
    assert ("Expecting Nonlabel_Do_Stmt as first child of "
            "Block_Nonlabel_Do_Construct but got" in str(err))


def test_no_inline():
    ''' Check that calling the NemoPSy.inline() method raises the expected
    error (since we haven't implemented it yet). '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "imperfect_nest.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    with pytest.raises(NotImplementedError) as err:
        psy.inline(None)
    assert ("The NemoPSy.inline method has not yet been implemented!"
            in str(err))


def test_empty_routine():
    ''' Check that we handle the case where a program unit does not
    contain any executable statements. '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "empty_routine.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    assert len(psy.invokes.invoke_list) == 1
    assert psy.invokes.invoke_list[0].schedule is None
    # Calling update() on this Invoke should do nothing
    psy.invokes.invoke_list[0].update()


def test_invoke_function():
    ''' Check that we successfully construct an Invoke if the program
    unit is a function. '''
    ast, invoke_info = parse(os.path.join(BASE_PATH, "afunction.f90"),
                             api=API, line_length=False)
    psy = PSyFactory(API, distributed_memory=False).create(invoke_info)
    assert len(psy.invokes.invoke_list) == 1
    invoke = psy.invokes.invoke_list[0]
    assert invoke.name == "afunction"
    assert isinstance(invoke.schedule.children[0], nemo.NemoCodeBlock)
