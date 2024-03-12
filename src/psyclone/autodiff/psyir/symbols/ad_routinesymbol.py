from enum import Enum

from psyclone.psyir.symbols import RoutineSymbol
from psyclone.autodiff.psyir import ADPSyIR, ADMotion


class ADRoutineSymbol(RoutineSymbol, ADPSyIR):
    _tangent_prefix = ""
    _tangent_postfix = "_tangent"

    _recording_prefix = ""
    _recording_postfix = "_rec"

    _returning_prefix = ""
    _returning_postfix = "_ret"

    _reversing_prefix = ""
    _reversing_postfix = "_rev"

    _psyir_to_ad = dict()

    def __init__(
        self, name, datatype=None, motion=ADMotion.ADVANCING, **kwargs
    ):
        if (not isinstance(motion, Enum)) or (motion not in ADMotion):
            raise TypeError("")
        super().__init__(name, datatype, **kwargs)
        self._motion = motion
        self._calls = []

        if motion is ADMotion.ADVANCING:
            self._advancing_routine_symbol = self
            self._tangent_routine_symbol = self.create_tangent_routine_symbol()
            self._recording_routine_symbol = (
                self.create_recording_routine_symbol()
            )
            self._returning_routine_symbol = (
                self.create_returning_routine_symbol()
            )
            self._reversing_routine_symbol = (
                self.create_reversing_routine_symbol()
            )

        self._routine = None

    # NOTE: this either creates a new ADDataSymbol or returns the existing one
    @classmethod
    def from_psyir(cls, routine_symbol):
        if not isinstance(routine_symbol, RoutineSymbol):
            raise TypeError("")
        if isinstance(routine_symbol, ADRoutineSymbol):
            raise TypeError("")

        if routine_symbol in cls._psyir_to_ad:
            return cls._psyir_to_ad[routine_symbol]
        else:
            ad_routine_symbol = cls(
                name=routine_symbol.name, motion=ADMotion.ADVANCING
            )
            cls._psyir_to_ad[routine_symbol] = ad_routine_symbol
            return ad_routine_symbol

        # TODO: jacobian forward, jacobian reverse, etc

    def create_tangent_routine_symbol(self):
        name = self._tangent_prefix + self.name + self._tangent_postfix
        routine_symbol = ADRoutineSymbol(name, motion=ADMotion.TANGENT)
        routine_symbol._advancing_routine_symbol = self
        return routine_symbol

    def create_recording_routine_symbol(self):
        name = self._recording_prefix + self.name + self._recording_postfix
        routine_symbol = ADRoutineSymbol(name, motion=ADMotion.RECORDING)
        routine_symbol._advancing_routine_symbol = self
        return routine_symbol

    def create_returning_routine_symbol(self):
        name = self._returning_prefix + self.name + self._returning_postfix
        routine_symbol = ADRoutineSymbol(name, motion=ADMotion.RETURNING)
        routine_symbol._advancing_routine_symbol = self
        return routine_symbol

    def create_reversing_routine_symbol(self):
        name = self._reversing_prefix + self.name + self._reversing_postfix
        routine_symbol = ADRoutineSymbol(name, motion=ADMotion.REVERSING)
        routine_symbol._advancing_routine_symbol = self
        return routine_symbol

    @property
    def motion(self):
        return self._motion

    @property
    def calls(self):
        return self._calls

    # Is initially None since RoutineSymbol is created before Routine?
    @property
    def routine(self):
        return self._routine

    @routine.setter
    def routine(self, routine):
        from psyclone.autodiff.psyir.nodes import ADRoutine

        if not isinstance(routine, ADRoutine):
            raise TypeError("")
        if routine.motion is not self.motion:
            raise ValueError("")
        if routine.routine_symbol != self:
            raise ValueError("")

        self._routine = routine

    @property
    def advancing_routine_symbol(self):
        return self._advancing_routine_symbol

    @advancing_routine_symbol.setter
    def advancing_routine_symbol(self, advancing_routine_symbol):
        if not isinstance(advancing_routine_symbol, ADRoutineSymbol):
            raise TypeError("")
        if advancing_routine_symbol.motion is not ADMotion.ADVANCING:
            raise ValueError("")
        self._advancing_routine_symbol = advancing_routine_symbol

    @property
    def tangent_routine_symbol(self):
        if self.motion is ADMotion.ADVANCING:
            return self._tangent_routine_symbol

        return self.advancing_routine_symbol.tangent_routine_symbol

    @property
    def recording_routine_symbol(self):
        if self.motion is ADMotion.ADVANCING:
            return self._recording_routine_symbol

        return self.advancing_routine_symbol.recording_routine_symbol

    @property
    def returning_routine_symbol(self):
        if self.motion is ADMotion.ADVANCING:
            return self._returning_routine_symbol

        return self.advancing_routine_symbol.returning_routine_symbol

    @property
    def reversing_routine_symbol(self):
        if self.motion is ADMotion.ADVANCING:
            return self._reversing_routine_symbol

        return self.advancing_routine_symbol.reversing_routine_symbol
