from abc import ABCMeta, abstractproperty
from itertools import count
from types import NoneType

from psyclone.psyir.transformations import TransformationError
from psyclone.psyir.symbols import (
    DataSymbol,
    ArgumentInterface,
    ScalarType,
    ArrayType,
)
from psyclone.autodiff.psyir import ADPSyIR


class ADDataSymbol(DataSymbol, ADPSyIR, metaclass=ABCMeta):

    @abstractproperty
    def is_variable(self):
        pass

    @abstractproperty
    def is_derivative(self):
        pass

    @abstractproperty
    def is_adjoint(self):
        pass

    @abstractproperty
    def is_operation_adjoint(self):
        pass


class ADVariableSymbol(ADDataSymbol):

    _psyir_to_ad = dict()

    def __init__(
        self,
        name,
        datatype,
        is_loop_variable=False,
        is_constant=False,
        initial_value=None,
        **kwargs,
    ):
        super().__init__(name, datatype, is_constant, initial_value, **kwargs)
        if (
            datatype is ScalarType
            and datatype.intrinsic is ScalarType.Intrinsic.REAL
        ) or (
            datatype is ArrayType
            and datatype.datatype.intrinsic is ScalarType.Intrinsic.REAL
        ):
            self._derivative_symbol = self.create_derivative_symbol()
            self._adjoint_symbol = self.create_adjoint_symbol()
        else:
            self._derivative_symbol = None
            self._adjoint_symbol = None

        if not isinstance(is_loop_variable, bool):
            raise TypeError("")

        self._version_0_is_loop_variable = is_loop_variable

        if self.version_0_is_argument_value or self.version_0_is_loop_variable:
            self._references_per_versions = [[]]
        else:
            self._references_per_versions = []

    @property
    def version_0_is_argument_value(self):
        return self.is_argument and (
            self.interface.access
            in (
                ArgumentInterface.Access.READ,
                ArgumentInterface.Access.READWRITE,
            )
        )

    @property
    def last_version_is_out_value(self):
        return self.is_argument and (
            self.interface.access
            in (
                ArgumentInterface.Access.WRITE,
                ArgumentInterface.Access.READWRITE,
            )
        )

    @property
    def version_0_is_loop_variable(self):
        return self._version_0_is_loop_variable

    @property
    def version_0_is_a_symbol(self):
        if self.version_0_is_argument_value:
            print(f"{self.name} is an arg")
        if self.version_0_is_loop_variable:
            print(f"{self.name} is a loop var")
        return (
            self.version_0_is_argument_value or self.version_0_is_loop_variable
        )

    @property
    def references_per_versions(self):
        return self._references_per_versions

    @property
    def versions(self):
        return len(self.references_per_versions)

    @property
    def references(self, version=None):
        if not isinstance(version, (int, slice, NoneType)):
            raise TypeError(
                f"'version' argument should be of type "
                f"'int', 'slice' or 'NoneType' but found "
                f"'{type(version).__name__}'."
            )

        # Get all references for all versions
        if version is None:
            references = []
            for refs in self.references_per_versions:
                references.extend(refs)
            return references

        # References of a specific version
        if isinstance(version, int):
            return self.references_per_versions[version]

        # References for all versions in the slice
        references = []
        for refs in self.references_per_versions[version]:
            references.extend(refs)
        return references

    def log_reference(self, reference):
        from psyclone.autodiff.psyir.nodes import ADReference

        if not isinstance(reference, ADReference):
            raise TypeError(
                f"'reference' argument should be of type "
                f"'ADReference' but found "
                f"'{type(reference).__name__}'."
            )
        if reference.symbol != self:
            raise ValueError(
                f"'reference' argument should have symbol "
                f"'{self.name}' but found '{reference.symbol.name}'."
            )

        version = reference.version

        if version > self.versions:
            raise ValueError(
                f"'reference' argument has version '{version}' "
                f"but this symbol last version is "
                f"'{self.versions - 1}'."
            )

        if version == self.versions:
            # if not (version == 0 and self.version_0_is_a_symbol) and (
            #    reference.access is not ADReference.Access.WRITE
            # ):
            #    raise ValueError(
            #        "First reference of a new version should have "
            #        "WRITE access attribute."
            #        f"{reference.name, version}"
            #    )
            self.references_per_versions.append([reference])
        else:
            # if reference.access is not ADReference.Access.READ:
            #    raise ValueError(
            #        "New reference of an existing version should "
            #        "have READ access attribute."
            #    )
            self.references_per_versions[version].append(reference)

        # debug = []
        # for ver in self.references_per_versions:
        #     if ver == []:
        #         debug.append([])
        #     else:
        #         l = []
        #         for ref in ver:
        #             l.append((ref.name, ref.version, ref.access))
        #         debug.append(l)
        # print(debug)

    def create_last_version_reference(self):
        from psyclone.autodiff.psyir.nodes import ADReference

        if self.versions == 0:
            version = 0
        else:
            version = self.versions - 1
        return ADReference(self, version, ADReference.Access.READ)
        # self.log_reference(reference)

    def create_new_version_reference(self):
        from psyclone.autodiff.psyir.nodes import ADReference

        version = self.versions
        return ADReference(self, version, ADReference.Access.WRITE)
        # self.log_reference(reference)

    # NOTE: this either creates a new ADVariableSymbol or returns the existing one
    @classmethod
    def from_psyir(cls, datasymbol, is_loop_variable=False):
        if not isinstance(datasymbol, DataSymbol):
            raise TypeError("")
        if isinstance(datasymbol, ADVariableSymbol):
            raise TypeError("")
        if not isinstance(is_loop_variable, bool):
            raise TypeError("")

        if datasymbol in cls._psyir_to_ad:
            return cls._psyir_to_ad[datasymbol]
        else:
            ad_datasymbol = cls(
                name=datasymbol.name,
                datatype=datasymbol.datatype,
                is_loop_variable=is_loop_variable,
                is_constant=datasymbol.is_constant,
                initial_value=datasymbol.initial_value,
                visibility=datasymbol.visibility,
                interface=datasymbol.interface,
            )
            cls._psyir_to_ad[datasymbol] = ad_datasymbol
            return ad_datasymbol

    @property
    def derivative_symbol(self):
        return self._derivative_symbol

    @property
    def adjoint_symbol(self):
        return self._adjoint_symbol

    def is_variable(self):
        return True

    def is_derivative(self):
        return False

    def is_adjoint(self):
        return False

    def is_operation_adjoint(self):
        return False

    def create_derivative_symbol(self):
        if self.derivative_symbol is not None:
            raise TransformationError(
                f"ADVariableSymbol '{self.name}' already "
                f"has a derivative symbol "
                f"'{self.derivative_symbol.name}'."
            )
        derivative_symbol = ADDerivativeSymbol(self)
        # self._derivative_symbol = derivative_symbol
        return derivative_symbol

    def create_adjoint_symbol(self):
        if self.adjoint_symbol is not None:
            raise TransformationError(
                f"ADVariableSymbol '{self.name}' already "
                f"has an adjoint symbol "
                f"'{self.adjoint_symbol.name}'."
            )
        adjoint_symbol = ADAdjointSymbol(self)
        # self._adjoint_symbol = adjoint_symbol
        return adjoint_symbol


class ADDerivativeSymbol(ADDataSymbol):
    _derivative_prefix = ""
    _derivative_postfix = "_d"

    def __init__(self, variable_symbol):
        if not isinstance(variable_symbol, ADVariableSymbol):
            raise TypeError("")
        name = (
            self._derivative_prefix
            + variable_symbol.name
            + self._derivative_postfix
        )
        datatype = variable_symbol.datatype
        super().__init__(name, datatype)
        self._variable_symbol = variable_symbol

    # Should never be None
    @property
    def variable_symbol(self):
        return self._variable_symbol

    @property
    def adjoint_symbol(self):
        return self.variable_symbol.adjoint_symbol

    def is_variable(self):
        return False

    def is_derivative(self):
        return True

    def is_adjoint(self):
        return False

    def is_operation_adjoint(self):
        return False


class ADAdjointSymbol(ADDataSymbol):
    _adjoint_prefix = ""
    _adjoint_postfix = "_adj"

    def __init__(self, variable_symbol):
        if not isinstance(variable_symbol, ADVariableSymbol):
            raise TypeError("")
        name = (
            self._adjoint_prefix + variable_symbol.name + self._adjoint_postfix
        )
        datatype = variable_symbol.datatype
        super().__init__(name, datatype)
        self._variable_symbol = variable_symbol

        # self._zero_assignments = []
        # self._incrementations = []

    @property
    def variable_symbol(self):
        return self._variable_symbol

    @property
    def derivative_symbol(self):
        return self.variable_symbol.derivative_symbol

    # @property
    # def assignments(self):
    #     return self.assignments
    #
    # # TODO:
    # # assign_zero
    # # increment_by
    #
    # @property
    # def incrementations(self):
    #     return self._incrementations
    #
    # def assign(self, value):
    #     assignment = ADAssignment.create(self.create_new_version_reference(),
    #                                    value)

    def is_variable(self):
        return False

    def is_derivative(self):
        return False

    def is_adjoint(self):
        return True

    def is_operation_adjoint(self):
        return False


class ADOperationAdjointSymbol(ADDataSymbol):
    _id = count(0)
    _operation_adjoint_name = "op_adj_"

    def __init__(self, operation_or_instrinsic):
        from psyclone.autodiff.psyir.nodes import (
            ADOperation,
        )  # , ADIntrinsicCall

        if not isinstance(
            operation_or_instrinsic, (ADOperation)
        ):  # , ADIntrinsicCall)):
            raise TypeError("")

        name = self._operation_adjoint_name + str(next(self._id))
        datatype = operation_or_instrinsic.datatype
        super().__init__(name, datatype)
        self._operation = operation_or_instrinsic

        # self._assignment = None

    @property
    def operation(self):
        return self._operation

    # @property
    # def assignement(self):
    #     return self._assignment

    # TODO:
    # assign_zero
    # assign_value

    def is_variable(self):
        return False

    def is_derivative(self):
        return False

    def is_adjoint(self):
        return True

    def is_operation_adjoint(self):
        return True
