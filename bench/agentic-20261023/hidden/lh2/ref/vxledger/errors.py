# SPDX-License-Identifier: MPL-2.0
class VxError(Exception):
    code: int = 4700


class AmountError(VxError):
    code = 4701


class ValidationError(VxError):
    code = 4702


class UnknownAccountError(VxError):
    code = 4703


class InsufficientFundsError(VxError):
    code = 4740


class CurrencyMismatchError(VxError):
    code = 4705


class LoadError(VxError):
    code = 4706


class UnknownHoldError(VxError):
    code = 4770
