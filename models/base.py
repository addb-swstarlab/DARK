#
# OtterTune - base.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from abc import ABCMeta, abstractmethod
class ModelBase(object, metaclass=ABCMeta):

    @abstractmethod
    def _reset(self):
        pass
