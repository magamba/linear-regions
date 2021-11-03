"""
Custom callables for ignite
"""

import abc
from enum import Enum


class _Comparison(Enum):
    LT = 1
    GT = 2


class _BaseActOnInterpolateByMetric(abc.ABC):
    def __init__(self, comp, metric_name, threshold):
        self._comp = comp
        self._metric_name = metric_name
        self._threshold = threshold

    @abc.abstractmethod
    def act_on_interpolate(self, evaluator, trainer):
        """Do some action once the trainer has gotten to a certain accuracy"""

    def __call__(self, evaluator, trainer):
        metric_val = evaluator.state.metrics[self._metric_name]
        print("metric_val: ", self._metric_name, metric_val)
        if self._comp == _Comparison.GT and metric_val >= self._threshold:
            print("metric criterion met - stopping training")
            self.act_on_interpolate(evaluator, trainer)
        elif self._comp == _Comparison.LT and metric_val <= self._threshold:
            self.act_on_interpolate(evaluator, trainer)


class StopOnInterpolateByAccuracy(_BaseActOnInterpolateByMetric):
    def __init__(self, acc_name="accuracy", threshold=1.0):
        super().__init__(_Comparison.GT, acc_name, threshold)

    def act_on_interpolate(self, evaluator, trainer):
        print("Accuracy is greater than threshold, terminating engine")
        trainer.terminate()


class StopOnInterpolateByLoss(_BaseActOnInterpolateByMetric):
    def __init__(self, loss_name="loss", threshold=0.0):
        super().__init__(_Comparison.LT, loss_name, threshold)

    def act_on_interpolate(self, evaluator, trainer):
        print("Loss is less than threshold, terminating engine")
        trainer.terminate()


class ContinueOnInterpolateByAccuracy(_BaseActOnInterpolateByMetric):
    def __init__(self, extra_epochs, acc_name="accuracy", threshold=1.0):
        self._extra_epochs = extra_epochs
        self._in_extra_epochs = False
        super().__init__(_Comparison.GT, acc_name, threshold)

    def act_on_interpolate(self, evaluator, trainer):
        if self._in_extra_epochs:
            return

        trainer.state.max_epochs += trainer.state.epoch + self._extra_epochs
        self._in_extra_epochs = True
