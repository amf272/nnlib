from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from . import utils


class Metric(ABC):
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("name is not implemented")

    @abstractmethod
    def value(self, *args, **kwargs):
        raise NotImplementedError("value is not implemented")

    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_iteration_end(self, *args, **kwargs):
        pass


class Accuracy(Metric):
    def __init__(self, output_key: str = 'pred', label_index=0, **kwargs):
        super(Accuracy, self).__init__(**kwargs)
        self.output_key = output_key
        self.label_index = label_index

        # initialize and use later
        self._accuracy_storage = defaultdict(list)
        self._accuracy = defaultdict(dict)

    @property
    def name(self):
        return "accuracy"

    def value(self, epoch, partition, **kwargs):
        return self._accuracy[partition].get(epoch, None)

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy_storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy_storage[partition])
        self._accuracy[partition][epoch] = accuracy
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}_{self.output_key}_{self.label_index}", accuracy, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(outputs[self.output_key]).argmax(axis=1).astype(np.int)
        batch_labels = utils.to_numpy(batch_labels[self.label_index]).astype(np.int)
        self._accuracy_storage[partition].append((pred == batch_labels).astype(np.float).mean())


class Loss(Metric):
    def __init__(self, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self._value = defaultdict(lambda: defaultdict(float))
        self._num_samples = defaultdict(lambda: defaultdict(float))

    @property
    def name(self):
        return "loss"

    def value(self, epoch, partition, **kwargs):
        return self._value[partition].get(epoch, None) / self._num_samples[partition].get(epoch, None)

    def on_epoch_start(self, partition, **kwargs):
        pass

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", self.value(epoch, partition), epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, batch_losses, epoch, num_samples, **kwargs):
        total_loss = sum(batch_losses.values())
        self._value[partition][epoch] += total_loss * num_samples
        self._num_samples[partition][epoch] += num_samples


# TODO implement this
# TODO also implement sklearn metric recording
class AUROC(Metric):
    def __init__(self, output_key: str = 'pred', label_index=0, **kwargs):
        super(AUROC, self).__init__(**kwargs)
        self.output_key = output_key
        self.label_index = label_index

        # initialize and use later
        self._storage = defaultdict(list)
        self._metric = defaultdict(dict)
        raise NotImplementedError("auroc is not yet correctly implemented")

    @property
    def name(self):
        return "auroc"

    def value(self, epoch, partition, **kwargs):
        return self._metric[partition].get(epoch, None)

    def on_epoch_start(self, partition, **kwargs):
        self._storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.concatenate(self._storage[partition])
        self._metric[partition][epoch] = accuracy
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}_{self.output_key}_{self.label_index}", accuracy, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(outputs[self.output_key]).argmax(axis=1).astype(np.int)
        batch_labels = utils.to_numpy(batch_labels[self.label_index]).astype(np.int)
        self._storage[partition].append((batch_labels, pred))


class TopKAccuracy(Metric):
    def __init__(self, k, output_key: str = 'pred', label_index=0, **kwargs):
        super(TopKAccuracy, self).__init__(**kwargs)
        self.k = k
        self.output_key = output_key
        self.label_index = label_index

        # initialize and use later
        self._accuracy_storage = defaultdict(list)
        self._accuracy = defaultdict(dict)

    @property
    def name(self):
        return f"top{self.k}_accuracy"

    def value(self, partition, epoch, **kwargs):
        return self._accuracy[partition].get(epoch, None)

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy_storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy_storage[partition])
        self._accuracy[partition][epoch] = accuracy
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}_{self.output_key}_{self.label_index}", accuracy, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(outputs[self.output_key])
        batch_labels = utils.to_numpy(batch_labels[self.label_index]).astype(np.int)

        topk_predictions = np.argsort(-pred, axis=1)[:, :self.k]
        batch_labels = batch_labels.reshape((-1, 1)).repeat(self.k, axis=1)
        topk_correctness = (np.sum(topk_predictions == batch_labels, axis=1) >= 1)

        self._accuracy_storage[partition].append(topk_correctness.astype(np.float).mean())
