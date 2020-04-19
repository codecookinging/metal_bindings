
from __future__ import print_function, division
import numpy as np

"""
For more information about precision, recall, true positive, true negative,
false positive, false negative. Please read the following Wikipedia article
https://en.wikipedia.org/wiki/Precision_and_recall
"""

_EPS_VALUE = 1e-7


# This pattern: value1 / np.amax([_EPS_VALUE, value1 + value2]) will prevent
# the output of NAN when value1 == value2 == 0

class ConfusionMatrix(object):
    """A multiclass confusion matrix.
    Args:
        num_class (int): Number of classes in the label.
    Attributes:
        confusion_matrix (ndarray): A num_class x num_class ndarray that
            is the confusion matrix.
    """

    def __init__(self, num_class):
        assert isinstance(num_class, int), \
            "The num_class has to be an int: {}".format(type(num_class))

        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
        self._true_positive_dict = {}
        self._false_positive_dict = {}
        self._true_negative_dict = {}
        self._false_negative_dict = {}

    def update(self, label, prediction):
        """Update the confusion matrix with new label and prediction
        Args:
            label (np.ndarray): 1D array that represents the label of the
                prediction, contains data of the type np.int. Higher
                dimensional label should be flattened beforehand.
            prediction (np.ndarray): 1D array that represents the prediction,
                contains data of the type np.int. Higher dimensional
                prediction should be flattened beforehand.
        Returns:
            ndarray: the ndarray of size num_class x num_class that contains
                the update that was used to update the self.confusion_matrix.
                E.g self.confusion_matrix += update_matrix
        """

        assert isinstance(label, np.ndarray)
        assert isinstance(prediction, np.ndarray)
        assert label.dtype == np.int, \
            'The label has to be of type np.int: {}'.format(label.dtype)
        assert prediction.dtype == np.int, \
            'The prediction has to be of type np.int: {}'.format(
                prediction.dtype)
        assert label.shape == prediction.shape, \
            'The shape of the label and prediction shoudl be the same: {} vs {}' \
                .format(label.shape, prediction.shape)

        self._reset_stats()

        update_matrix = np.zeros((self.num_class, self.num_class))
        for (l_value, p_value) in zip(label, prediction):
            update_matrix[l_value][p_value] += 1

        self.confusion_matrix = update_matrix + self.confusion_matrix

        return update_matrix

    def _reset_stats(self):
        self._true_positive_dict = {}
        self._false_positive_dict = {}
        self._true_negative_dict = {}
        self._false_negative_dict = {}

    def reset(self):
        """Resets the confusion matrix."""

        self._reset_stats()
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

    def get_confusion_matrix(self):
        """Get confusion matrix."""

        return self.confusion_matrix

    def get_true_positive(self, class_id):
        """Get true positive of the given class id.
        Args:
            class_id (int): class id for the class to obtain the stats for
        Returns:
            float: True positive of the class with given class_id.
        """
        assert isinstance(class_id, int), \
            "The type of class_id must be int: {}".format(type(int))

        if class_id in self._true_positive_dict:
            return self._true_positive_dict[class_id]

        true_positive = self.confusion_matrix[class_id][class_id]

        self._true_positive_dict[class_id] = true_positive
        return true_positive

    def get_false_positive(self, class_id):
        """Get false positive of the given class id.
        Args:
            class_id (int): class id for the class to obtain the stats for
        Returns:
            float: False positive of the class with given class_id.
        """
        assert isinstance(class_id, int), \
            "The type of class_id must be int: {}".format(type(int))

        if class_id in self._false_positive_dict:
            return self._false_positive_dict[class_id]

        false_positive = np.sum(self.confusion_matrix[:, class_id]) - \
                         self.get_true_positive(class_id)

        self._false_positive_dict[class_id] = false_positive

        return false_positive

    def get_false_negative(self, class_id):
        """Get false negative of the given class id.
        Args:
            class_id (int): class id for the class to obtain the stats for
        Returns:
            float: False negative of the class with given class_id.
        """
        assert isinstance(class_id, int), \
            "The type of class_id must be int: {}".format(type(int))

        if class_id in self._false_negative_dict:
            return self._false_negative_dict[class_id]

        false_negative = np.sum(self.confusion_matrix[class_id, :]) - \
                         self.get_true_positive(class_id)

        self._false_negative_dict[class_id] = false_negative

        return false_negative

    def get_true_negative(self, class_id):
        """Get true negative of the given class id.
        Args:
            class_id (int): class id for the class to obtain the stats for
        Returns:
            float: True negative of the class with given class_id.
        """
        assert isinstance(class_id, int), \
            "The type of class_id must be int: {}".format(type(int))

        if class_id in self._true_negative_dict:
            return self._true_negative_dict[class_id]

        true_negative = np.sum(self.confusion_matrix) - (
                self.get_true_positive(class_id) +
                self.get_false_positive(class_id) +
                self.get_false_negative(class_id))

        self._true_negative_dict[class_id] = true_negative

        return true_negative


# TODO make ConfusionMatrixBinary use methods in ConfusionMatrix
class ConfusionMatrixBinary(ConfusionMatrix):
    """A class that stores a Binary Confusion Matrix
    Args:
        threshold (float): The threshold at which the predicion is considered
            true.
        class_name (str): Can be use to id the object (default None)
    Attributes:
        confusion_matrix (np.ndarray): A 2 by 2 confusion matrix containing data
            with the type int in the following format:
                -----------
                | TP | FN |
                -----------
                | FP | TN |
                -----------
    """

    def __init__(self, class_name=None):
        if class_name:
            assert isinstance(class_name, str), \
                'The class_name needs to be str: {}'.format(type(class_name))

        self.class_name = class_name
        self.num_class = 2
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

    def get_true_positive(self):
        """Returns the numper of True Positives.
        Returns:
            int: Number of True Positives.
        """
        return self.confusion_matrix[0, 0]

    def get_false_positive(self):
        """Returns the numper of False Positives.
        Returns:
            int: Number of False Positives.
        """
        return self.confusion_matrix[1, 0]

    def get_true_negative(self):
        """Returns the numper of True Negatives.
        Returns:
            int: Number of True Negatives.
        """
        return self.confusion_matrix[1, 1]

    def get_false_negative(self):
        """Returns the numper of False Negatives.
        Returns:
            int: Number of False Negatives.
        """
        return self.confusion_matrix[0, 1]

    def update(self, label, prediction):
        """Update the confusion matrix with new label and prediction
        Args:
            label (np.ndarray): 1D array that represents the label of the
                prediction, contains data of the type np.bool. Higher
                dimensional label should be flattened beforehand.
            prediction (np.ndarray): 1D array that represents the prediction,
                contains data of the type np.bool. Higher dimensional
                prediction should be flattened beforehand.
        Returns:
            dict: the dictionary that contains the update that was used to
                update the self.confusion_matrix. The dictionary is in the
                following format:
                {'tp':true_positives, 'fp':false_positives,
                 'tn':true_negatives, 'fn':false_negatives}
        """
        assert isinstance(label, np.ndarray)
        assert isinstance(prediction, np.ndarray)
        assert label.dtype == np.bool, \
            'The label have to be of type np.bool: {}'.format(label.dtype)
        assert prediction.dtype == np.bool, \
            'The prediction havs to be of type np.bool: {}'.format(
                prediction.dtype)
        assert label.shape == prediction.shape, \
            'The shape of the label and prediction shoudl be the same: {} vs {}' \
                .format(label.shape, prediction.shape)

        update_dict = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        update_dict['tp'] = np.sum(prediction * label)
        update_dict['tn'] = np.sum(~prediction * ~label)
        update_dict['fp'] = np.sum(prediction * ~label)
        update_dict['fn'] = np.sum(~prediction * label)

        self.confusion_matrix[0, 0] += update_dict['tp']
        self.confusion_matrix[1, 1] += update_dict['tn']
        self.confusion_matrix[1, 0] += update_dict['fp']
        self.confusion_matrix[0, 1] += update_dict['fn']

        return update_dict

    def get_precision(self):
        true_positive = self.get_true_positive()
        false_positive = self.get_false_positive()
        precision = true_positive / np.amax([_EPS_VALUE,
                                             true_positive + false_positive])
        return precision

    def get_recall(self):

        true_positive = self.get_true_positive()
        false_negative = self.get_false_negative()
        recall = true_positive / np.amax([_EPS_VALUE,
                                          true_positive + false_negative])
        return recall
    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_mcc(self):
        import math
        true_positive = self.get_true_positive()
        false_negative = self.get_false_negative()
        false_positive = self.get_false_positive()
        true_negative = self.get_true_negative()
        mcc = (true_positive*true_negative-false_negative*false_negative)/np.amax([_EPS_VALUE,math.sqrt((true_positive+false_negative)*(true_positive+false_positive)*(true_negative+false_positive)*(false_negative+true_negative))])
        #print(self.confusion_matrix)
        return mcc







