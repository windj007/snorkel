import os, numpy
from ...learning.disc_learning import NoiseAwareModel
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

try:
    import cPickle as pickle
except ImportError:
    import pickle


class SKLearnNoiseAwareModel(NoiseAwareModel):

    def __init__(self, classifier, save_file=None, name='TFModel'):
        """Interface for a SkLearn model
        The @train_fn, @loss, @prediction, and @save_dict
        fields should be populated by @_build()
        """
        super(SKLearnNoiseAwareModel, self).__init__(name)
        self.classifier = classifier
        if not getattr(self.classifier, 'predict_proba'):
            self.classifier = CalibratedClassifierCV(base_estimator=self.classifier)
        # Load model
        if save_file is not None and os.path.isfile(save_file):
            self.load(save_file)

    def save(self, model_name=None, verbose=True):
        """Save current SkLearn model
            @model_name: save file names
            @verbose: be talkative?
        """
        with open(self._get_fname(model_name or self.name), 'wb') as f:
            pickle.dump(self.classifier, f, 2)
        if verbose:
            print("[{0}] Model saved. To load, use name\n\t\t{1}".format(
                self.name, model_name
            ))

    def load(self, model_name, verbose=True):
        """Load TensorFlow model from file
            @model_name: save file names
            @verbose: be talkative?
        """
        with open(self._get_fname(model_name), 'rb') as f:
            self.classifier = pickle.load(f)
        if verbose:
            print("[{0}] Loaded model <{1}>".format(self.name, model_name))

    def _get_fname(self, model_name):
        return './' + model_name

    def train(self, X, training_marginals, pos_prob = 0.5, **kwargs):
        y = numpy.where(training_marginals > pos_prob, 1.0, 0.0)
        self.classifier.fit(X, y)

    def marginals(self, X, **kwargs):
        return self.classifier.predict_proba(X)
