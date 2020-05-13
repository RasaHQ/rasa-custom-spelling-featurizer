import typing
import os
from typing import Any, Optional, Text, Dict, List, Type
import numpy as np
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.featurizers.featurizer import sequence_to_sentence_features
from rasa.nlu.constants import DENSE_FEATURE_NAMES, TEXT, DENSE_FEATURIZABLE_ATTRIBUTES
from sklearn.ensemble import IsolationForest


from joblib import dump, load

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


def _is_list_tokens(v):
    if isinstance(v, List):
        if len(v) > 0:
            if isinstance(v[0], Token):
                return True
    return False


class OutlierComponent(DenseFeaturizer):
    """A new component"""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""
        return [DenseFeaturizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    defaults = {"n_estimators": 100}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        self.isolation = IsolationForest(self.component_config["n_estimators"])

    def _set_outlier_features(self, message: Message, attribute: Text = TEXT):
        X = message.data[DENSE_FEATURE_NAMES[attribute]]
        scores = self.isolation.score_samples(X).reshape(-1, 1)

        features = self._combine_with_existing_dense_features(
            message,
            additional_features=scores,
            feature_name=DENSE_FEATURE_NAMES[attribute],
        )
        message.set(DENSE_FEATURE_NAMES[attribute], features)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        X = np.stack(
            [
                sequence_to_sentence_features(example.get(DENSE_FEATURE_NAMES[TEXT]))
                for example in training_data.intent_examples
            ]
        )
        X = X.reshape(X.shape[0], X.shape[2])
        self.isolation.fit(X)
        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_outlier_features(example, attribute)

    def process(self, message: Message, **kwargs: Any) -> None:
        self._set_outlier_features(message)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        path = os.path.join(model_dir, file_name + ".pkl")
        dump(self, path)
        return {"isolation_file": file_name + ".pkl"}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        else:
            filename = meta.get("isolation_file")
            filepath = os.path.join(model_dir, filename)
        return load(filepath)
