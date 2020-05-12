import typing
from typing import Any, Optional, Text, Dict, List, Type
import numpy as np
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.featurizers.featurizer import sequence_to_sentence_features
from rasa.nlu.constants import SPARSE_FEATURE_NAMES, DENSE_FEATURE_NAMES, TEXT
from sklearn.ensemble import IsolationForest

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


def _is_list_tokens(v):
    if isinstance(v, List):
        if len(v) > 0:
            if isinstance(v[0], Token):
                return True
    return False


class OutlierComponent(Component):
    """A new component"""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""
        return [SparseFeaturizer]
    
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    defaults = {"n_estimators": 100}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        X = np.stack(
            [
                sequence_to_sentence_features(
                    example.get(SPARSE_FEATURE_NAMES[TEXT]).toarray()
                )
                for example in training_data.intent_examples
            ]
        )
        # reduce dimensionality
        X = np.reshape(X, (len(X), -1))
        self.isolation = IsolationForest(self.defaults['n_estimators'])
        print(X)
        self.isolation.fit(X)

    def process(self, message: Message, **kwargs: Any) -> None:
        X_new = message.data['text_sparse_features']
        message.set('text_dense_features', self.isolation.predict(X_new) == -1, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        # the reason why it is failing is because this is not persisting
        pass

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
            return cls(meta)
