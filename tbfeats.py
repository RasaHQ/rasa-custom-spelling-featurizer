import typing
from typing import Any, Optional, Text, Dict, List, Type
import re
from textblob import TextBlob, Word
import numpy as np

from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata
from rasa.nlu.constants import DENSE_FEATURE_NAMES, DENSE_FEATURIZABLE_ATTRIBUTES, TEXT


def split_text(text):
    return re.sub(
        r"[^\w#@&]+(?=\s|$)|"
        r"(\s|^)[^\w#@&]+(?=[^0-9\\s])|"
        r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
        " ",
        text,
    ).split()


class TextBlobTokenizer(Tokenizer):
    language_list = ["en"]
    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Text will be tokenized with case sensitive as default
        "case_sensitive": True,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the TextBlobTokenizer framework."""
        super().__init__(component_config)
        self.case_sensitive = self.component_config["case_sensitive"]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["textblob"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        orig_text = message.get(attribute)
        if not self.case_sensitive:
            orig_text = orig_text.lower()
        text = str(TextBlob(orig_text).correct())
        orig_words = split_text(orig_text)
        words = split_text(text)

        if not words:
            words = [text]

        message.set("original_words", orig_words)
        return self._convert_words_to_tokens(words, text)


class TextBlobFeaturizer(DenseFeaturizer):
    """A new component"""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""
        return [TextBlobTokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["textblob"]

    defaults = {}
    language_list = "en"

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_textblob_features(example, attribute)

    def _set_textblob_features(self, message: Message, attribute: Text = TEXT):
        tokens = [t.text for t in message.data["tokens"] if t != "__CLS__"]
        orig = message.data["original_words"]
        correction_made = [t != o for t, o in zip(tokens, orig)]
        correction_made += [any(correction_made)]  # for __CLS__
        confidence = [Word(o).spellcheck()[0][1] for o in orig]
        confidence += [min(confidence)]  # for __CLS__

        X = np.stack(
            [
                np.array(correction_made).astype(np.float),
                np.array(confidence).astype(np.float),
            ]
        ).T
        features = self._combine_with_existing_dense_features(
            message, additional_features=X, feature_name=DENSE_FEATURE_NAMES[attribute]
        )
        message.set(DENSE_FEATURE_NAMES[attribute], features)

    def process(self, message: Message, **kwargs: Any) -> None:
        self._set_textblob_features(message)

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
