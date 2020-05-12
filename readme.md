<img src="square-logo.svg" width=200 height=200 align="right">

This repository contains a demo project with a custom made printer component.

It is maintained by Vincent D. Warmerdam, Research Advocate as [Rasa](https://rasa.com/).

# Make a Printer Component

Rasa offers many useful components to build a digital assistant but 
sometimes you may want to write your own. This document will be part
of a series where we will create increasingly complex components from
scratch. In this document we build two components that depend on eachother;
a tokenizer that applies a spelling correction and a featurizer that
will generate features telling the model if there's been one. 

## Example Project

You can clone the repository found [here]() if you'd like to be able
to run the same project. The repository contains a relatively small 
rasa project; we're only dealing with four intents and one entity. 
Here's some of the files in the project:

### `data/nlu.md`

```md
## intent:greet
- hey
- hello
...

## intent:goodbye
- bye
- goodbye
...

## intent:bot_challenge
- are you a bot?
- are you a human?
...

## intent:talk_code
- i want to talk about [python](proglang)
- Code to ask yes/no question in [javascript](proglang)
...
```

### `data/stories.md`

```md
## just code
* talk_code
  - utter_proglang

## check bot
* bot_challenge
  - utter_i_am_bot
* goodbye
  - utter_goodbye

## hello and code
* greet
    - utter_greet
* talk_code{"proglang": "python"}
    - utter_proglang
```

Once we call `rasa train` on the command line these files will generate training data for our machine
learning pipeline. You can see the definition of this pipeline in the `config.yml` file.

## `config.yml`

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: LexicalSyntacticFeaturizer
- name: DIETClassifier
  epochs: 20
policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
  - name: MappingPolicy
```

## Printing Context 

The schematic below shows the lifecycle of components in Rasa.

![](https://blog.rasa.com/content/images/2019/02/Rasa-Component-Lifecycle--Train-.png)

Let's make our own component to make it explicit what data gets created. The 
goal of the component will be to print all available information known
at a certain point in the pipeline. This way, our new pipeline may
look something like this; 

## `config.yml`

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: printer.Printer 
  alias: after tokenizer
- name: CountVectorsFeaturizer
- name: printer.Printer
  alias: after 1st cv
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: printer.Printer
  alias: after 2nd cv
- name: LexicalSyntacticFeaturizer
- name: printer.Printer
  alias: after lexical syntactic featurizer
- name: DIETClassifier
  epochs: 20
- name: printer.Printer
  alias: after diet classifier
policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
  - name: MappingPolicy
```

Let's note a few things. 

1. We've added new steps that have the name `printer.Printer`. This is a custom component that we'll need to create. 
2. We've placed the `printer.Printer` component after each featurization step. The goal is that this component prints what information is created in each step.
3. We've also placed the `printer.Printer` component after the `DIETClassifier` step. This should allow us to directly see the model output.
4. The custom component takes an argument `alias` that allows us to give it an extra name. This means that the component that we'll create needs to be able to read in parameters passed in `config.yml`.

## Making the `printer.Printer` Component

We will need to create a new file called `printer.py` to put the new
`Printer` component in. Note that this is also how `config.yml` is 
able to find the `printer.Printer` component. To get started writing
the component I took the example from [the documentation](https://rasa.com/docs/rasa/api/custom-nlu-components/)
and made some changes to it. 

### `printer.py`

```python
import typing
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Token

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


def _is_list_tokens(v):
  """
  This is a helper function.
  It checks if `v` is a list of tokens. 
  If so, we need to print differently.
  """
    if isinstance(v, List):
        if len(v) > 0:
            if isinstance(v[0], Token):
                return True
    return False


class Printer(Component):
    
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return []

    defaults = {"alias": None}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        if self.component_config['alias']:
            print("\n")
            print(self.component_config['alias'])
        for k, v in message.data.items():
            if _is_list_tokens(v):
                print(f"{k}: {[t.text for t in v]}")
            else:
                print(f"{k}: {v.__repr__()}")

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
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
```

Most of the code in this file is exactly the same as what you will find in
[the documentation](https://rasa.com/docs/rasa/api/custom-nlu-components/).
Let's observe a few things here. 

1. We've created a `Printer` object that inherits from `rasa.nlu.components.Component`.
1. This component does not depend on other components. You can confirm this by
looking at the `required_components` method. If this component was a 
`CountVectorizer` then it would depend on tokens being present and this method 
would be the place where you would specify that.
2. Right after this method we declare `defaults = {"alias": None}`. This sets
the default value for the `alias` setting that we could set in the `config.yml`.
3. Right after this statement we declare `language_list = None`. This means that 
the component does not depend on a language. It's important to note that some 
components only work for certain languages. For example, the `ConveRTFeaturizer`
will only work for the English language.
4. The `load`, `persist` and `train` methods are untouched and are also not relevant
for this component. Since we're merely printing there's no need for a training
phase or a phase where we load/store everything we've trained on disk.

The main change that we've made is in the `process` method which we'll zoom in on below. 

```python
def process(self, message: Message, **kwargs: Any) -> None:
    if self.component_config['alias']:
        print("\n")
        print(self.component_config['alias'])
    for k, v in message.data.items():
        if _is_list_tokens(v):
            print(f"{k}: {[t.text for t in v]}")
        else:
            print(f"{k}: {v.__repr__()}")
```

The `process` method of the `Component` object is where all the logic gets
applied. In our case this is where all the printing happens. We can access all 
the available data by parsing the `message` that the method receives. In particular, we peek
inside of `message.data` and iterate over all the items. These all get printed. 

## See the Effect 

Let's train and run this. 

```
> rasa train
> rasa shell
```

When you now talk to the assistant you'll see extra printed lines appear. When 
we type `hello there` you should see the following messages being printed. 

### `printer.Printer` with alias `after tokenizer`

This is the information that we see right after tokenisation. 

```
after tokenizer
intent: {'name': None, 'confidence': 0.0}
entities: []
tokens: ['hello', 'there', '__CLS__']
```

Note that we have three tokens. The `__CLS__` token serves as
a token that summarises the entire sentence.

### `printer.Printer` with alias `after 1st cv`

We now see that there are some sparse text features that 
have been added. 
```
after 1st cv
intent: {'name': None, 'confidence': 0.0}
entities: []
tokens: ['hello', 'there', '__CLS__']
text_sparse_features: <3x272 sparse matrix of type '<class 'numpy.int64'>'
        with 4 stored elements in COOrdinate format>
```

Note the size of the sparse matrix. We keep track of features for three tokens,
one of which is the `__CLS__` token.

### `printer.Printer` with alias `after 2nd cv`

We now see that more sparse text features have been added. Because
the settings specify that we're counting bigrams we also see that 
we add about 2250 features for each token by doing so.

```
after 2nd cv
intent: {'name': None, 'confidence': 0.0}
entities: []
tokens: ['hello', 'there', '__CLS__']
text_sparse_features: <3x2581 sparse matrix of type '<class 'numpy.longlong'>'
        with 80 stored elements in COOrdinate format>
```

### `printer.Printer` with alias `after tokenizer`

The `LexicalSyntacticFeaturizer` adds another 24 features per token.

```
after lexical syntactic featurizer
intent: {'name': None, 'confidence': 0.0}
entities: []
tokens: ['hello', 'there', '__CLS__']
text_sparse_features: <3x2605 sparse matrix of type '<class 'numpy.float64'>'
        with 112 stored elements in COOrdinate format>
```

Note that the features for the `__CLS__` token at this point is the sum
of all the sparse tokens. Since all the features are sparse this is a
reasonable way to summarise the features of all the words into a set of
features that represents the entire utterance. 

### `printer.Printer` with alias `after diet classifier`

All the sparse features went into the `DIETClassifier` and this produced some 
output. You can confirm that the pipeline now actually produces an intent.

```
after diet classifier
intent: {'name': 'greet', 'confidence': 0.9849509000778198}
entities: []
tokens: ['hello', 'there', '__CLS__']
text_sparse_features: <3x2605 sparse matrix of type '<class 'numpy.float64'>'
        with 112 stored elements in COOrdinate format>
intent_ranking: [{'name': 'greet', 'confidence': 0.9849509000778198}, {'name': 'talk_code', 'confidence': 0.008203224278986454}, {'name': 'goodbye', 'confidence': 0.005775876808911562}, {'name': 'bot_challenge', 'confidence': 0.0010700082639232278}]
```

If you were now to utter `i want to talk about python` you should see 
similar lines being printer but at the end you will now also see that
entities have been detected.

## Conclusion 

So what have we seen in this guide? 

- We've seen how to create a custom component that can read in settings from `config.yml`. 
- We've seen what features the component receives by looking at the output from the `printer.Printer`.
- We've seen that the Rasa components continously add information to the message that is passed. 

