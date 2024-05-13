from abc import ABC, abstractmethod
from dataclasses import dataclass
from modelgauge.secret_values import Injector
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.tracked_object import TrackedObject
from pydantic import BaseModel
from typing import Generic, List, Mapping, Sequence, Type, TypeVar

AnnotatorsConfig = Mapping[str, Mapping[str, str]]

AnnotatorType = TypeVar("AnnotatorType", bound="Annotator")
AnnotationType = TypeVar("AnnotationType", bound=BaseModel)

class ConfigDescription(BaseModel):
    """How to look up a provider config."""

    scope: str
    key: str


ConfigType = TypeVar("ConfigType", bound="Config")


@dataclass(frozen=True)
class Config(ABC):
    """Base class for all secrets."""

    @classmethod
    @abstractmethod
    def description(cls) -> ConfigDescription:
        """Information about how to lookup/obtain the config."""
        pass

    @classmethod
    @abstractmethod
    def make(cls: Type[ConfigType], raw_config: AnnotatorsConfig) -> ConfigType:
        """Read the config value from `raw_config` to make this class."""
        pass

RequiredConfigType = TypeVar("RequiredConfigType", bound="RequiredConfig")

class RequiredConfig(Config):
    """Base class for all required providers."""

    def __init__(self, value: str):
        super().__init__()
        self._value = value

    @property
    def value(self) -> str:
        """Get the value of the provider."""
        return self._value
    
    @classmethod
    def make(cls: Type[RequiredConfigType], raw_config: AnnotatorsConfig) -> RequiredConfigType:
        """Construct this class from the config.
        
        Raises MissingConfigValues if required info is missing.
        """
        config = cls.description()
        try:
            return cls(raw_config[config.scope][config.key])
        except KeyError:
            raise MissingConfigValues([config])

class MissingConfigValues(LookupError):
    """Exception describing one or more missing required config values."""

    def __init__(self, descriptions: Sequence[ConfigDescription]):
        assert descriptions, "Must have at least 1 description to raise an error."
        self.descriptions = descriptions
    
    @staticmethod
    def combine(errors: Sequence["MissingConfigValues"]) -> "MissingConfigValues":
        """Combine multiple exceptions into one."""
        descriptions: List[ConfigDescription] = []
        for error in errors:
            descriptions.extend(error.descriptions)
        return MissingConfigValues(descriptions)

    def __str__(self):
        message = "Missing the following configuration:\n"
        for d in self.descriptions:
            # TODO: Make this nicer
            message += str(d) + "\n"
        return message

class Annotator(TrackedObject, ABC):
    """The base class for all annotators."""

    pass


class InjectAnnotator(Injector, Generic[AnnotatorType]):
    def __init__(self, annotator_class: Type[AnnotatorType]):
        self.annotator_class = annotator_class

    def inject(self, annotators_config: AnnotatorsConfig) -> AnnotatorType:
        return self.annotator_class.make(annotators_config)

    def __repr__(self):
        return f"InjectAnnotator({self.annotator_class.__name__})"

class CompletionAnnotator(Annotator, Generic[AnnotationType]):
    """Annotator that examines a single prompt+completion pair at a time.

    Subclasses can report whatever class they want, as long as it inherits from Pydantic's BaseModel.
    """

    @abstractmethod
    def translate_request(self, prompt: PromptWithContext, completion: SUTCompletion):
        """Convert the prompt+completion into the native representation for this annotator."""
        pass

    @abstractmethod
    def annotate(self, annotation_request):
        """Perform annotation and return the raw response from the annotator."""
        pass

    @abstractmethod
    def translate_response(self, request, response) -> AnnotationType:
        """Convert the raw response into the form read by Tests."""
        pass
