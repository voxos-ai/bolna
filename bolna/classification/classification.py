from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseClassifier:
    def __init__(self, model, prompt, labels, threshold = 0.6, multi_label = False):
        self.model_name = model
        self.prompt = prompt
        self.classification_labels = labels
        self.multi_label = multi_label
        self.threshold = threshold

    async def classify(self, messages):
        pass