

from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

from bolna.classification.classification import BaseClassifier
from bolna.helpers.logger_config import configure_logger


logger = configure_logger(__name__)


load_dotenv()
    

class DeBERTaClassifier(BaseClassifier):
    def __init__(self, model_id, prompt, labels, threshold, multi_label=False, filename = None):
        super().__init__(model_id, prompt, labels, multi_label, threshold)
        self.model_args = { "model_id": self.model_name}
        logger.info(f"Creating for {self.model_name}, classifier {model_id}")
        if filename is not None:
            self.model_args['file_name'] = filename
        self.model = ORTModelForSequenceClassification.from_pretrained(**self.model_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.model_input_names = ['input_ids', 'attention_mask']
        self.classifier = pipeline("zero-shot-classification", model=self.model, tokenizer=self.tokenizer)
        
    def classify(self, text):
        output = self.classifier(text, self.classification_labels, multi_label=self.multi_label)
        logger.info(f"Most eligible response {output['labels'][0]}")
        return output['labels'][0] if output['scores'][0] > self.threshold else None

