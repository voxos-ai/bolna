from typing import Dict

from embeddings.openai_emded import OpenAI
from embeddings.base import BaseEmbed

# EmbedProviders dictionary with the key being the name of the provider and the value being the class of the provider
EmbedProviders : Dict[str, BaseEmbed] = {
    "OpenAI": OpenAI
}   