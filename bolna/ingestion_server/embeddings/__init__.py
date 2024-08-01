from typing import Dict

from bolna.ingestion_server.embeddings.openai_emded import OpenAI
from bolna.ingestion_server.embeddings.base import BaseEmbed

# EmbedProviders dictionary with the key being the name of the provider and the value being the class of the provider
EmbedProviders : Dict[str, BaseEmbed] = {
    "OpenAI": OpenAI
}   