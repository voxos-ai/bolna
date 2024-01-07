import random
import json
import asyncio
import traceback
from .base_agent import BaseAgent
from bolna.constants import USERS_KEY_ORDER
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

class Node:
    def __init__(self, node_id, node_label, content, classification_labels: list = None, prompt=None, milestone_check_prompt=None,
                 children: list = None):
        self.node_id = node_id
        self.node_label = node_label
        self.content = content
        self.children = children
        self.classification_labels = classification_labels
        self.prompt = prompt
        self.need_response_generation = False
        self.milestone_check_prompt = milestone_check_prompt


class Graph:
    def __init__(self, conversation_data, preprocessed=False):
        self.preprocessed = preprocessed
        self.root = None
        self.graph = self._create_graph(conversation_data)

    def _create_graph(self, data):
        node_map = dict()
        for node_id, node_data in data.items():
            node = Node(
                node_id=node_id,
                node_label=node_data["label"],
                content=node_data["content"],
                classification_labels=node_data.get("classification_labels", []),
                prompt=node_data.get("prompt"),
                children=[],
                milestone_check_prompt=node_data.get("milestone_check_prompt", ""),
            )
            node_map[node_id] = node
            if node_data["is_root"]:
                self.root = node

        for node_id, node_data in data.items():
            children_ids = node_data.get("children", [])
            node_map[node_id].children = [node_map[child_id] for child_id in children_ids] if children_ids else []
        return node_map

    # @TODO complete this function
    def remove_node(self, parent, node):
        print("Not yet implemented")


class GraphBasedConversationAgent(BaseAgent):
    def __init__(self, llm, prompts, context_data=None, preprocessed=True, log_dir_name=None):
        super().__init__()
        # Config
        self.brain = llm
        self.context_data = context_data
        self.preprocessed = preprocessed

        # Goals
        self.graph = Graph(prompts)
        self.current_node = self.graph.root
        self.conversation_intro_done = False

    def _get_audio_text_pair(self, node):
        ind = random.randint(0, len(node.content) - 1)
        audio_pair = node.content[ind]
        if "audio" not in audio_pair:
            recipient_data = self.context_data.get('recipient_data', {})
            # @TODO: Need to make this dynamic
            audio_pair["text"] = audio_pair["text"].format(' '.join([recipient_data.get(k, '') for k in USERS_KEY_ORDER]))
            audio_pair["audio"] = recipient_data.get('audio')
        return audio_pair

    async def _get_next_formulaic_agent_next_step(self, history, stream=True, synthesize=False):
        pass

    def _handle_intro_message(self):
        audio_pair = self._get_audio_text_pair(self.current_node)
        self.conversation_intro_done = True
        logger.info("Conversation intro done")
        if self.current_node.prompt == None:
            #These are convos with two step intros
            ind = random.randint(0, len(self.current_node.children) - 1)
            self.current_node  = self.current_node.children[ind]
        return audio_pair

    async def _get_next_preprocessed_step(self, history):
        logger.info(f"current node {self.current_node.node_label}, self.current_node == self.graph.root {self.current_node == self.graph.root}, and self.conversation_intro_done {self.conversation_intro_done}")
        if self.current_node == self.graph.root and not self.conversation_intro_done:
            return self._handle_intro_message()

        logger.info(f"Conversation intro was done and hence moving forward")
        if len(history) > 7:
            # @TODO: Add summary of left out messages
            prev_messages = history[-6:]
        else:
            prev_messages = history[1:]

        message = [{"role": "system", "content": self.current_node.prompt}] + prev_messages
        # Get classification label from LLM
        response = await self.brain.generate(message, True, False, request_json=True)
        classification_result = json.loads(response)
        label = classification_result["classification_label"]
        for child in self.current_node.children:
            if child.node_label.strip().lower() == label.strip().lower():
                self.current_node = child
                return self._get_audio_text_pair(child)

    async def generate(self, history, stream=False, synthesize=False, label_flow=None):
        try:
            if self.preprocessed:
                if len(self.current_node.children) == 0:
                    ind = random.randint(0, len(self.current_node.content) - 1)
                    audio_pair = self.current_node.content[ind]
                    logger.info('Agent: {}'.format(audio_pair.get('text')))
                    yield audio_pair["audio"]
                else:
                    next_state = await self._get_next_preprocessed_step(history)
                    logger.info('Agent: {}'.format(next_state.get('text')))
                    history.append({'role': 'assistant', 'content': next_state['text']})
                    yield next_state["audio"]
                
                if len(self.current_node.children) == 0:
                    await asyncio.sleep(1)
                    yield "<end_of_conversation>"
            else:
                # @TODO add non-preprocessed flow
                async for message in self._get_next_formulaic_agent_next_step(history, True, False):
                    yield message

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error sending intro text: {e}")