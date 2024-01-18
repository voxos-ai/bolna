from bolna.models import *
from bolna.agent_manager import AssistantManager
# "tasks": [
#             {
#                 "task_type": "conversation",
#                 "tools_config": {
#                     "llm_agent": {
#                         "max_tokens": 100,
#                         "family": "openai",
#                         "streaming_model": "gpt-3.5-turbo-16k",
#                         "agent_flow_type": "streaming",
#                         "classification_model": "gpt-3.5-turbo-16k",
#                         "use_fallback": true,
#                         "temperature": 0.2
#                     },
#                     "synthesizer": {
#                         "provider": "elevenlabs",
#                         "buffer_size": 40,
#                         "audio_format": "pcm",
#                         "stream": true
#                     },
#                     "transcriber": {
#                         "model": "deepgram",
#                         "stream": true,
#                         "language": "en",
#                         "endpointing": 400
#                     },
#                     "input": {
#                         "provider": "default",
#                         "format": "pcm"
#                     },
#                     "output": {
#                         "provider": "default",
#                         "format": "pcm"
#                     }
#                 },
#                 "toolchain": {
#                     "execution": "parallel",
#                     "pipelines": [
#                         [
#                             "transcriber",
#                             "llm",
#                             "synthesizer"
#                         ]
#                     ]
#                 }
#             }
#         ]


class Assistant:
    def __init__(self, name = "trial_agent"):
        self.name = name

    def add_task(self, task_type, llm_agent, input_handler = None, output_handler = None, transcriber = None, synthesizer = None, enable_textual_input = False):
        pipelines = []
        toolchain_args = {}
        tools_config_args = {}
        toolchain_args['execution'] = 'parallel'
        toolchain_args['pipelines'] = pipelines
        tools_config_args['llm_agent'] = llm_agent
        tools_config_args['input_handler'] = input_handler
        if transcriber is None:
            pipelines.append(["llm"])
            tools_config_args['transcriber'] = transcriber
        
        pipeline = ["transcriber", "llm"]
        if synthesizer is not None:
            pipeline.append("synthesizer") 
            tools_config_args["synthesizer"] = synthesizer
        pipelines.append(pipeline)

        if enable_textual_input:
            pipelines.append(["llm"])
        
        toolchain = ToolsChainModel(execution = "parallel", pipelines = pipelines)
        task = Task(tools_config = ToolsConfig(**tools_config_args), toolchain = toolchain, task_type = task_type)
        self.tasks.append(task)


    def execute(self): 
        self.manager = AssistantManager(agent_config, ws= None)
        for message in self.manager.run():
            yield message