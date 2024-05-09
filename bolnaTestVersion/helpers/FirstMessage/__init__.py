from .provider import PollySynthesizer
from .provider import PROVIDER
import asyncio





class FirstVoice():
    def __init__(self,synthesizer_config:dict,synthesizer_name:str,first_message:str) -> None:
        self.synthesizer = PROVIDER.get(synthesizer_name)(synthesizer_config.get('voice'),synthesizer_config.get("language"))
        self.audio:bytes = None
        self.output_handle = None
        self.stream_sid = None
        self.first_message = first_message
        self.once_done:bool = False

    def setOutputHandel(self,output_handle):
        self.output_handle = output_handle
    async def genrateAudio(self):
        self.audio = await self.synthesizer.synthesize(self.first_message)
    async def ready(self,stream_sid:str):
        if not self.once_done:
            self.stream_sid = stream_sid
            # if self.audio == None:
            #     await self.genrateAudio()
            await self.putFirstMessage()
            self.once_done = True

    def genrate_meta_info(self):
        if self.stream_sid != None:
            return {"data":self.audio,"meta_info":{'stream_sid':self.stream_sid,'request_id':0,'type':'audio'}}
        return dict()
    
    async def putFirstMessage(self):
        if self.stream_sid is not None and self.output_handle is not None:
            await self.output_handle.handle(self.genrate_meta_info())

        