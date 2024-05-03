from bolnaTestVersion.input_handlers.telephony import TelephonyInputHandler
from dotenv import load_dotenv
from bolnaTestVersion.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class TwilioInputHandler(TelephonyInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_set=None, connected_through_dashboard=False):
        super().__init__(queues, websocket, input_types, mark_set, connected_through_dashboard)
        self.io_provider = 'twilio'

    async def call_start(self, packet):
        start = packet['start']
        self.call_sid = start['callSid']
        self.stream_sid = start['streamSid']
