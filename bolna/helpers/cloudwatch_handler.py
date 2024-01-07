import asyncio
import logging
from botocore.exceptions import NoCredentialsError
from logging import Handler, LogRecord
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack


async def create_client(service: str, session: AioSession, exit_stack: AsyncExitStack):
    # creates AWS session from system environment credentials & config
    return await exit_stack.enter_async_context(session.create_client(service))


class AsyncCloudWatchLogHandler(Handler):
    def __init__(self, log_group_name, log_stream_name):
        super().__init__()
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name
        #self.loop = asyncio.get_event_loop()

    async def _create_log_stream(self, client):
        try:
            await client.create_log_stream(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name
            )
        except client.exceptions.ResourceAlreadyExistsException:
            pass

    async def emit(self, record: LogRecord):
        session = AioSession()
        try:
            async with AsyncExitStack() as exit_stack:
                async with create_client("logs", session, exit_stack) as client:
                    await self._create_log_stream(client)

                    a = await client.put_log_events(
                        logGroupName=self.log_group_name,
                        logStreamName=self.log_stream_name,
                        logEvents=[
                            {
                                'timestamp': int(record.created * 1000),
                                'message': self.format(record)
                            }
                        ],
                        sequenceToken=None
                    )
                    b = 1
        except NoCredentialsError:
            print("Credentials not available")