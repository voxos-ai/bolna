## Local docker setup

A basic local setup includes usage of [Twilio](local_setup/telephony_server/twilio_api_server.py) or [Plivo](local_setup/telephony_server/plivo_api_server.py) for telephony. We have dockerized the setup in `local_setup/`. One will need to populate an environment `.env` file from `.env.sample`.

The setup consists of four containers:

1. Telephony web server:
   * Choosing Twilio: for initiating the calls one will need to set up a [Twilio account](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account)
   * Choosing Plivo: for initiating the calls one will need to set up a [Plivo account](https://www.plivo.com/)
2. Bolna server: for creating and handling agents 
3. `ngrok`: for tunneling. One will need to add the `authtoken` to `ngrok-config.yml`
4. `redis`: for persisting agents & prompt data

Use docker to build the images using `.env` file as the environment file and run them locally
1. `docker-compose build --no-cache <twilio-app | plivo-app>`: rebuild images for `twilio-app` or `plivo-app` as defined in the `docker-compose.yml`.
2. `docker-compose up`: run the build images

Once the docker containers are up, you can now start to create your agents and instruct them to initiate calls.



## Creating your agent and invoking calls
Once you have the above docker setup and running, you can create agents and initiate calls.
1. Use the below payload to create an Agent via `http://localhost:5001/agent`

<details>
<summary>Agent Payload</summary><br>

```yaml
{
    "agent_config": {
        "agent_name": "Alfred",
        "agent_type": "other",
        "agent_welcome_message": "Welcome",
        "tasks": [
            {
                "task_type": "conversation",
                "toolchain": {
                    "execution": "parallel",
                    "pipelines": [
                        [
                            "transcriber",
                            "llm",
                            "synthesizer"
                        ]
                    ]
                },
                "tools_config": {
                    "input": {
                        "format": "pcm",
                        "provider": "twilio"
                    },
                    "llm_agent": {
                        "agent_flow_type": "streaming",
                        "provider": "openai",
                        "request_json": true,
                        "model": "gpt-3.5-turbo-16k",
                        "use_fallback": true
                    },
                    "output": {
                        "format": "pcm",
                        "provider": "twilio"
                    },
                    "synthesizer": {
                        "audio_format": "wav",
                        "provider": "elevenlabs",
                        "stream": true,
                        "provider_config": {
                            "voice": "Meera - high quality, emotive",
                            "model": "eleven_multilingual_v2",
                            "voice_id": "TTa58Hl9lmhnQEvhp1WM"
                        },
                        "buffer_size": 100.0
                    },
                    "transcriber": {
                        "encoding": "linear16",
                        "language": "en",
                        "model": "deepgram",
                        "stream": true
                    }
                },
                "task_config": {
                    "hangup_after_silence": 30.0
                }
            }
        ]
    },
    "agent_prompts": {
        "task_1": {
            "system_prompt": "Ask if they are coming for party tonight"
        }
    }
}
```
</details>

2. The response of the previous API will return a uuid as the `agent_id`. Use this `agent_id` to initiate a call via the telephony server running on `8001` port (for Twilio) or `8002` port (for Plivo) at `http://localhost:8001/call`

<details>
<summary>Call Payload</summary><br>

```yaml
{
    "agent_id": "4c19700b-227c-4c2d-8bgf-42dfe4b240fc",
    "recipient_phone_number": "+19876543210",
}
```
</details>