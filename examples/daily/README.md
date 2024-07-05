# Bolna With Daily
Introducing our Dockerized solution! Seamlessly merge [Bolna](https://github.com/bolna-ai/bolna) with [Daily](https://github.com/daily-co) for websocket connection we use daily and for tunning we use ngrok. This is docker compose by which you can host bolna server Daily together in cloud just by clone this repo  and follow these simple steps to deploy, but before that you have to make sure that you have [docker](https://docs.docker.com/engine/install/) and [docker compose](https://docs.docker.com/compose/install/) and make a .env file refer to .env-sample and also put ngrok auth token in ngrok-config.yml file


### Start Serices
```shell
docker compose up -d
```

note: make sure that your all service were runing

`let assume your server IP is 192.168.1.10`

### Creating Agent
for creating agent you have to execute following command mention below
```shell
curl --location 'http://192.168.1.10:5001/agent' \
--header 'Content-Type: application/json' \
--data '{
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
            "format": "wav",
            "provider": "daily"
          },
          "output": {
            "format": "wav",
            "provider": "daily"
          },
          "llm_agent": {
            "max_tokens": 100,
            "presence_penalty": 0,
            "top_p": 0.9,
            "agent_flow_type": "streaming",
            "request_json": false,
            "min_p": 0.1,
            "frequency_penalty": 0,
            "provider": "openai",
            "top_k": 0,
            "temperature": 0.2,
            "model": "gpt-3.5-turbo",
            "family": "openai"
          },
          "synthesizer": {
            "audio_format": "wav",
            "buffer_size": 150,
            "caching": true,
            "provider": "polly",
            "provider_config": {
              "engine": "neural",
              "language": "en-US",
              "voice": "Danielle"
            },
            "stream": true
          },
          "transcriber": {
            "encoding": "linear16",
            "endpointing": 100,
            "keywords": "",
            "language": "en",
            "model": "nova-2",
            "provider": "deepgram",
            "sampling_rate": 16000,
            "stream": true,
            "task": "transcribe"
          }
        },
        "task_config": {
          "hangup_after_silence": 30
        }
      }
    ]
  },
  "agent_prompts": {
    "task_1": {
      "system_prompt": "You are assistant at Dr. Sharma clinic you have to book an appointment"
    }
  }
}
'

```
below given is the response 
```
{"agent_id":"dcfe02de-bOdf-4589-b15b-64c76f0077d0", "state" : "created" }
```
save / copy the agent_id we have to use in next step

### Connect frontend to the websocket call
1. Connect your frontend with the (’/chat/v1/{agent_id}’) websocket and use the agent_id you saved in above step. It's that easy!
2. Our daily we-socket call will be up and running, so you can start chatting with the agent through your frontend!

### Stop Services
```shell
docker compose down
```

### Conservation DENO
This is demo using below prompt to the LLM
```json
"task_1": {
      "system_prompt": "You are assistant at Dr. Sharma clinic you have to book an appointment"
}
```