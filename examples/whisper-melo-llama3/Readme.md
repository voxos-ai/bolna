# Bolna With MeloTTS and WhisperASR
Introducing our Dockerized solution! Seamlessly merge [Bolna](https://github.com/bolna-ai/bolna) with [Whisper ASR](https://github.com/bolna-ai/streaming-whisper-server) and [Melo TTS](https://github.com/anshjoseph/MiloTTS-Server) for telephone provider we use Twillo and for tunning we use ngrok. This is docker compose by which you can host bolna server Whisper ASR, Melo TTS together in cloud just by clone this repo  and follow these simple steps to deploy ,but before that you have to make sure that you have [docker](https://docs.docker.com/engine/install/) and [docker compose](https://docs.docker.com/compose/install/) and make a .env file refer to .env-sample and also put ngrok auth token in ngrok-config.yml file


### Start Serices
```shell
docker compose up -d
```
the output something like this
![alt text](./img/docker_up.png "docker compose up -d")

note: make sure that your all service were runing

`let assume your server IP is 192.168.1.10`

### Creating Agent
for creating agent you have to execute following command mention below
```shell
curl --location 'http://192.168.1.10:5001/agent' \
--header 'Content-Type: application/json' \
--data '{
    "agent_config": {
        "agent_name": "Test agent with a transfer call prompt",
        "agent_welcome_message": "Hey how are you!",
        "tasks": [
            {
                "tools_config": {
                    "output": {
                        "format": "wav",
                        "provider": "twilio"
                    },
                    "input": {
                        "format": "wav",
                        "provider": "twilio"
                    },
                    "synthesizer": {
                        "provider": "melotts",
                        "provider_config": {
                        "voice": "Alex",
                        "sample_rate": 8000,
                        "sdp_ratio" : 0.2,
                        "noise_scale" : 0.6,
                        "noise_scale_w" :  0.8,
                        "speed" : 1.0
                        },
                        "stream": true,
                        "buffer_size": 123,
                        "audio_format": "pcm"
                    },
                    "llm_agent": {
                        "max_tokens": 100.0,
                        "presence_penalty": 0.0,
                        "summarization_details": null,
                        "base_url": null,
                        "extraction_details": null,
                        "top_p": 0.9,
                        "agent_flow_type": "streaming",
                        "request_json": false,
                        "routes": null,
                        "min_p": 0.1,
                        "frequency_penalty": 0.0,
                        "stop": null,
                        "provider": "openai",
                        "top_k": 0.0,
                        "temperature": 0.2,
                        "model": "gpt-3.5-turbo",
                        "family": "openai",
                        "extra_config": null
                    },
                    "transcriber": {
                        "encoding": "linear16",
                        "language": "en",
                        "model": "whisper",
                        "stream": true,
                        "modeltype":"distil-large-v3",
                        "keywords":"ansh,joseph,hola",
                        "task": "transcribe",
                        "provider":"whisper"
                    },
                    "api_tools": {
                        "tools": [
                            {
                                "name": "check_availability_of_slots",
                                "description": "Fetch the available free slots of appointment booking before booking the appointment",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "startTime": {
                                            "type": "string",
                                            "description": "It is an ISO FORMATTED DATE on which time user is available (convert it automatically to hr:min such as 3:30 PM is 15:30)"
                                        },
                                        "endTime": {
                                            "type": "string",
                                            "description": "It is an ISO FORMATTED DATE. endDate is always 15 minutes more than startDate always i.e. increment one day to the startDate. such if startDate is 2024-05-15T16:30:00.000Z then endDate is 2024-05-15T16:45:00.000Z"
                                        },
                                        "eventTypeId": {
                                            "type": "integer",
                                            "description": "it is the type of event. For women'\''s haircut use event type id as 814889 and for men'\''s haircut use 798483 "
                                        }
                                    },
                                    "required": [
                                        "startTime",
                                        "eventTypeId",
                                        "endTime"
                                    ]
                                }
                            },
                            {
                                "name": "book_appointment",
                                "description": "Use this tool to book an appointment with given details and save the appointment in the calendar.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "name of the person."
                                        },
                                        "email": {
                                            "type": "string",
                                            "description": "email name of the person."
                                        },
                                        "address": {
                                            "type": "string",
                                            "description": "address name of the person."
                                        },
                                        "preferred_date": {
                                            "type": "string",
                                            "description": "Preferred date provided by the person, ask the date when user wants to book an appointment such as tomorrow or day after tomorrow. and convert the user'\''s response into a python readable format i.e. yyyy-mm-dd"
                                        },
                                        "preferred_time": {
                                            "type": "string",
                                            "description": "Preferred time provided by the person, ask the time when users wants to book an appointment such as 9am or 10:30am and convert it to python readable format for example if users said 9am then it is 09:00 or 1:30PM then it 13:30 i.e. in hr:min "
                                        },
                                        "timezone": {
                                            "type": "string",
                                            "description": "fetch timezone by yourself, don'\''t ask user, find timezone based on the address provided by the user"
                                        },
                                        "eventTypeId": {
                                            "type": "integer",
                                            "description": "it is the type of event. For women'\''s haircut use event type id as 814889 and for men'\''s haircut use 798483 "
                                        }
                                    },
                                    "required": [
                                        "name",
                                        "email",
                                        "address",
                                        "eventTypeId",
                                        "preferred_date",
                                        "preferred_time",
                                        "timezone"
                                    ]
                                }
                            },
                            {
                                "name": "transfer_call",
                                "description": "Transfer calls to human agent if user asks to transfer the call or if it'\''s not in your instructions on how to deal with a certain question user is asking. Call can be routed either to '\''hair_dresser'\'' or a '\''recruiter'\''. Make sure you have this information. It can only be either of the two",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "humanAgentType": {
                                            "type": "string",
                                            "description": "This is the department the call should be routed to. Make sure we know the user has mentioned if they want the call to be routed to '\''hair_dresser'\'' or a '\''recruiter'\'' before sending. It can only be either of the two"
                                        }
                                    },
                                    "required": [
                                        "humanAgentType"
                                    ]
                                }
                            }
                        ],
                        "tools_params": {
                            "book_appointment": {
                                "method": "POST",
                                "param": "{\"responses\":{\"name\":\"%(name)s\",\"email\":\"%(email)s\",\"location\":{\"optionValue\":\"\",\"value\":\"inPerson\"}},\"start\":\"%(preferred_date)sT%(preferred_time)s:00.000Z\",\"eventTypeId\":%(eventTypeId)d,\"timeZone\":\"%(timezone)s\",\"language\":\"en\",\"metadata\":{}}",
                                "url": "https://api.cal.com/v1/bookings?apiKey=API_KEY",
                                "api_token": null
                            },
                            "check_availability_of_slots": {
                                "method": "GET",
                                "param": "{\"eventTypeId\":%(eventTypeId)d,\"startTime\":\"%(startTime)s\",\"endTime\":\"%(endTime)s\"}",
                                "url": "https://api.cal.com/v1/slots?apiKey=API_KEY",
                                "api_token": null
                            },
                            "transfer_call": {
                         "method": "POST",
                "param": "{\"humanAgentType\": \"(humanAgentType)s\"}",
                                "url": "https://webhook.site/3174ce9e-b398-4ec6-bbd5-f3a61473dfce",
                                "api_token": null
                            }
                        }
                    }
                },
                "task_config": {
                    "ambient_noise_track": "office-ambience",
                    "hangup_after_LLMCall": false,
                    "hangup_after_silence": 10.0,
                    "ambient_noise": false,
                    "interruption_backoff_period": 0.0,
                    "backchanneling": false,
                    "backchanneling_start_delay": 5.0,
                    "optimize_latency": true,
                    "incremental_delay": 100.0,
                    "call_cancellation_prompt": null,
                    "number_of_words_for_interruption": 3.0,
                    "backchanneling_message_gap": 5.0,
                    "use_fillers": false
                },
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
                }
            }
        ],
        "agent_type": "Lead Qualification"
    },
    "agent_prompts": {
        "task_1": {
            "system_prompt": "### Agent Description You'\''re an, Mellisa, a helpful agent whose job is to book appointments for Boston Barber Co. at Beacon Hill. There are two type of appointments available - 1. Haircut for men. event id - 798483 2. Appointment for women - 814889 ### About store - Shop is opened Tuesday to Sunday from 9 am to 9pm. - For premium treatment one beer is on the house ### Flow Users can ask you to find available slots & booking for an appointment. You will ask the users about their availability i.e. when they are available the date and time and check if that slot is available or not then you will ask other details as mentioned in function calling and proceed with this information to do the function calling for finding available slots. If slots are available then you must tell only those slots or that slot to user which is or are very close to the user'\''s availability. ### You have access to following functions/tools 1. *check_availability_of_slots* - To check availability of slots from the calendar before booking the appointment. 2. *book_appointment* - Use this tool to book an appointment with given details and save the appointment in the calendar. 3.*transfer_call* - Use this tool to transfer the call to human agent whenever required\n\n### Important instructions 1. MAKE SURE YOU GET ALL THE REQUIRED DETAILS BEFORE DOING A FUNCTION CALL. 2. PLEASE MAKES SURE YOUR RESPONSES ARE GEARED TO BE SYNTHESIZED BY THE SYNTHESISER IN AN EXPRESSIVE WAY. 3. Just speak 1 sentence at a time. If the user says to transfer call MAKE SURE YOU KNOW IF THE CALL SHOULD BE TRANSFERRED FOR haircut or recruitment. It can only be either of the two!"
        }
    }
}'

```
below given is the response 
![alt text](./img/agent_res.png "agent response")
copy the agent_id we have to use in next step

if you want to [Change voice](#change-voice)

### Make call
```shell
curl --location 'http://192.168.1.10:8001/call' \
--header 'Content-Type: application/json' \
--data '{
    "agent_id": "bf2a9e9c-6038-4104-85c4-b71a0d1478c9",
    "recipient_phone_number": "+91XXXXXXXXXX"
}'
```
it gonna give output `Done` for succees

note: if you are using trial account use you register phone no

### Stop Services
```shell
docker compose down
```
![alt text](./img/docker_dw.png "docker compose up -d")


### Changing the voice MeloTTS
<a id="change-voice"></a>
by default we resrtict Melo EN but there were 5 option for voice as mention below
- ['EN-US'](./audio/audio_sample/EN_US.wav) 
- ['EN-BR'](./audio/audio_sample/EN-BR.wav) 
- ['EN-AU'](./audio/audio_sample/EN-AU.wav) 
- ['EN-Default'](./audio/audio_sample/EN-Default.wav) 
- ['EN_INDIA'](./audio/audio_sample/EN_INDIA.wav)

you have to just change the following section mention below
```JSON
"synthesizer": {
            "provider": "melo",
            "provider_config": {
              "voice": "<put your selected voice here>",
              "sample_rate": 8000,
              "sdp_ratio" : 0.2,
              "noise_scale" : 0.6,
              "noise_scale_w" :  0.8,
              "speed" : 1.0
            },
            "stream": true,
            "buffer_size": 123,
            "audio_format": "pcm"
          }
```
and rest of the config gonna be same mention above

### Conservation DENO
This is demo using below prompt to the LLM
```json
"task_1": {
      "system_prompt": "You are assistant at Dr. Sharma clinic you have to book an appointment"
}
```



[chat GPT 3.5 turbo 16k demo](./audio/demo_audio.mp3)

you can give prompt as per your use case 