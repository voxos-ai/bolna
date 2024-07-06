## Create a RAG based assistant

### Step 1 :Create an assistant via the API (only openai supported yet)

```sh
curl --request POST \
  --url https://api.bolna.dev/assistant \
  --header 'Authorization: Bearer bn-***' \
  --header 'Content-Type: multipart/form-data' \
  --header 'User-Agent: insomnia/8.5.1' \
  --form 'name=A simple insurance agent assistant with human fallback' \
  --form 'instructions=You are a Max Bupa Life Insurance virtual agent. Answer policy questions using policy documents, check available appointment slots, and book appointments. Greet users warmly, provide clear policy details, confirm appointments, and maintain professionalism and user confidentiality.' \
  --form model=gpt-3.5-turbo \
  --form 'tools=[{"name":"check_availability_of_slots","description":"Fetch the available free slots of appointment booking before booking the appointment","parameters":{"type":"object","properties":{"startTime":{"type":"string","description":"It is an ISO FORMATTED DATE on which time user is available (convert it automatically to hr:min such as 3:30 PM is 15:30)"},"endTime":{"type":"string","description":"It is an ISO FORMATTED DATE. endDate is always 15 minutes more than startDate always i.e. increment one day to the startDate. such if startDate is 2024-05-15T16:30:00.000Z then endDate is 2024-05-15T16:45:00.000Z"},"eventTypeId":{"type":"integer","description":"it is the type of event. use event type id as 814889 "}},"required":["startTime","eventTypeId", "endTime"]}},{"name":"book_appointment","description":"Use this tool to book an appointment with given details and save the appointment in the calendar.","parameters":{"type":"object","properties":{"name":{"type":"string","description":"name of the person."},"email":{"type":"string","description":"email name of the person."},"address":{"type":"string","description":"address name of the person."},"preferred_date":{"type":"string","description":"Preferred date provided by the person, ask the date when user wants to book an appointment such as tomorrow or day after tomorrow. and convert the user'\''s response into a python readable format i.e. yyyy-mm-dd"},"preferred_time":{"type":"string","description":"Preferred time provided by the person, ask the time when users wants to book an appointment such as 9am or 10:30am and convert it to python readable format for example if users said 9am then it is 09:00 or 1:30PM then it 13:30 i.e. in hr:min "},"timezone":{"type":"string","description":"fetch timezone by yourself, don'\''t ask user, find timezone based on the address provided by the user"},"eventTypeId":{"type":"integer","description":"it is the type of event. use event type id as 814889"}},"required":["name","email","address","eventTypeId","preferred_date","preferred_time","timezone"]}}, {"name":"transfer_call","description":"Transfer calls to human agent if user asks to transfer the call or if it'\''s not in your instructions on how to deal with a certain question user is asking"}]' \
  --form 'file=@/Users/marmikpandya/Desktop/indie-hacking/Health Companion-Health Insurance Plan_GEN617.pdf'
```

### Step 2 - Create an assistant

```json
{
	"agent_config": {
	"agent_name": "Openai assistant agent agent with call transfer",
	"agent_status": "processed",
	"agent_welcome_message": "This call is being recorded for quality assurance and training. Please speak now.",
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
					"audio_format": "wav",
					"provider": "deepgram",
					"stream": true,
					"caching": true,
					"provider_config": {
						"voice": "Asteria",
						"model": "aura-asteria-en"
					},
					"buffer_size": 100.0
				},
				"llm_agent": {
					"agent_flow_type": "openai_assistant",
					"extra_config": {
						"name": "a Insurance rag agent",
						"assistant_id": "asst_Djy3W6i1n4fZpTMcylSiyVpE"
					}
				},
			 "transcriber": {
				"sampling_rate": 16000,
				"endpointing": 123.0,
				"task": "transcribe",
				"keywords": null,
				"stream": true,
				"provider": "deepgram",
				"model": "nova2",
				"language": "en",
				"encoding": "linear16"
			},
				"api_tools": {
					"tools": "[{\"name\":\"check_availability_of_slots\",\"description\":\"Fetch the available free slots of appointment booking before booking the appointment\",\"parameters\":{\"type\":\"object\",\"properties\":{\"startTime\":{\"type\":\"string\",\"description\":\"It is an ISO FORMATTED DATE on which time user is available (convert it automatically to hr:min such as 3:30 PM is 15:30)\"},\"endTime\":{\"type\":\"string\",\"description\":\"It is an ISO FORMATTED DATE. endDate is always 15 minutes more than startDate always i.e. increment one day to the startDate. such if startDate is 2024-05-15T16:30:00.000Z then endDate is 2024-05-15T16:45:00.000Z\"},\"eventTypeId\":{\"type\":\"integer\",\"description\":\"it is the type of event. For women's haircut use event type id as 814889 and for men's haircut use 798483 \"}},\"required\":[\"startTime\",\"eventTypeId\", \"endTime\"]}},{\"name\":\"book_appointment\",\"description\":\"Use this tool to book an appointment with given details and save the appointment in the calendar.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\",\"description\":\"name of the person.\"},\"email\":{\"type\":\"string\",\"description\":\"email name of the person.\"},\"address\":{\"type\":\"string\",\"description\":\"address name of the person.\"},\"preferred_date\":{\"type\":\"string\",\"description\":\"Preferred date provided by the person, ask the date when user wants to book an appointment such as tomorrow or day after tomorrow. and convert the user's response into a python readable format i.e. yyyy-mm-dd\"},\"preferred_time\":{\"type\":\"string\",\"description\":\"Preferred time provided by the person, ask the time when users wants to book an appointment such as 9am or 10:30am and convert it to python readable format for example if users said 9am then it is 09:00 or 1:30PM then it 13:30 i.e. in hr:min \"},\"timezone\":{\"type\":\"string\",\"description\":\"fetch timezone by yourself, don't ask user, find timezone based on the address provided by the user\"},\"eventTypeId\":{\"type\":\"integer\",\"description\":\"it is the type of event. For women's haircut use event type id as 814889 and for men's haircut use 798483 \"}},\"required\":[\"name\",\"email\",\"address\",\"eventTypeId\",\"preferred_date\",\"preferred_time\",\"timezone\"]}}, {\"name\":\"transfer_call\",\"description\":\"Transfer calls to human agent if user asks to transfer the call or if it's not in your instructions on how to deal with a certain question user is asking\"}]",
					"tools_params": {
						"book_appointment": {
							"method": "POST",
							"param": "{\"responses\":{\"name\":\"%(name)s\",\"email\":\"%(email)s\",\"location\":{\"optionValue\":\"\",\"value\":\"inPerson\"}},\"start\":\"%(preferred_date)sT%(preferred_time)s:00.000Z\",\"eventTypeId\":%(eventTypeId)d,\"timeZone\":\"%(timezone)s\",\"language\":\"en\",\"metadata\":{}}",
							"url": "https://api.cal.com/v1/bookings?apiKey=",
							"api_token": null
						},
						"check_availability_of_slots": {
							"method": "GET",
							"param": "{\"eventTypeId\":%(eventTypeId)d,\"startTime\":\"%(startTime)s\",\"endTime\":\"%(endTime)s\"}",
							"url": "https://api.cal.com/v1/slots?apiKey=",
							"api_token": null
						},
						"transfer_call": {
							"method": "POST",
							"param": null,
							"url": "https://webhook.site/c5dfe764-8888-4be5-a28c-8cfe4d54d475",
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
		},
		{
			"tools_config": {
				"output": null,
				"input": null,
				"synthesizer": null,
				"llm_agent": {
					"max_tokens": 100.0,
					"presence_penalty": 0.0,
					"summarization_details": null,
					"base_url": null,
					"extraction_details": null,
					"top_p": 0.9,
					"agent_flow_type": "streaming",
					"request_json": true,
					"routes": null,
					"min_p": 0.1,
					"frequency_penalty": 0.0,
					"stop": null,
					"provider": "openai",
					"top_k": 0.0,
					"temperature": 0.1,
					"model": "gpt-3.5-turbo-1106",
					"family": "openai"
				},
				"transcriber": null,
				"api_tools": null
			},
			"task_config": {
				"ambient_noise_track": "convention_hall",
				"hangup_after_LLMCall": false,
				"hangup_after_silence": 30.0,
				"ambient_noise": true,
				"interruption_backoff_period": 100.0,
				"backchanneling": false,
				"backchanneling_start_delay": 5.0,
				"optimize_latency": true,
				"incremental_delay": 100.0,
				"call_cancellation_prompt": null,
				"number_of_words_for_interruption": 1.0,
				"backchanneling_message_gap": 5.0,
				"call_terminate": 300
			},
			"task_type": "summarization",
			"toolchain": {
				"execution": "parallel",
				"pipelines": [
					[
						"llm"
					]
				]
			}
		},
		{
			"tools_config": {
				"output": null,
				"input": null,
				"synthesizer": null,
				"llm_agent": {
					"max_tokens": 100.0,
					"presence_penalty": 0.0,
					"summarization_details": null,
					"base_url": null,
					"extraction_details": "slot: Slot booked by user",
					"top_p": 0.9,
					"agent_flow_type": "streaming",
					"request_json": true,
					"routes": null,
					"min_p": 0.1,
					"frequency_penalty": 0.0,
					"stop": null,
					"provider": "openai",
					"top_k": 0.0,
					"temperature": 0.1,
					"model": "gpt-3.5-turbo-1106",
					"family": "openai"
				},
				"transcriber": null,
				"api_tools": null
			},
			"task_config": {
				"ambient_noise_track": "office-ambience",
				"hangup_after_LLMCall": false,
				"hangup_after_silence": 10.0,
				"ambient_noise": true,
				"interruption_backoff_period": 100.0,
				"backchanneling": false,
				"backchanneling_start_delay": 5.0,
				"optimize_latency": true,
				"incremental_delay": 100.0,
				"call_cancellation_prompt": null,
				"number_of_words_for_interruption": 1.0,
				"backchanneling_message_gap": 5.0
			},
			"task_type": "extraction",
			"toolchain": {
				"execution": "parallel",
				"pipelines": [
					[
						"llm"
					]
				]
			}
		}
	],
	"agent_type": "Lead Qualification"
	},
    "agent_prompts": {
        "task_2" : {
            "system_prompt" : "summarize the conversation"
        },
        "task_3" : {
             "system_prompt" : "Extract the user sentiment in json with the key  'sentiment' "
        }
    }
}
```