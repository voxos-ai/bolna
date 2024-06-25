### With Fillers

Fillers can be used to give an agent time to think or to make sure with expressive TTS like elevenlabs, you're able to ensir elow latency. This is the example task config fot the agent to enable the same. 

```json
   "task_config": {
                    "ambient_noise_track": "office-ambience",
                    "hangup_after_LLMCall": false,
                    "hangup_after_silence": 10.0,
                    "ambient_noise": true,
                    "interruption_backoff_period": 0.0,
                    "backchanneling": true,
                    "backchanneling_start_delay": 5.0,
                    "optimize_latency": true,
                    "incremental_delay": 300.0,
                    "call_cancellation_prompt": null,
                    "number_of_words_for_interruption": 3.0,
                    "backchanneling_message_gap": 5.0,
                    "use_fillers": true
                },
```