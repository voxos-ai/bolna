### With Guardrails

You can add guardrails to make sure agent doesn't respond to unwanted queries
```json
  "routes": {
                            "embedding_model": "Snowflake/snowflake-arctic-embed-l",
                            "routes": [
                                {
                                    "route_name": "politics",
                                    "utterances": [
                                        "Are you a Trump supporter",
                                        "How many black people live in my neighborhood",
                                        "Are you a democrat?",
                                        "Are you a republican",
                                        "Are you black?",
                                        "What is the gender split of my society",
                                        "Are you a democrat?",
                                        "Tell me about your political ideologies",
                                        "Who is winning the elections this year?",
                                        "Are there hispanics in the area",
                                        "I do not like democrats",
                                        "I don't want faggots",
                                        "Don't give me homosexuals",
                                        "I need a white hair dresser only"
                                    ],
                                    "response": "Hey, thanks but I don't want to entertain hate speech",
                                    "score_threshold": 0.90
                                }
                            ]
                        },
```