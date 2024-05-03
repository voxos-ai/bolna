from datetime import datetime, timezone
PREPROCESS_DIR = 'agent_data'

HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA = {
        "extraction_details":{}, 
        "cost_details": {
            "average_transcriber_cost_per_conversation": 0, 
            "average_llm_cost_per_conversation": 0,
            "average_synthesizer_cost_per_conversation": 1.0
        },
        "historical_spread": {
            "number_of_conversations_in_past_5_days": [], 
            "cost_past_5_days": [],
            "average_duration_past_5_days": []
        },
        "conversation_details": { 
            "total_conversations": 0,
            "finished_conversations": 0, 
            "rejected_conversations": 0
        },
        "execution_details": {
            "total_conversations": 0, 
            "total_cost": 0,
            "average_duration_of_conversation": 0
        },
        "last_updated_at": datetime.now(timezone.utc).isoformat()
    }