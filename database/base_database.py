class BaseDatabase:
    def __init__(self, table_name):
        self._init(table_name)

    def _init(self, table_name):
        pass

    def store_agent(self, user_id, sort_key, agent_data):
        pass

    def store_run(self, user_id, agent_id, run_id, run_parameters):
        pass

    def update_run(self, user_id, agent_id, run_id, updated_parameters):
        pass

    def update_table(self, user_id, agent_id, updated_parameters):
        pass

    def get_agents(self, user_id):
        pass

    def get_agent_runs(self, user_id, agent_id):
        pass