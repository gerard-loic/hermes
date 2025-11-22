from hermes.agent.agent import Agent

class AgentsManager:
    @staticmethod
    def init(envName:str):
        AgentsManager.agents = {}
        AgentsManager.envName = envName

    @staticmethod
    def getAgent(name:str):
        if name in AgentsManager.agents:
            return AgentsManager.agents[name]
        return None
    
    @staticmethod
    def addAgent(name:str):
        agent = Agent(name, AgentsManager.envName)
        AgentsManager.agents[name] = agent
        return agent