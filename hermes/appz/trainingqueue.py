import queue
import json

class TrainingQueue:
    @staticmethod
    def init():
        TrainingQueue.training_queues = {}

    @staticmethod
    def add(sessionId:str):
        TrainingQueue.training_queues[sessionId] = queue.Queue()

    @staticmethod
    def get(sessionId:str):
        return TrainingQueue.training_queues.get(sessionId)

    @staticmethod
    def delete(sessionId:str):
        del TrainingQueue.training_queues[sessionId]

    @staticmethod
    def sendMessage(sessionId:str, message:dict):
        if sessionId != None:
            q = TrainingQueue.get(sessionId)
            q.put(json.dumps(message))

    @staticmethod
    def closeSession(sessionId:str):
        if sessionId != None:
            q = TrainingQueue.get(sessionId)
            q.put('DONE')