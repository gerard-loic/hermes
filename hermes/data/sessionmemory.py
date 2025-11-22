from cachetools import TTLCache
import time
import os
from hermes.utils.uid import Uid
from hermes.data.config import Config

class SessionMemory:
    @staticmethod
    def init():
        # Cache de 100 éléments max, TTL de 300 secondes
        SessionMemory.cache = TTLCache(maxsize=Config.get("SESSIONMEMORY_MAXSIZE","int"), ttl=Config.get("SESSIONMEMORY_TTL","int"))

    @staticmethod
    def getSession(sessionId:str):
        return SessionMemory.cache.get(str(sessionId))
    
    @staticmethod
    def createSession(data:dict):
        uid = Uid.create()
        SessionMemory.cache[str(uid)] = data
        return uid
    
    @staticmethod
    def updateSession(sessionId:str, data:dict):
        SessionMemory.cache[str(sessionId)] = data