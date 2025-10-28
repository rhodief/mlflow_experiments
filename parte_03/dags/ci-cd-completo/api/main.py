# source venv/bin/activate
# pip install onnxruntime

import pickle
import onnxruntime as rt
import numpy
import time

from fastapi import FastAPI
from pydantic import BaseModel

class Monitor:
    def __init__(self):
        self.eventos_latencia = list()
        self.quantidade_predicoes = 0

    def registrar_latencia(self, latencia):
        self.eventos_latencia.append(latencia)
        self.quantidade_predicoes += 1

    def latencia_media(self):
        if len(self.eventos_latencia) == 0:
            return 0
        
        return sum(self.eventos_latencia) / len(self.eventos_latencia)

class InstanciaStudent(BaseModel):
    academic_stage: int
    peer_pressure: int
    academic_pressure_from_home: int
    study_environment: int
    coping_strategies: int
    academic_competition: int
    bad_habits: int

MONITOR = Monitor()
APP = FastAPI()

with open("scaler.pkl", 'rb') as arquivo:
    SCALER = pickle.loads(arquivo.read())

SESSION = rt.InferenceSession("aula-16-10-25.onnx")
INPUT_NAME = SESSION.get_inputs()[0].name
LABEL_NAME = SESSION.get_outputs()[0].name

@APP.get("/")
async def hello():
    return {"mensagem": "Seja bem-vindo(a)!"}

@APP.get("/metricas")
async def metricas():
    global MONITOR
    return {"latencia_media": MONITOR.latencia_media(), "quantidade_predicoes": MONITOR.quantidade_predicoes}

@APP.post("/predict")
async def predict(instancia: InstanciaStudent):
    global MONITOR
    data = instancia.dict()

    academic_stage = data['academic_stage']
    peer_pressure = data['peer_pressure']
    academic_pressure = data['academic_pressure_from_home']
    study_environment = data['study_environment']

    inicio = time.perf_counter()
    X = numpy.array([[academic_stage, peer_pressure, academic_pressure, study_environment]])
    X_transform = SCALER.transform(X).astype(numpy.float32)

    pred = SESSION.run([LABEL_NAME], {INPUT_NAME: X_transform.reshape(1, 4)})
    pred = int(pred[0][0])
    latencia = time.perf_counter() - inicio

    MONITOR.registrar_latencia(latencia)

    if pred == 1:
        return "Ausência de estresse"
    elif pred == 2:
        return "Pouco estresse"
    elif pred == 3:
        return "Estresse moderado"
    elif pred == 4:
        return "Estresse elevado"
    elif pred == 5:
        return "Necessita atenção"
    