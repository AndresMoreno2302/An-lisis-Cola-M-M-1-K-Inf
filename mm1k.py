from mesa import Model, Agent
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import numpy as np
import random

class Cliente(Agent):
    def __init__(self, unique_id, model, arrival_time):
        super().__init__(unique_id, model)
        self.arrival_time = arrival_time
        self.service_start_time = None
        self.departure_time = None

class Servidor(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.busy = False
        self.current_client = None
        self.service_end_time = None

    def step(self):
        if self.busy and self.model.current_time >= self.service_end_time:
            # Cliente termina servicio
            self.current_client.departure_time = self.model.current_time
            self.model.served_clients.append(self.current_client)
            self.current_client = None
            self.busy = False

        if not self.busy and len(self.model.queue) > 0:
            # Atender siguiente cliente
            cliente = self.model.queue.pop(0)
            cliente.service_start_time = self.model.current_time
            service_time = np.random.exponential(1/self.model.mu)
            self.service_end_time = self.model.current_time + service_time
            self.current_client = cliente
            self.busy = True

class MM1KModel(Model):
    def __init__(self, lam=2.0, mu=3.0, K=5, max_time=10000, seed=42):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.lam = lam
        self.mu = mu
        self.K = K
        self.max_time = max_time
        self.current_time = 0.0

        self.schedule = BaseScheduler(self)
        self.servidor = Servidor("Servidor", self)
        self.schedule.add(self.servidor)

        self.queue = []  # cola + servidor (capacidad total K)
        self.served_clients = []
        self.blocked_clients = 0

        # primer arribo
        self.next_arrival = np.random.exponential(1/self.lam)

        self.datacollector = DataCollector(
            model_reporters={
                "NS": lambda m: len(m.queue) + (1 if m.servidor.busy else 0),
                "Nw": lambda m: len(m.queue),
            }
        )

    def step(self):
        # avanzar al próximo evento
        if self.next_arrival < self.servidor.service_end_time if self.servidor.busy else float("inf"):
            # evento de llegada
            self.current_time = self.next_arrival
            if len(self.queue) + (1 if self.servidor.busy else 0) < self.K:
                cliente = Cliente(self.next_id(), self, self.current_time)
                self.queue.append(cliente)
            else:
                self.blocked_clients += 1
            self.next_arrival = self.current_time + np.random.exponential(1/self.lam)
        else:
            # evento de salida
            self.current_time = self.servidor.service_end_time

        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self):
        while self.current_time < self.max_time:
            self.step()

    def compute_metrics(self):
        df = self.datacollector.get_model_vars_dataframe()
        NS = df["NS"].mean()
        Nw = df["Nw"].mean()

        # tiempos de los clientes servidos
        times_in_system = [c.departure_time - c.arrival_time for c in self.served_clients]
        times_in_queue = [c.service_start_time - c.arrival_time for c in self.served_clients]

        TS = np.mean(times_in_system) if times_in_system else 0
        Tw = np.mean(times_in_queue) if times_in_queue else 0

        return {
            "NS": NS,
            "Nw": Nw,
            "TS": TS,
            "Tw": Tw,
            "Bloqueados": self.blocked_clients,
            "Atendidos": len(self.served_clients)
        }

if __name__ == "__main__":
    model = MM1KModel(lam=2, mu=3, K=5, max_time=50000)
    model.run_model()
    results = model.compute_metrics()
    print("Resultados simulación M/M/1/K en Mesa:")
    for k,v in results.items():
        print(f"{k}: {v}")
