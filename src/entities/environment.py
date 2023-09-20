import datetime as dt
from typing import Optional


class Environment:
    def __init__(
        self, 
        current_date: dt.date,
        agents: list,
        timetable: Optional[dict]=None,
        ):
        self.current_date: dt.date = current_date
        self.tick = 0
        self.weekday: int = self.current_date.weekday()
        self.timetable: Optional[dict] = timetable
        self.current_interventions: Optional[dict] = None
        self.n_currently_infected = 0
        self.n_ever_infected = 0
        self.contact_tracing_currently_traced = 0
        self.agents_dict = {agent.id: agent for agent in agents}

        # assert, that the dates in the timetable are in ascending order
        if self.timetable is not None:
            for i, date in enumerate(timetable):
                if i > 0:
                    assert date > list(timetable.keys())[i-1], "The dates in the timetable are not in ascending order."

    
    def update_current_interventions(self):
        if self.timetable is not None:
            for date_key in self.timetable:
                if date_key <= self.current_date:
                    self.current_interventions = self.timetable[date_key]
    
    def step(self):
        # step one day forward
        self.tick += 1
        self.current_date = self.current_date + dt.timedelta(days=1)
        self.weekday = self.current_date.weekday()
        
        # get the current plan of measures
        if self.timetable is not None:
            self.update_current_interventions()