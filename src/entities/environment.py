import datetime as dt
from typing import Optional


class Environment:
    def __init__(
        self,
        start_date: dt.date,
        current_date: dt.date,
        agents: list,
        timetable: Optional[dict] = None,
        contact_tracing_max_currently_traced = None,
    ):
        self.start_date: dt.date = start_date
        self.current_date: dt.date = current_date
        self.tick = 0
        self.weekday: int = self.current_date.weekday()
        self.timetable: Optional[dict] = timetable
        self.current_interventions: Optional[dict] = None
        self.n_currently_infected = 0
        self.n_ever_infected = 0
        self.n_ever_infected_and_prio = 0
        self.n_currently_traced = 0
        self.n_currently_infected_and_prio = 0
        self.new_tracings_today = 0
        self.new_tracings_last7days = []
        self.agents_dict = {agent.id: agent for agent in agents}
        self.sum_tracings_last7days = 0
        self.contact_tracing_max_currently_traced = contact_tracing_max_currently_traced

        self.new_potential_contact_tracers = []

        self.date_data = []

        # assert, that the dates in the timetable are in ascending order
        if self.timetable is not None:
            for i, date in enumerate(timetable):
                if i > 0:
                    assert (
                        date > list(timetable.keys())[i - 1]
                    ), "The dates in the timetable are not in ascending order."

    def update_current_interventions(self):
        if self.timetable is not None:
            for date_key in self.timetable:
                if date_key <= self.current_date:
                    self.current_interventions = self.timetable[date_key]

    def step(self):
        self.new_tracings_last7days.append(self.new_tracings_today)
        assert len(self.new_tracings_last7days) <= 7
        self.sum_tracings_last7days = sum(self.new_tracings_last7days)

        self.date_data.append(
            {
                "date": self.current_date,
                "n_currently_infected": self.n_currently_infected,
                "n_currently_traced": self.n_currently_traced,
                "n_currently_infected_and_prio": self.n_currently_infected_and_prio,
                "new_tracings_today": self.new_tracings_today,
                "sum_tracings_last7days": self.sum_tracings_last7days,
            }
        )

        # step one day forward
        self.tick += 1
        self.current_date = self.current_date + dt.timedelta(days=1)
        self.weekday = self.current_date.weekday()
        self.new_tracings_today = 0
        if len(self.new_tracings_last7days) == 7:
            self.new_tracings_last7days = self.new_tracings_last7days[1:]

        # get the current plan of measures
        if self.timetable is not None:
            self.update_current_interventions()
