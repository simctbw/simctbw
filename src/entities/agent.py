from typing import Dict, List, Optional, Union
import datetime as dt
import random
import collections
import pandas as pd

from src.entities.environment import Environment


SUSCEPTIBLE = 0
INFECTED = 10  # = exposed
INFECTIOUS = 20  # = presymptomatic infectious
ILL_SYMPTOMATIC = 31
ILL_ASYMPTOMATIC = 32
ICU = 40
RECOVERED = 51

INFECTIOUS_STATES = [
    INFECTIOUS,
    ILL_SYMPTOMATIC,
]


class Agent:
    """
    Represents the people in the simulated world.
    An agent can visit multiple locations during the day.
    The other agents an agent encounters at an location, are recorded in its contact diary.
    Besides the default attributes, an agent will get its attributes from the micro-data (e.g. GSOEP).
    """

    def __init__(self, id: int) -> None:
        # population-specific agent identifier
        self.id: int = id

        # model (is set by the pop_maker)
        self.model = None

        # population-specific household identifier
        self.household_id: Optional[int] = None

        # a dict of dicts containing information about the locations the agent is assigned to
        self.location_dict: Dict[Dict] = {}

        # the location-object the agent returns to when it does not visit any other location
        self.home_location: Optional["Location"] = None

        # A list of dicts. Each dict contains information about an encounter with another agent at an specific location in a specific time step.
        self.contact_diary: List[dict] = collections.deque()

        # the same as the contact diary, but its content is deleted regulary
        self.temp_contact_diary = collections.deque()

        # a list of the household members of the agent, including the agent itself
        self.household_members: List[Agent] = []
        self.df_daily_location_contacts: Optional[pd.DataFrame] = None
        self.df_daily_contacts: Optional[pd.DataFrame] = None

        # a list of agents which the agent met at least once during the simulation
        self.agents_ever_met: List[str] = []

        # the number of agents which the agent met at least once during the simulation
        self.n_agents_ever_met: int = 0

        # a list containing the numbers of "hours" the agent is not at home
        self.list_hours_not_at_home: List[float] = []

        # the average amount of hours not at home per time step (day)
        self.mean_hours_not_at_home: Optional[float] = None

        # the highest amount of hours not at home per time step (day)
        self.max_hours_not_at_home: Optional[float] = None

    def add_contact_to_temp_diary(
        self,
        date: dt.date,
        agent_j: "Agent",
        weight: Union[float, int],
        location: str,
        infection_probability_category: Optional[str] = None,
    ) -> None:
        """Creates an entry of the temporary contact diary."""

        assert agent_j is not self

        # create and append the contact diary entry
        self.temp_contact_diary.append(
            {
                "date": date,
                "id_j": agent_j.id,
                "age_j": agent_j.age,
                "weight": weight,
                "location": location,
                "infection_probability_category": infection_probability_category,
            }
        )

        # save id in unique contact list
        if agent_j.id not in self.agents_ever_met:
            self.agents_ever_met.append(agent_j.id)
            self.n_agents_ever_met += 1

    def clean_temp_diary(self, n_days=1, environment=None):
        """(Partially) Deletes the content of the temporary contact diary."""

        # Reset the whole temporary contact diary
        if n_days <= 1:
            self.temp_contact_diary = collections.deque()

        # Delete only those entries which are too far in the past
        else:
            self.temp_contact_diary = [
                entry
                for entry in self.temp_contact_diary
                if (environment.current_date - entry["date"]).days < n_days
            ]

    def add_contact_to_diary(
        self,
        date: dt.date,
        agent_j: "Agent",
        weight: Union[float, int],
        location: str,
    ) -> None:
        """Adds an entry about the date, location, weight, duration etc. of an encountering to the contact diary."""

        assert agent_j is not self

        self.contact_diary.append(
            {
                "date": date,
                "id_j": agent_j.id,
                "age_j": agent_j.age,
                "weight": weight,
                "location": location,
            }
        )

    def get_df_daily_contacts(
        self, location: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        if len(self.contact_diary) > 0:
            df = pd.DataFrame(self.contact_diary)

            if location is not None:
                df = df.loc[df["location"] == location, :]

            df["id_i"] = self.id
            df["age_i"] = self.age
            df = df.groupby(["date", "id_j"], as_index=False).agg(
                {
                    "weight": "sum",
                    "date": "first",
                    "id_i": "first",
                    "id_j": "first",
                    "age_i": "first",
                    "age_j": "first",
                }
            )

            self.df_daily_contacts = df
            return self.df_daily_contacts
        else:
            return None

    def visit(self, environment: Environment) -> None:
        """
        Agent visits all its locations and writes its visit in the locations' daily guest books,
        if the corresponding visiting conditions are met.
        """

        # the number of hours the agent spends at other locations than home
        hours_not_at_home = 0

        # for each location category the agent has in its dictionary of locations
        for location_category in self.location_dict:
            if location_category != "home":
                # get the condition that determines whether the agent visits this location today
                cond = self.location_dict[location_category]["visit_condition"]

                # if there is no condition or the condition is already True or a given condition(environment, agent) that was checked is True
                if cond is None or cond is True or cond(environment, self):
                    # choose a location-object from the category (most of the time it is just one object)
                    location = random.choice(
                        self.location_dict[location_category]["objects"]
                    )

                    # get the number of hours the agent spends at the location
                    hours = self.location_dict[location_category]["n_hours_per_visit"]

                    # only if the agent spends time at the location, it gets registered as a visitor at the location
                    if hours > 0:
                        location.visitors_of_the_day.append([self, hours])
                        hours_not_at_home += hours

        self.list_hours_not_at_home.append(hours_not_at_home)

        # if the agent has any hours at home, register him as an visitor at the home location
        hours_at_home = 24 - hours_not_at_home
        if hours_at_home > 0:
            self.home_location.visitors_of_the_day.append([self, hours_at_home])


class PandemicAgent(Agent):
    def __init__(self, id) -> None:
        super().__init__(id)

        # infection status
        self.infection_status = 0

        # has the agent ever been infected during the simulation?
        self.ever_infected = False

        # Has the agent ever been symptomatic ill during the simulation?
        self.ever_infected_with_symptoms = False

        # time steps / dates when something happend to the infection status
        self.tick_infection: Optional[int] = None
        self.date_infection: Optional[dt.date] = None
        self.date_infectious: Optional[dt.date] = None
        self.date_ill: Optional[dt.date] = None
        self.date_symptomatic: Optional[dt.date] = None
        self.date_icu: Optional[dt.date] = None
        self.date_recovered: Optional[dt.date] = None

        # the agent that infected this agents
        self.infection_source: PandemicAgent = None

        # the location type where the infection has happened
        self.infection_location: str = None

        # the number of agents that had been infected before this agent has been infected
        self.n_ever_infected_when_infected: Optional[int] = None

        # a list of agents this agent has infected
        self.infected_agents: list[PandemicAgent] = []

        # the number of agents this agent infected directly
        self.n_directly_infected_agents: int = 0

        # the number of household members this agent infected directly
        self.n_directly_infected_household_members: int = 0

        # the number of agents in the infection chain from patient0 to this agent
        self.len_pre_infection_chain: int = None

        # the number of directly and indirectly caused infections by this agent
        self.len_post_infection_chain: int = 0

        # the number of directly and indirectly caused infections above the age of 79
        self.len_post_infection_chain_80: int = 0

        # the number of directly and indirectly caused infections above the age of 59
        self.len_post_infection_chain_60: int = 0

        # the number of directly and indirectly caused hospitalizations
        self.len_post_icu_chain: int = 0

        # will this agent develop symptoms when infected?
        self.will_develop_symptoms: bool = False

        # is this agent one of the intial infections?
        self.patient0: bool = False

        # is this agent currently locked in?
        self.lockdown: bool = False

        # a list in which contacts are saved to trace them eventually
        self.contact_tracing_diary = []
        self.contact_tracing_date = None
        self.contact_traced = False
        self.contact_tracer = False
        self.n_times_contact_traced = 0
        self.n_contacts_traced = 0
        self.n_times_contact_tracing = 0
        self.contact_tracing_dates_list = []

        self.vaccinated = False

        self.quarantined = False

    def inform_pre_infection_chain_about_infection(self) -> None:
        if self.infection_source is not None:
            self.infection_source.len_post_infection_chain += 1
            self.infection_source.inform_pre_infection_chain_about_infection()

    def inform_pre_infection_chain_about_infection_60(self) -> None:
        if self.infection_source is not None:
            self.infection_source.len_post_infection_chain_60 += 1
            self.infection_source.inform_pre_infection_chain_about_infection_60()

    def inform_pre_infection_chain_about_infection_80(self) -> None:
        if self.infection_source is not None:
            self.infection_source.len_post_infection_chain_80 += 1
            self.infection_source.inform_pre_infection_chain_about_infection_80()

    def inform_pre_infection_chain_about_icu(self) -> None:
        if self.infection_source is not None:
            self.infection_source.len_post_icu_chain += 1
            self.infection_source.inform_pre_infection_chain_about_icu()

    def adjust_p_by_age(self) -> float:
        # source: kerr 2021

        if 0 <= self.age < 10:
            return 0.34

        elif 10 <= self.age < 20:
            return 0.67

        elif 20 <= self.age < 30:
            return 1

        elif 30 <= self.age < 40:
            return 1

        elif 40 <= self.age < 50:
            return 1

        elif 50 <= self.age < 60:
            return 1

        elif 60 <= self.age < 70:
            return 1

        elif 70 <= self.age < 80:
            return 1.24

        elif 80 <= self.age < 90:
            return 1.47

        elif 90 <= self.age:
            return 1.47

        else:
            raise ValueError("The age of the agent is smaller than 0.")

    def probably_go_to_icu(self) -> bool:
        """
        Determines whether an agent will be hospitalized.
        Source: Kerr 2021
        """

        if 0 <= self.age < 10:
            return True if random.random() < 0.00050 else False

        elif 10 <= self.age < 20:
            return True if random.random() < 0.00165 else False

        elif 20 <= self.age < 30:
            return True if random.random() < 0.00720 else False

        elif 30 <= self.age < 40:
            return True if random.random() < 0.02080 else False

        elif 40 <= self.age < 50:
            return True if random.random() < 0.03430 else False

        elif 50 <= self.age < 60:
            return True if random.random() < 0.07650 else False

        elif 60 <= self.age < 70:
            return True if random.random() < 0.13280 else False

        elif 70 <= self.age < 80:
            return True if random.random() < 0.20655 else False

        elif 80 <= self.age < 90:
            return True if random.random() < 0.24570 else False

        elif 90 <= self.age:
            return True if random.random() < 0.24570 else False

        else:
            raise ValueError("The age of the agent is smaller than 0.")

    def has_symptoms(self) -> bool:
        """
        Determines whether the agent will develop symptoms when infected.
        Source: Kerr 2021
        """

        if 0 <= self.age < 10:
            return True if random.random() < 0.50 else False

        elif 10 <= self.age < 20:
            return True if random.random() < 0.55 else False

        elif 20 <= self.age < 30:
            return True if random.random() < 0.60 else False

        elif 30 <= self.age < 40:
            return True if random.random() < 0.65 else False

        elif 40 <= self.age < 50:
            return True if random.random() < 0.70 else False

        elif 50 <= self.age < 60:
            return True if random.random() < 0.75 else False

        elif 60 <= self.age < 70:
            return True if random.random() < 0.80 else False

        elif 70 <= self.age < 80:
            return True if random.random() < 0.85 else False

        elif 80 <= self.age < 90:
            return True if random.random() < 0.90 else False

        elif 90 <= self.age:
            return True if random.random() < 0.90 else False

        else:
            raise ValueError("The age of the agent is smaller than 0.")

    def calculate_transition_dates(self, duration_infectious=2) -> None:
        """
        Determines the dates when the agent will change the stage of infection.
        Source: Klüsener et al. (2020): Forecasting intensive care unit demand during the COVID-19 pandemic: A spatial age-structured microsimulation model.
        """

        DURATION_EXPOSED = 3 + random.randint(-1, 1)
        DURATION_INFECTIOUS = duration_infectious + random.randint(-1, 1)
        DURATION_ILL = 8 + random.randint(-1, 1)
        DURATION_ICU = 14 + random.randint(-1, 1)

        # exposed -> presymptomatic infectious
        self.date_infectious = self.date_infection + dt.timedelta(days=DURATION_EXPOSED)

        # presymptomatic infectious -> ill (asymptomatic or symptomatic)
        self.date_ill = self.date_infectious + dt.timedelta(days=DURATION_INFECTIOUS)

        # symptomatic ill?
        self.will_develop_symptoms = self.has_symptoms()

        # if agent is symptomatic ill
        if self.will_develop_symptoms:
            # if agent has to be hospitalized
            if self.probably_go_to_icu():
                # symptomatic ill -> hospitalized
                self.date_icu = self.date_ill + dt.timedelta(days=DURATION_ILL)

                # hospitalized -> recovered
                self.date_recovered = self.date_icu + dt.timedelta(days=DURATION_ICU)

            else:
                # symptomatic ill -> hospitalized
                self.date_recovered = self.date_ill + dt.timedelta(days=DURATION_ILL)

        else:
            # asymptomatic ill -> recovered
            self.date_recovered = self.date_ill + dt.timedelta(days=DURATION_ILL)

    def start_contact_tracing(
        self, 
        contact_tracing_detected_infections, 
        environment, 
        ct_start_date,
    ) -> bool:
        trace = False

        # if environment.current_date >= environment.start_date:
        if environment.current_date >= ct_start_date:
            if (
                contact_tracing_detected_infections in ["ill", "ill_symptomatic"]
                and self.infection_status == ILL_SYMPTOMATIC
            ):
                trace = True
            elif (
                contact_tracing_detected_infections == "ill"
                and self.infection_status == ILL_ASYMPTOMATIC
            ):
                trace = True
        return trace

    def update_infection(
        self,
        environment: Environment,
        contact_tracing_contact_selection_mode: Optional[str] = None,
        contact_tracing_contact_selection_frac: Optional[float] = None,
        contact_tracing_detected_infections: Optional[str] = None,
        contact_tracing_allowed=True,
        n_days_contact_tracing_isolation=10,
        ct_start_date=None,
        duration_infectious=2,
    ) -> None:
        """
        Updates the infection status of infected agents.
        """

        # if the agent is currently isolated due to having contact to an infected agent
        if self.contact_traced:
            # check if the time of isolation is over
            if (
                environment.current_date - self.contact_tracing_date
            ).days > n_days_contact_tracing_isolation + random.randint(-1, 1):
                self.contact_traced = False
                self.contact_tracing_date = None
                environment.n_currently_traced -= 1

        # if the agent is infected (exposed)
        if self.infection_status == INFECTED:
            # check if the transition dates have not already been determined
            if self.date_infectious is None:
                # Determine disease progression
                self.calculate_transition_dates(duration_infectious=duration_infectious)

                # update infection statistics
                environment.n_currently_infected += 1
                environment.n_ever_infected += 1

                if self.contact_tracer:
                    environment.n_ever_infected_and_prio += 1
                    environment.n_currently_infected_and_prio += 1

                # inform the infecting agent(s) about the infection
                self.inform_pre_infection_chain_about_infection()
                if self.age >= 60:
                    self.inform_pre_infection_chain_about_infection_60()
                if self.age >= 80:
                    self.inform_pre_infection_chain_about_infection_80()

            # exposed/infected -> presymptomatic infectious
            if environment.current_date >= self.date_infectious:
                self.infection_status = INFECTIOUS

        # if the agent is currently presymptomatic infectious and it is about time to change to the next status (ill)
        elif (
            self.infection_status == INFECTIOUS
            and environment.current_date >= self.date_ill
        ):
            # presymptomatic infectious -> ill
            if environment.current_date >= self.date_ill:
                # if the agent is meant to develop symptoms
                if self.will_develop_symptoms:
                    self.infection_status = ILL_SYMPTOMATIC
                    self.ever_infected_with_symptoms = True
                    self.date_symptomatic = self.date_ill

                # if the agent is not meant to develop symptoms
                else:
                    self.infection_status = ILL_ASYMPTOMATIC

                # contact tracing:
                if (
                    self.contact_tracer
                    and self.start_contact_tracing(
                        contact_tracing_detected_infections=contact_tracing_detected_infections,
                        environment=environment,
                        ct_start_date=ct_start_date,
                    )
                    and contact_tracing_allowed
                ):
                    self.trace_contacts(
                        contact_tracing_contact_selection_mode=contact_tracing_contact_selection_mode,
                        contact_tracing_contact_selection_frac=contact_tracing_contact_selection_frac,
                        environment=environment,
                    )

        # if the agent is either symptomatic or asymptomatic ill
        elif (
            self.infection_status == ILL_SYMPTOMATIC
            or self.infection_status == ILL_ASYMPTOMATIC
        ):
            # if the agent does not need to go to hospital
            if self.date_icu is None:
                if environment.current_date >= self.date_recovered:
                    self.infection_status = RECOVERED
                    environment.n_currently_infected -= 1
                    if self.contact_tracer:
                        environment.n_currently_infected_and_prio -= 1

            # if the agent has to be hospitalized
            else:
                if environment.current_date >= self.date_icu:
                    self.infection_status = ICU
                    self.inform_pre_infection_chain_about_icu()

        # if the agent currently is in hospital and now it is time now time to go home
        elif (
            self.infection_status == ICU
            and environment.current_date >= self.date_recovered
        ):
            self.infection_status = RECOVERED
            environment.n_currently_infected -= 1

    def get_infected(
        self,
        environment: Environment,
        infection_location: Optional[str] = None,
        infection_source: Optional[Agent] = None,
        date_infection: Optional[dt.date] = None,
        duration_infectious=2,
    ) -> None:
        """Represents the receiving of the virus."""

        self.infection_status = INFECTED
        self.ever_infected = True
        self.date_infection = (
            environment.current_date if date_infection is None else date_infection
        )
        self.tick_infection = environment.tick
        self.infection_location = infection_location
        self.infection_source = infection_source
        self.n_ever_infected_when_infected = environment.n_ever_infected

        self.update_infection(environment, duration_infectious=duration_infectious)

        # if the agent is not patient0
        if infection_source is not None:
            self.len_pre_infection_chain = infection_source.len_pre_infection_chain + 1
        else:
            self.len_pre_infection_chain = 0

    def infect(
        self,
        environment: Environment,
        infection_probability: Optional[float] = None,
        infection_probability_dict: Optional[dict] = None,
        contact_tracing: bool = False,
        vaccination_efficacy: float = 0,
        lin_age_inf_p_adj: Optional[float] = None,
        ct_isolation_level: float = 0,
        contact_tracing_detected_infections: Optional[str] = None,
        ct_start_date=None,
        duration_infectious=2,
    ) -> None:
        """Models the potential infection of the contacts saved in the temporary contact diary."""

        # if the agent is infectious
        if self.infection_status in INFECTIOUS_STATES:
            # go through all the entries in the temporary contact diary
            for entry in self.temp_contact_diary:
                # get the agent-j-object which had contact to this agent
                agent_j = environment.agents_dict[entry["id_j"]]

                # if agent-j is susceptible
                if agent_j.infection_status == SUSCEPTIBLE:
                    # get infection probability either from the infection probability dict or from the general value
                    infection_probability = (
                        infection_probability_dict[
                            entry["infection_probability_category"]
                        ]
                        if infection_probability is None
                        else infection_probability
                    )

                    # calculate weighted infection probability
                    p = infection_probability * entry["weight"]

                    # adjust infection probability by age of agent_j
                    if lin_age_inf_p_adj is not None:
                        p += lin_age_inf_p_adj * agent_j.age

                    # adjust infection probability by age-specific susceptibility-factor
                    p = p * agent_j.adjust_p_by_age()

                    # if isolated due to contact tracing, reduce infection probability
                    if self.contact_traced:
                        p = p * (1 - ct_isolation_level)

                    # if vaccinated, reduce infection probability
                    if agent_j.vaccinated:
                        p = p * (1 - vaccination_efficacy)

                    # Use random value to determine whether infection occurs or not
                    if random.random() < p:
                        # infect the other agent
                        agent_j.get_infected(
                            environment=environment,
                            infection_location=entry["location"],
                            infection_source=self,
                            duration_infectious=duration_infectious,
                        )

                        # increment personal infection statistics
                        self.infected_agents.append(agent_j)
                        self.n_directly_infected_agents += 1
                        if agent_j in self.household_members:
                            self.n_directly_infected_household_members += 1

            # if the agent is infectious and if the agent traces its contacts
            if contact_tracing:
                # add the whole temporary contact diary to the contact tracing diary
                # in other words: the agent starts to collect contacts to trace with the beginning of the infectious phase
                self.contact_tracing_diary.extend(self.temp_contact_diary)

    def trace_contacts(
        self,
        contact_tracing_contact_selection_mode: str,
        contact_tracing_contact_selection_frac: float,
        environment: Environment,
    ) -> None:
        """Models the process of contact tracing."""

        # only if the agent is not himself currently under quarantine due to contact tracing 
        # and if the current date is at least the official start date (and not within the warm up phase)
        if (
            not self.contact_traced
            and environment.current_date >= self.model.start_date
        ):
            # if there are enough contact tracing capacities:
            if (environment.contact_tracing_max_currently_traced is None 
                or environment.sum_tracings_last7days + environment.new_tracings_today < environment.contact_tracing_max_currently_traced):

                # increase contact tracing counter for the index person
                # (the index person is counted as one tracing)
                self.n_times_contact_tracing += 1
                environment.new_tracings_today += 1
                environment.n_currently_traced += 1

                # set the status of the agent to currently traced (index case is treated as a traced agent)
                self.contact_traced = True
                self.contact_tracing_date = environment.current_date
                self.contact_tracing_dates_list.append(environment.current_date)

                # go through the contact tracing diary and keep only those contacts 
                # that were made today, yesterday or the day before yesterday
                self.contact_tracing_diary = [
                    entry
                    for entry in self.contact_tracing_diary
                    if (environment.current_date - entry["date"]).days
                    <= 2
                ]

                # sort contacts in contact tracing diary by the contact intensity
                self.contact_tracing_diary.sort(
                    key=lambda entry: entry["weight"], reverse=True
                )

                # filter duplicates in contact tracing diary
                temp_contact_tracing_diary = self.contact_tracing_diary
                self.contact_tracing_diary = []
                collision = set()
                for entry in temp_contact_tracing_diary:
                    if entry["id_j"] not in collision:
                        self.contact_tracing_diary.append(entry)
                        collision.add(entry["id_j"])

                # if there are any contacts to trace
                if len(self.contact_tracing_diary) > 0:

                    # for each contact to trace, get the agent-object and 
                    # add attribute-information to the diary entry (to sort the entries in the next step)
                    if contact_tracing_contact_selection_mode != "weight":
                        for entry in self.contact_tracing_diary:
                            agent_j = environment.agents_dict[entry["id_j"]]
                            entry["pred_value"] = agent_j.pred_value
                            entry["weighted_pred_value"] = entry["pred_value"] * entry["weight"]
                            entry["random"] = random.random()
                            entry["old"] = agent_j.age
                            entry["young"] = -agent_j.age

                    # sort contact tracing diary by the defined contact attribute
                    self.contact_tracing_diary.sort(
                        key=lambda entry: entry[contact_tracing_contact_selection_mode],
                        reverse=True,
                    )

                    # calculate the number of contacts which should be traced
                    n_agents_traced = round(
                        len(self.contact_tracing_diary)
                        * contact_tracing_contact_selection_frac
                    )

                    # for each agent to be traced
                    for i in range(n_agents_traced):
                        
                        # if there are enough contact tracing capacities:
                        if (environment.contact_tracing_max_currently_traced is None 
                            or environment.sum_tracings_last7days + environment.new_tracings_today < environment.contact_tracing_max_currently_traced):
                            
                            # get the agent
                            agent_j = environment.agents_dict[
                                self.contact_tracing_diary[i]["id_j"]
                            ]

                            # if the agent is not already traced
                            if not agent_j.contact_traced:
                            
                                # set the status of the agent to currently traced
                                agent_j.contact_traced = True
                                agent_j.contact_tracing_date = environment.current_date

                                # increase contact tracing counters
                                agent_j.n_times_contact_traced += 1
                                agent_j.contact_tracing_dates_list.append(
                                    environment.current_date
                                )
                                self.n_contacts_traced += 1

                                environment.n_currently_traced += 1
                                environment.new_tracings_today += 1

        # Reset contact tracing diary
        self.contact_tracing_diary = []
