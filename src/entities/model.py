from typing import Dict, List, Optional, Union
import datetime as dt
import collections
import copy
import random
from pathlib import Path
import os
import gc

import pandas as pd

from src.entities.model_output import ModelOutput
from src.entities.pop_maker import PopMaker
from src.entities.environment import Environment
from src.entities.location import LocationDeclarant
from src.entities.agent import Agent, PandemicAgent
import src.entities.utils as utils
from src.logging import log


class Model:
    def __init__(
        self,
        mode: str,
        start_date: dt.date,
        end_date: dt.date,
        n_agents: int,
        df_agents_in_households: pd.DataFrame,
        household_weight_column: Optional[str] = None,
        location_declarants: Optional[List[LocationDeclarant]] = None,
        timetable: Optional[dict] = None,
        n_replications: int = 1,
        population_seed: Optional[int] = None,
        initial_infections: Optional[Union[int, float, dict]] = None,
        n_days_save_contact_network: int = 0,
        stop_sim_if_0_infections: bool = False,
        infection_probability: Optional[float] = None,
        infection_probability_dict: Optional[dict] = None,
        lin_age_inf_p_adj: Optional[float] = None,
        ct_isolation_level: float = 0,
        replace_population: bool = True,
        master_model_index: Optional[int] = None,
        folder_path=None,
        output_type: str = "model_output",
        name_of_run: str = None,
        add_info: Optional[Dict] = None,
        save_only_these_columns: List[str] = None,
        n_days_warm_up_period: int = 14,
        agent_class=Agent,
        lockdown_population_sort_mode: Optional[str] = None,
        lockdown_frac: Optional[float] = None,
        contact_tracing_population_sort_mode: Optional[str] = None,
        contact_tracing_frac: Optional[float] = None,
        contact_tracing_contact_selection_mode: Optional[str] = None,
        contact_tracing_contact_selection_frac: Optional[float] = None,
        contact_tracing_detected_infections: Optional[str] = None,
        contact_tracing_max_currently_traced: Optional[int] = None,
        ct_start_date=None,
        n_days_contact_tracing_isolation: int = None,
        vaccination_population_sort_mode: Optional[str] = None,
        vaccination_frac: Optional[float] = None,
        vaccination_efficacy: Optional[float] = None,
        population_selection_mode: Optional[str] = None,
        degree: bool = True,
        scaling_factor: float = 1,
        duration_infectious: int = 2,
        run_folder_path=None,
    ):
        self.run_folder_path = run_folder_path

        # misc
        self.scaling_factor: float = scaling_factor
        self.mode: str = mode
        self.degree = degree

        # start date
        self.start_date: dt.date = start_date

        # end date
        self.end_date: dt.date = end_date

        # warm up period: simulation starts at (start_date - n_days_warm_up_period)
        self.n_days_warm_up_period = n_days_warm_up_period

        # population size
        self.n_agents: int = n_agents

        # The agent class from which the agent instances are built
        self.agent_class = agent_class

        # dataset that contains empirical information on agent attributes
        self.df_agents_in_households: pd.DataFrame = df_agents_in_households.copy()

        # the name of the column in which the household selection probability is
        self.household_weight_column: Optional[str] = household_weight_column

        # location definitions
        self.location_declarants: Optional[List[LocationDeclarant]] = (
            location_declarants
        )

        # definitions of events
        self.timetable: Optional[Dict] = timetable

        # number of (internal) replications, when the model is running
        self.n_replications: int = n_replications

        # seed to reproduce the population
        self.population_seed: int = (
            population_seed
            if population_seed is not None
            else random.randint(0, 1000000)
        )

        # share of initially infected agents
        self.initial_infections: Optional[Union[int, dict]] = initial_infections

        # number of days from the beginning in which the contacts are saved
        self.n_days_save_contact_network: int = n_days_save_contact_network

        # finish simulation if there are not any active cases left
        self.stop_sim_if_0_infections: bool = stop_sim_if_0_infections

        # infection probability
        self.infection_probability: Optional[float] = infection_probability

        # dict containing specific infection probabilities for different locations
        self.infection_probability_dict: Optional[dict] = infection_probability_dict

        self.lin_age_inf_p_adj = lin_age_inf_p_adj

        self.duration_infectious = duration_infectious

        # an index to distinguish different models
        self.master_model_index: Union[int, str] = master_model_index

        # path to folder in which the output data will be stored
        self.folder_path = folder_path

        # Define how the output of the simulation is returned. Options: "model_output" (default), "save_rep_on_disk", "list", "save_on_disk"
        self.output_type: str = output_type

        # a string to name the model
        self.name_of_run: str = name_of_run

        # a dict with additional information to add to the df_agents. Keys are added as column names, values (scalar) are added as column data.
        self.add_info: Optional[Dict] = add_info

        # a list with column names. Only these columns are saved in df_agents.
        self.save_only_these_columns: List[str] = save_only_these_columns

        # a string defining the way how agents are selected for lockdown, contact tracing or vaccination
        self.population_selection_mode: str = population_selection_mode

        # CONTACT TRACING
        self.contact_tracing_population_sort_mode = contact_tracing_population_sort_mode
        self.contact_tracing_frac = contact_tracing_frac
        self.contact_tracing_contact_selection_mode = (
            contact_tracing_contact_selection_mode
        )
        self.contact_tracing_contact_selection_frac = (
            contact_tracing_contact_selection_frac
        )
        self.contact_tracing_detected_infections = contact_tracing_detected_infections
        self.contact_tracing_max_currently_traced: int = (
            contact_tracing_max_currently_traced
        )
        self.n_days_contact_tracing_isolation = n_days_contact_tracing_isolation
        self.ct_start_date = ct_start_date
        self.ct_isolation_level = ct_isolation_level

        # LOCKDOWN
        self.lockdown_population_sort_mode = lockdown_population_sort_mode
        self.lockdown_frac = lockdown_frac

        # VACCINATION
        self.vaccination_population_sort_mode = vaccination_population_sort_mode
        self.vaccination_frac = vaccination_frac
        self.vaccination_efficacy = vaccination_efficacy

        # creates the population of agents
        self.population_maker = PopMaker(
            df_agents_in_households=self.df_agents_in_households,
            location_declarants=self.location_declarants,
            household_weight_column=self.household_weight_column,
            agent_class=self.agent_class,
            seed=self.population_seed,
            model=self,
        )

        # create new population each replication?
        self.replace_population: bool = replace_population

        # check if this is an infection simulation
        if (
            isinstance(initial_infections, int)
            and initial_infections > 0
            or isinstance(initial_infections, dict)
        ):
            assert (
                infection_probability is not None
                or infection_probability_dict is not None
            ), "No infection probability (dict) specified."

        # create directory
        if self.folder_path is not None:
            try:
                os.mkdir(self.folder_path)
            except Exception as e:
                pass

    def add_pred_value(self, agents: List[Agent]) -> List[Agent]:
        """Adds the predicted pandemic relevance of as an attribute to each agent."""
        for agent in agents:
            agent.pid = int(agent.pid)
            agent.pred_value = utils.predict_caused_infections(
                age_rki=agent.age_group_rki, household_size=agent.household_size
            )
            agent.age_group_rki_inc_rank_emp = (
                utils.recode_age_group_rki_to_inc_rank_emp(agent.age_group_rki)
            )
            agent.age_group_rki_inc_rank_sim = (
                utils.recode_age_group_rki_to_inc_rank_sim(agent.age_group_rki)
            )
            agent.age_group_rki_age_rank = utils.recode_age_group_rki_to_age_rank(
                agent.age_group_rki
            )
        return agents

    def sort_agents(
        self, agents: List[Agent], sort_mode: str, add_random: bool = False
    ) -> List[Agent]:
        """Sorts the list of agents (population) as defined by sort_mode."""

        def add_jitter(execute, agents, agent_attr, low=0.000001, high=0.00001):
            if execute:
                for agent in agents:
                    setattr(
                        agent,
                        agent_attr,
                        getattr(agent, agent_attr) + random.uniform(low, high),
                    )

        # random order
        if sort_mode == "random":
            random.shuffle(agents)

        # high values of pandemic relevance first
        elif sort_mode == "super-spreaders":
            add_jitter(execute=add_random, agents=agents, agent_attr="pred_value")
            agents.sort(key=lambda x: x.pred_value, reverse=True)

        # high values of age first
        elif sort_mode == "old":
            add_jitter(execute=add_random, agents=agents, agent_attr="age")
            agents.sort(key=lambda x: x.age, reverse=True)

        # low values of age first
        elif sort_mode == "young":
            add_jitter(execute=add_random, agents=agents, agent_attr="age")
            agents.sort(key=lambda x: x.age, reverse=False)

        # large households first
        elif sort_mode == "household_size":
            add_jitter(execute=add_random, agents=agents, agent_attr="household_size")
            agents.sort(key=lambda x: x.household_size, reverse=True)

        elif sort_mode == "age":
            add_jitter(
                execute=add_random, agents=agents, agent_attr="age_group_rki_age_rank"
            )
            agents.sort(key=lambda x: x.age_group_rki_age_rank, reverse=False)

        elif sort_mode == "inc_rank_emp":
            add_jitter(
                execute=add_random,
                agents=agents,
                agent_attr="age_group_rki_inc_rank_emp",
            )
            agents.sort(key=lambda x: x.age_group_rki_inc_rank_emp, reverse=False)

        elif sort_mode == "inc_rank_sim":
            add_jitter(
                execute=add_random,
                agents=agents,
                agent_attr="age_group_rki_inc_rank_sim",
            )
            agents.sort(key=lambda x: x.age_group_rki_inc_rank_sim, reverse=False)

        # no input or typo?
        else:
            raise ValueError

        return agents

    def select_agents(
        self,
        agents: List[Agent],
        frac: float,
        attribute: str,
        value,
        how: str = "individuals",
    ):
        """
        Selects a defined share of agents from the start of the agent list and sets a specific value on a defined attribute of the selected agents.
        The option "how" allows to select the whole household of the selected agents. The household members are counted as selected agents so the share of agents selected remains the same.
        """
        # absolute number of agents to be selected
        n_selected = round(len(agents) * frac)

        # select only the agent
        if how == "individuals":
            for i in range(n_selected):
                setattr(agents[i], attribute, value)

        # select the whole household
        elif how == "households_of_individuals":
            n_already_selected = 0
            for agent in agents:
                if getattr(agent, attribute) is not value:
                    for hh_member in agent.household_members:
                        assert getattr(hh_member, attribute) is not value

                        if n_already_selected < n_selected:
                            setattr(hh_member, attribute, value)
                            n_already_selected += 1

                if n_already_selected >= n_selected:
                    break
        else:
            raise ValueError

        return agents

    def create_model_output(
        self, list_of_replication_output_dicts: List[dict]
    ) -> ModelOutput:
        """Creates a model_output-object from a list of output_dicts generated by multiple replications."""
        model_output = ModelOutput(
            list_of_replication_output_dicts,
            self.start_date,
            self.end_date,
            self.population_seed,
            n_replications=self.n_replications,
            scaling_factor=self.scaling_factor,
        )
        model_output.n_agents = self.n_agents
        model_output.run_folder_path = self.run_folder_path
        return model_output

    def save_on_disk(self, model_output: ModelOutput, rep_index=None) -> None:
        """Saves the output data stored in a model_output-object as csv-files on disk."""

        # create the df_agents from the model_output-object
        # degree = False if self.mode == "calib" else True
        if self.mode != "calib":
            df_agents = model_output.get_df_agents(degree=self.degree)

            # add some additional information
            def add_params(df: pd.DataFrame) -> pd.DataFrame:
                df["master_model_index"] = self.master_model_index
                df["name_of_run"] = self.name_of_run
                df["lockdown_population_sort_mode"] = str(
                    self.lockdown_population_sort_mode
                )
                df["lockdown_frac"] = self.lockdown_frac
                df["vaccination_population_sort_mode"] = str(
                    self.vaccination_population_sort_mode
                )
                df["vaccination_frac"] = self.vaccination_frac
                df["vaccination_efficacy"] = self.vaccination_efficacy
                df["population_selection_mode"] = self.population_selection_mode
                df["contact_tracing_population_sort_mode"] = str(
                    self.contact_tracing_population_sort_mode
                )
                df["contact_tracing_frac"] = self.contact_tracing_frac
                df["contact_tracing_contact_selection_mode"] = str(
                    self.contact_tracing_contact_selection_mode
                )
                df["contact_tracing_contact_selection_frac"] = (
                    self.contact_tracing_contact_selection_frac
                )
                df["contact_tracing_detected_infections"] = (
                    self.contact_tracing_detected_infections
                )
                df["n_days_contact_tracing_isolation"] = (
                    self.n_days_contact_tracing_isolation
                )
                df["ct_isolation_level"] = self.ct_isolation_level
                df["contact_tracing_max_currently_traced"] = (
                    self.contact_tracing_max_currently_traced
                )
                df["n_agents"] = self.n_agents
                df["ct_start_date"] = self.ct_start_date
                df["duration_infectious"] = self.duration_infectious

                df["end_date"] = self.end_date
                return df

            df_agents = add_params(df_agents)
            df_agents["unique_agent_id"] = (
                df_agents["master_model_index"].astype(str)
                + "_"
                + df_agents["id"].astype(str)
            )
            df_agents["unique_household_id"] = (
                df_agents["master_model_index"].astype(str)
                + "_"
                + df_agents["household_id"].astype(str)
            )

            # keep only specific columns?
            if self.save_only_these_columns is not None:
                df_agents = df_agents.loc[:, self.save_only_these_columns]

            # add additional information stored in self.add_info (dict)
            if self.add_info is not None:
                for k in self.add_info.keys():
                    df_agents[k] = self.add_info[k]

            if "model" in df_agents.columns:
                df_agents = df_agents.drop(columns=["model"])
            
            # create file name and save df_agents as csv-file
            file_name = f"df_agents_{str(self.master_model_index)}_{str(rep_index)}.parquet"
            df_agents.to_parquet(Path.joinpath(self.folder_path, file_name), index=False, compression="gzip")

            # save df_dates
            df_dates = model_output.df_dates
            df_dates = add_params(df_dates)
            file_name = f"df_dates_{str(self.master_model_index)}_{str(rep_index)}.parquet"
            df_dates.to_parquet(Path.joinpath(self.folder_path, file_name), index=False, compression="gzip")

            del df_agents
            del df_dates

        # create and save a meta-df (df_params)
        rmse_cum_infections_by_age = model_output.eval_cum_infections_by_age(
            sim_infection_col="ever_infected_with_symptoms",
            plot=False,
            emp_start_date=model_output.start_date,
            emp_end_date=model_output.end_date,
            age_specific=True,
        )["rmse"]

        rmse_incidence = model_output.eval_incidence(
            "date_symptomatic",
            plot=False,
        )["rmse"]

        output_dict = {
            "rmse_cum_infections_by_age": rmse_cum_infections_by_age,
            "rmse_incidence": rmse_incidence,
        }

        df_params = pd.DataFrame(output_dict, index=[0])
        if self.add_info is not None:
            for k in self.add_info.keys():
                df_params[k] = self.add_info[k]

        file_name = f"df_params_{str(self.master_model_index)}_{str(rep_index)}.parquet"
        #df_params.to_parquet(Path.joinpath(self.folder_path, file_name), index=False, compression="gzip")

        # delete objects
        del model_output
        del df_params
        del output_dict
        del rmse_cum_infections_by_age
        del rmse_incidence

    def run(self) -> ModelOutput:
        """Executes the simulation model with one or more replications."""

        try:
            # list for storing the output data of each replication
            list_of_replication_output_dicts = []

            # run the replications
            for i in range(self.n_replications):
                # run one replication and save output in dict
                replication_output_dict = self.replication(internal_replication_index=i)

                # append output-dict to list
                list_of_replication_output_dicts.append(replication_output_dict)

                # save every df_agent of rep on disk and return nothing (?)
                if self.output_type == "save_rep_on_disk":
                    model_output = self.create_model_output(
                        list_of_replication_output_dicts
                    )
                    self.save_on_disk(model_output, rep_index=i)

                    list_of_replication_output_dicts = []
                    del model_output
                    del replication_output_dict
                    gc.collect()

            # return a model_output-object that contains/processes all replication_output_dicts (?)
            if self.output_type == "model_output":
                model_output = self.create_model_output(
                    list_of_replication_output_dicts
                )
                return model_output

            # return all replication_output_dicts in a list (?)
            elif self.output_type == "list":
                return list_of_replication_output_dicts

            # save unified df_agents of all reps on disk and return nothing (?)
            elif self.output_type == "save_on_disk":
                model_output = self.create_model_output(
                    list_of_replication_output_dicts
                )
                self.save_on_disk(model_output)

        except Exception as e:
            log.exception("RUN DIED")
            raise e

    def replication(self, internal_replication_index: str) -> dict:
        """The actual execution of one replication of the simulation model."""

        gc.collect()
        # ------------------- Setup ------------------- #

        agents_and_locations = self.population_maker.create_agents_and_locations(
            n_agents=self.n_agents
        )

        agents_and_locations.rep_index = internal_replication_index
        agents_and_locations.master_model_index = self.master_model_index

        agents = agents_and_locations.agents
        locations = agents_and_locations.locations

        # select agents for lockdown, contact tracing or vaccination
        agents = self.add_pred_value(agents)

        if self.lockdown_population_sort_mode is not None:
            agents = self.sort_agents(
                agents=agents,
                sort_mode=self.lockdown_population_sort_mode,
            )
            agents = self.select_agents(
                agents=agents,
                frac=self.lockdown_frac,
                attribute="lockdown",
                value=True,
                how=self.population_selection_mode,
            )

        if self.vaccination_population_sort_mode is not None:
            agents = self.sort_agents(
                agents=agents,
                sort_mode=self.vaccination_population_sort_mode,
            )
            agents = self.select_agents(
                agents=agents,
                frac=self.vaccination_frac,
                attribute="vaccinated",
                value=True,
                how=self.population_selection_mode,
            )

        if self.contact_tracing_population_sort_mode is not None:
            agents = self.sort_agents(
                agents=agents,
                sort_mode=self.contact_tracing_population_sort_mode,
            )
            agents = self.select_agents(
                agents=agents,
                # frac=(1 if self.contact_tracing_population_sort_mode == "random" else self.contact_tracing_frac),
                frac=self.contact_tracing_frac,
                attribute="contact_tracer",
                value=True,
                how=self.population_selection_mode,
            )

        # important: shuffle the list of agents again, because maybe it was sorted above
        random.shuffle(agents)

        # Initialize the environment-object
        self.environment = Environment(
            start_date=self.start_date,
            current_date=self.start_date
            - dt.timedelta(days=self.n_days_warm_up_period),
            agents=agents,
            timetable=self.timetable,
            contact_tracing_max_currently_traced = self.contact_tracing_max_currently_traced,
        )
        self.environment.update_current_interventions()

        # Calculate the number of days that will be simulated
        n_simulated_days = (
            (self.end_date - self.start_date).days + 1 + self.n_days_warm_up_period
        )
        agents_and_locations.n_simulated_days = n_simulated_days

        # initial infections
        if self.initial_infections is not None:
            # simple random selection
            if (
                isinstance(self.initial_infections, float)
                or isinstance(self.initial_infections, int)
            ) and self.initial_infections > 0:
                patients_zero = random.sample(
                    agents, round(self.initial_infections * self.n_agents)
                )

            # group specific random selection
            elif isinstance(self.initial_infections, dict):
                patients_zero = []
                group_attribute = self.initial_infections["attribute"]
                for key in self.initial_infections.keys():
                    if key != "attribute":
                        subpopulation = [
                            agent
                            for agent in agents
                            if getattr(agent, group_attribute) == key
                        ]
                        n = round(self.initial_infections[key] * len(subpopulation))
                        patients_zero.extend(random.sample(subpopulation, n))

            # infect the selected patients0
            for agent in patients_zero:
                # sample an infection date a few days before the simulation starts
                date_infection = random.choice(
                    utils.dates_between(
                        self.environment.current_date - dt.timedelta(days=8),
                        self.environment.current_date - dt.timedelta(days=1),
                    )
                )
                agent.get_infected(
                    environment=self.environment,
                    infection_source=None,
                    date_infection=date_infection,
                )
                agent.patient0 = True

            assert len(patients_zero) > 0

        list_n_currently_infected = collections.deque()

        # ------------------- Simulation loop ------------------- #

        # for each simulated day
        for day in range(n_simulated_days):
            if not self.stop_sim_if_0_infections or (
                self.stop_sim_if_0_infections
                and self.environment.n_currently_infected > 0
            ):
                # each agent: visit locations that are on the schedule for this day
                for agent in agents:
                    agent.visit(environment=self.environment)

                save_contact_network = (
                    True if day < self.n_days_save_contact_network else False
                )

                # each location: create contacts between agents visiting this location today
                for location in locations:
                    location.connect_visitors_of_the_day(
                        environment=self.environment,
                        save_contact_network=save_contact_network,
                    )

                # if it is an infection simulation
                if self.initial_infections is not None:
                    
                    # every agent: infect others and update infection status
                    for agent in agents:
                        agent.infect(
                            environment=self.environment,
                            infection_probability=self.infection_probability,
                            infection_probability_dict=self.infection_probability_dict,
                            vaccination_efficacy=self.vaccination_efficacy,
                            contact_tracing=(
                                True
                                if self.contact_tracing_population_sort_mode is not None
                                else False
                            ),
                            lin_age_inf_p_adj=self.lin_age_inf_p_adj,
                            ct_isolation_level=self.ct_isolation_level,
                            ct_start_date=self.ct_start_date,
                            duration_infectious=self.duration_infectious,
                        )
                        
                        agent.update_infection(
                            environment=self.environment,
                            contact_tracing_contact_selection_mode=self.contact_tracing_contact_selection_mode,
                            contact_tracing_contact_selection_frac=self.contact_tracing_contact_selection_frac,
                            contact_tracing_detected_infections=self.contact_tracing_detected_infections,

                            # this is a first test. it is tested again later.
                            contact_tracing_allowed=(
                                True
                                if self.contact_tracing_max_currently_traced is None
                                or (
                                    self.environment.sum_tracings_last7days < self.contact_tracing_max_currently_traced
                                    )
                                else False
                            ),
                            n_days_contact_tracing_isolation=self.n_days_contact_tracing_isolation,
                            ct_start_date=self.ct_start_date,
                        )

                # delete all contacts from temporary contact diary
                for agent in agents:
                    agent.temp_contact_diary = []

                # self.environment.contact_tracing_currently_traced = 0

                # if there are still days left to simulate
                if day < n_simulated_days:
                    # Update environment
                    self.environment.step()

                list_n_currently_infected.append(self.environment.n_currently_infected)

        # ------------------- Output preparation ------------------- #

        for agent in agents:

            # delete some agent attributes
            delattr(agent, "location_dict")
            delattr(agent, "home_location")
            delattr(agent, "household_members")
            delattr(agent, "agents_ever_met")
            delattr(agent, "contact_tracing_diary")
            delattr(agent, "list_hours_not_at_home")

            if isinstance(agent, PandemicAgent):
                delattr(agent, "infection_source")
                delattr(agent, "infected_agents")

        # create df_dates
        df_dates = pd.DataFrame(self.environment.date_data)
        df_dates["master_model_index"] = self.master_model_index
        df_dates["internal_replication_index"] = internal_replication_index

        # prepare output
        replication_output_dict = {
            "master_model_index": self.master_model_index,
            "population_seed": self.population_seed,
            "internal_replication_index": internal_replication_index,
            "agents_and_locations": agents_and_locations,
            "n_currently_infected": list_n_currently_infected,
            "df_dates": df_dates,
        }
        del agents_and_locations
        for agent in agents:
            del agent
        for location in locations:
            del location
        del agents
        del locations
        gc.collect()
        return replication_output_dict
