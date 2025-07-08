from random import Random
from typing import List
from typing import Optional

import pandas as pd

from src.entities.agent import Agent
from src.entities.agents_and_locations import AgentsAndLocations
from src.entities.location import Location
from src.entities.location import LocationDeclarant


class PopMaker:
    def __init__(
        self,
        df_agents_in_households: pd.DataFrame,
        location_declarants: Optional[List[LocationDeclarant]] = None,
        household_weight_column: Optional[str] = None,
        household_id_column: str = "hid",
        agent_class=Agent,
        seed=1,
        model=None,
    ):
        self.population_counter = 0
        self.df_agents_in_households: pd.DataFrame = df_agents_in_households
        self.location_declarants: Optional[LocationDeclarant] = location_declarants
        self.household_weight_column: Optional[str] = household_weight_column
        self.household_id_column = household_id_column
        self.agent_class = agent_class
        self.location_class = Location
        self.model = model
        self.nace2_division_codes = [
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            35,
            36,
            37,
            38,
            39,
            41,
            42,
            43,
            45,
            46,
            47,
            49,
            50,
            51,
            52,
            53,
            55,
            56,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
        ]
        self.seed = seed
        # self.randomizer = Random(self.seed)

    def create_agents(self, n_agents: int, agent_class: Agent = Agent) -> List[Agent]:
        # load a copy of the dataset
        df = self.df_agents_in_households.copy()

        # if households should be weighted
        if self.household_weight_column is not None:
            # get unique household-IDs and corresponding weight
            df_hids_and_weights = df.drop_duplicates(
                subset=[self.household_id_column]
            ).loc[:, [self.household_id_column, self.household_weight_column]]
            hids = list(df_hids_and_weights[self.household_id_column])
            weights = list(df_hids_and_weights[self.household_weight_column])

        else:
            hids = list(df[self.household_id_column].drop_duplicates())

        # population size counter
        agents_counter = 0
        households_counter = 0

        n_tries = 0

        # list for storing agents created below
        population = []

        # while actual population size is smaller than desired population size
        while agents_counter < n_agents:
            if self.household_weight_column is not None:
                # choose a random household-ID from the weighted list of IDs
                hid = self.randomizer.choices(hids, weights=weights, k=1)[0]
            else:
                # choose a random household-ID
                hid = self.randomizer.choice(hids)

            # get data of people living in this household
            df_one_household = df[df[self.household_id_column] == hid].reset_index()

            if agents_counter + len(df_one_household) <= n_agents or n_tries > 100:
                households_counter += 1

                household = []
                # for each agent/row in household/data
                for i, row in df_one_household.iterrows():
                    # create new agent instance
                    agent = agent_class(id=agents_counter)

                    agent.model = self.model

                    agent.household_id = households_counter

                    household.append(agent)

                    # copy attributes from soep to agent
                    for column in df.columns:
                        setattr(agent, column, row[column])

                    # append Agent to population-list
                    population.append(agent)

                    # increase the number of already created instances
                    agents_counter += 1

                for agent in household:
                    agent.household_size = len(household)
                    agent.household_members = household  # [hh_member for hh_member in household if hh_member is not agent]

            else:
                n_tries += 1

        return population

    def create_agents_and_locations(
        self,
        n_agents: int,
    ) -> AgentsAndLocations:
        self.population_counter += 1
        
        self.randomizer = Random(self.seed)

        agents = self.create_agents(n_agents=n_agents, agent_class=self.agent_class)

        locations = []

        # create home locations for all households/agents
        for agent in agents:
            if "home" not in agent.location_dict:
                home_location = Location(
                    category="home",
                    type="home",
                    id="home_" + str(agent.household_id),
                    infection_probability_category="close",
                )

                locations.append(home_location)

                for household_member in agent.household_members:
                    household_member.location_dict.update(
                        {
                            "home": {
                                "type": home_location.type,
                                "id": home_location.id,
                                "objects": [home_location],
                                "n_hours_per_visit": None,
                                "visit_condition": None,
                            },
                        },
                    )
                    household_member.home_type = agent.household_id
                    household_member.home_label = "home"
                    household_member.home_location = home_location

        # create all other locations besides the home locations
        if self.location_declarants is not None:
            for location_declarant in self.location_declarants:
                for agent in agents:
                    setattr(agent, "in_" + location_declarant.category, 0)

            for i, location_declarant in enumerate(self.location_declarants):
                # get all agents that fulfill the condition
                if location_declarant.association_condition is not None:
                    category_related_agents = [
                        agent
                        for agent in agents
                        if location_declarant.association_condition(agent)
                    ]
                else:
                    category_related_agents = agents

                location_types = []
                # set location types as agent attribute
                for agent in category_related_agents:
                    setattr(agent, "in_" + location_declarant.category, 1)

                    # später noch als methode des location_declarants umsetzen
                    location_type = (
                        location_declarant.type(agent)
                        if callable(location_declarant.type)
                        else location_declarant.type
                    )

                    location_types.append(location_type)

                    hours = (
                        location_declarant.n_hours_per_visit(agent)
                        if callable(location_declarant.n_hours_per_visit)
                        else location_declarant.n_hours_per_visit
                    )

                    agent.location_dict.update(
                        {
                            location_declarant.category: {
                                "category": location_declarant.category,
                                "type": location_type,
                                "id": None,
                                "objects": None,
                                "n_hours_per_visit": hours,
                                "visit_condition": location_declarant.visit_condition,
                            },
                        },
                    )

                unique_location_types = set(location_types)

                # for each location type
                for location_type in unique_location_types:
                    # get type-related agents
                    type_related_agents = [
                        agent
                        for agent in category_related_agents
                        if agent.location_dict[location_declarant.category]["type"]
                        == location_type
                    ]

                    # calculate the number of locations of this type
                    if location_declarant.n_agents_per_location is not None:
                        n_locations_of_this_type = max(
                            len(type_related_agents)
                            // location_declarant.n_agents_per_location,
                            1,
                        )
                    else:
                        n_locations_of_this_type = 1

                    # create all locations of this type
                    locations_of_this_type = []
                    for i in range(n_locations_of_this_type):
                        location = self.location_class(
                            category=location_declarant.category,
                            type=location_type,
                            id=location_declarant.category
                            + "_"
                            + str(location_type)
                            + "_"
                            + str(i),
                            appointment=location_declarant.appointment,
                            size=location_declarant.size,
                            n_contacts=location_declarant.n_contacts,
                            infection_probability_category=location_declarant.infection_probability_category,
                        )
                        locations_of_this_type.append(location)

                    locations.extend(locations_of_this_type)

                    # connect with an existing location object?
                    if location_declarant.existing_location_object is not None:
                        for agent in type_related_agents:
                            existing_location_object = (
                                location_declarant.existing_location_object(
                                    locations,
                                )
                            )
                            agent.location_dict[location_declarant.category][
                                "objects"
                            ] = [
                                existing_location_object,
                            ]

                    else:
                        for agent in type_related_agents:
                            random_locations = self.randomizer.choices(
                                locations_of_this_type,
                                k=location_declarant.n_associated_locations_per_agent,
                            )
                            agent.location_dict[location_declarant.category][
                                "objects"
                            ] = random_locations

        # add agents to associated agents for each location
        for agent in agents:
            for location_type in agent.location_dict:
                for location in agent.location_dict[location_type]["objects"]:
                    location.n_associated_agents += 1

        agents_and_locations = AgentsAndLocations(agents=agents, locations=locations)
        return agents_and_locations
