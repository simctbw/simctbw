from typing import Dict, List, Optional, Union

from src.entities.agent import Agent
from src.entities.environment import Environment

class Location:
    """The place where agents encounter each other."""
    def __init__(
        self, 
        category: str, 
        type: Union[str, int], 
        id: str,
        size: Optional[int] = None, 
        appointment: bool = False, 
        n_contacts: Optional[int] = None,
        infection_probability_category: Optional[str] = None,
        ):
        self.category: str = category
        self.type: Union[str, int] = type
        self.id : str = id
        self.visitors_of_the_day: List[(Agent, float)] = []
        self.n_associated_agents: int = 0
        self.size: int = size
        self.appointment: bool = appointment
        self.n_contacts: Optional[int] = n_contacts
        self.infection_probability_category: Optional[str] = infection_probability_category
    
    
    def connect_visitors_of_the_day(
            self, 
            environment: Environment, 
            save_contact_network: bool, 
            network_type="line", 
            sort_by_id=True,
            ):
        """
        Goes through the daily guest book, calculates the duration per encountering,
        saves this information in the corresponding agents and then cleans up the guest book.
        """     

        if sort_by_id:
            self.visitors_of_the_day = sorted(self.visitors_of_the_day, key=lambda x: x[0].id)

        # number of agents visited this location today
        n_visitors = len(self.visitors_of_the_day)

        # number of neighbors on each side of the "line-network" (drawback: the total number of an agent's contacts at this location is always an even number)
        n_neighbors = round(self.n_contacts / 2) if self.n_contacts is not None else None
        
        # if the contact network within each location is a line
        if network_type == "line":

            # for each agent that visited this location
            for i, (agent_i, hours_i) in enumerate(self.visitors_of_the_day):

                # List of agents the agent had contact to and the amount of time.
                # Did the agent have contact to all other visitors or only to a specified amount of neighbors?
                contacts: [(Agent, float)] = self.visitors_of_the_day if n_neighbors is None else self.visitors_of_the_day[ (i - n_neighbors) % n_visitors : (i + n_neighbors + 1) % n_visitors ]

                # for each (contact, hours)
                for agent_j, hours_j in contacts:
                    
                    # check if the contact is the agent himself
                    if agent_j is not agent_i:
                        
                        # choose a method of contact weight calculation
                        if self.appointment:
                            weight = min([hours_i, hours_j]) / 24
                        else:
                            weight = (hours_i * hours_j) / 576
                        
                        # save contact to general contact diary
                        if save_contact_network:
                            agent_i.add_contact_to_diary(
                                date=environment.current_date,
                                agent_j=agent_j,
                                weight=weight,
                                location=self.category,
                                )
                        
                        # save contact to temporary contact diary
                        agent_i.add_contact_to_temp_diary(
                            date=environment.current_date,
                            agent_j=agent_j,
                            weight=weight,
                            location=self.category,
                            infection_probability_category=self.infection_probability_category,
                            )

        # clear the list of visitors
        self.visitors_of_the_day = []


class LocationDeclarant:
    """A helper-object which acts like an interface to define the number and type of locations etc.."""
    def __init__(
        self,
        category: str,
        type: Union[int, str, callable],
        association_condition: Optional[callable]=None,
        n_hours_per_visit: Optional[Union[int, callable]]=None,
        n_agents_per_location: Optional[int]=None,
        n_associated_locations_per_agent: int = 1,
        visit_condition: Optional[callable] = None,
        existing_location_object: Optional[Location] = None,
        size: Optional[int] = None,
        appointment: bool = False,
        n_contacts: Optional[int] = None,
        set_agent_attribute: Optional[dict] = None,
        infection_probability_category: Optional[str] = None,
    ):
        self.association_condition = association_condition
        self.category = category
        self.type = type
        self.n_hours_per_visit = n_hours_per_visit
        self.n_agents_per_location = n_agents_per_location
        self.n_associated_locations_per_agent = n_associated_locations_per_agent
        self.visit_condition = visit_condition
        self.existing_location_object = existing_location_object
        self.size = size
        self.appointment = appointment
        self.n_contacts = n_contacts
        self.set_agent_attribute = set_agent_attribute
        self.infection_probability_category: Optional[str] = infection_probability_category