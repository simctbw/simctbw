from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
import pandas as pd
import numpy as np

import src
from src.entities import utils as utils


class AgentsAndLocations:
    """Holds the population of all agents and locations and provides methods that preprocess those populations for further analysis by the ModelOutput-class."""
    def __init__(
        self,
        agents: List["Agent"], 
        locations: List["Location"],
        ):
        self.agents: List["Agent"] = agents
        self.agents_dict: Dict[int: "Agent"] = {agent.id: agent for agent in self.agents}
        self.locations: List["Location"] = locations
        self.df_agents: Optional[pd.DataFrame] = None
        self.df_locations: Optional[pd.DataFrame] = None
        self.df_daily_contacts: Optional[pd.DataFrame] = None
        self.df_average_daily_contacts: Optional[pd.DataFrame] = None
        self.df_daily_edges: Optional[pd.DataFrame] = None
        self.df_average_edges: Optional[pd.DataFrame] = None
        self.average_contact_network: Optional[nx.Graph] = None
        self.daily_age_contact_matrices: Optional[dict] = None
        self.average_age_contact_matrix: Optional[np.array] = None
        self.freqs_associated_agents: Optional[pd.DataFrame] = None
        self.location_types: Optional[pd.DataFrame] = None
        self.n_simulated_days: Optional[int] = None
        self.df_infection_chain: Optional[pd.DataFrame] = None
        self.infection_network: Optional[nx.Graph] = None
        self.model_index: Optional[int] = None
        self.rep_index: Optional[int] = None
    

    def get_df_agents(self, centrality: bool = False, degree: bool = False) -> pd.DataFrame:
        """Returns dataframe containing all agent attributes."""
    
        df = pd.DataFrame([vars(agent) for agent in self.agents])
        df = df.drop(["contact_diary", "temp_contact_diary", "df_daily_location_contacts", "df_daily_contacts"], axis=1)
        df["replication"] = self.rep_index
        df["model_index"] = self.model_index
        df["age_group_5"] = df["age"].apply(
            lambda age: utils.group_it(age, 0, 5, 18, return_value="lower_bound")
            )

        # calculate betweenness-centrality?
        if centrality:
            centrality_dict = nx.betweenness_centrality(self.get_average_contact_network())
            centrality_df = pd.DataFrame({"betweennees_centrality_contacts": pd.Series(centrality_dict)})
            centrality_df = centrality_df.reset_index().rename(columns={"index": "id"})
            df = pd.merge(df, centrality_df, on="id")
        else:
            df["betweennees_centrality_contacts"] = np.nan

        # calculate degree-centrality?
        if degree:
            cn = self.get_average_contact_network()
            degree_df = pd.DataFrame({"degree": pd.Series(dict(cn.degree()))})
            degree_df = degree_df.reset_index().rename(columns={"index": "id"})
            df = pd.merge(df, degree_df, on="id")
        else:
            df["degree"] = np.nan
        
        self.df_agents = df
        return self.df_agents
    

    def get_df_locations(self) -> pd.DataFrame:
        """Returns dataframe containing all location attributes."""
        if self.df_location is None:
            self.df_locations = pd.DataFrame([vars(location) for location in self.locations])
        return self.df_locations
        

    def get_df_daily_contacts(self, location: Optional[str] = None) -> pd.DataFrame:
        """Returns dataframe containing the daily frequency of contact for every agent (i) to every other agent (j)."""
        dfs = []
        for agent in self.agents:
            df_agents_daily_contacts = agent.get_df_daily_contacts(location=location)
            
            if df_agents_daily_contacts is not None:
                dfs.append(df_agents_daily_contacts)

        assert len(dfs) > 0, "No contacts saved during simulation."
        self.df_daily_contacts = pd.concat(dfs).reset_index(drop=True)
        return self.df_daily_contacts


    def get_df_average_daily_contacts(self, location: Optional[str] = None) -> pd.DataFrame:
        """Returns the average daily contacts between agent i and agent j."""
        df = self.get_df_daily_contacts(location=location)
        df = df.groupby(["id_i", "id_j"], as_index=False).agg({
            "weight": "sum", 
            "age_i": "first", 
            "age_j": "first",
            })
        df["weight"] = df["weight"] / self.n_simulated_days
        self.df_average_daily_contacts = df
        return self.df_average_daily_contacts


    def get_df_daily_edges(self) -> pd.DataFrame:
        df = self.get_df_daily_contacts().copy()
        for i, row in df.iterrows():
            if row["id_i"] > row["id_j"]:
                df.loc[i, "id_i"] = row["id_j"]
                df.loc[i, "id_j"] = row["id_i"]
        df = df.drop_duplicates(subset=["date", "id_i", "id_j"]).reset_index(drop=True)
        self.df_daily_edges = df
        return self.df_daily_edges


    def get_df_average_edges(self, location: Optional[str] = None) -> pd.DataFrame:
        df = self.get_df_average_daily_contacts(location=location).copy()
        df = df[df["id_i"] > df["id_j"]]
        self.df_average_edges = df
        return self.df_average_edges


    def get_average_contact_network(self) -> nx.Graph:
        """Returns a networkx-graph-object containing all agents linked by their average daily contact frequency."""
        
        if self.average_contact_network is None:
            df_average_daily_contacts = self.get_df_average_edges()
        
            graph = nx.Graph()
            
            nodes = [
                (int(agent.id), {
                    "age": agent.age, 
                    "gender": agent.gender,
                    }
                ) for agent in self.agents]
            
            edges = [(int(row["id_i"]), int(row["id_j"]), {"weight": row["weight"]}) for _,row in df_average_daily_contacts.iterrows()]
            
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            
            self.average_contact_network = graph
        
        return self.average_contact_network
    

    def contact_network_to_gephi(self, path: Optional[Path]=None) -> None:
        """Saves average contact network as a gephi-file (.gexf)."""
        graph = self.get_average_contact_network()

        if path is None:
            path = Path.joinpath(src.PATH, "data_output", "contact_network_" + utils.str_time() + ".gexf")
        
        nx.write_gexf(graph, path)
    

    def infection_network_to_gephi(self, path: Optional[Path]=None) -> None:
        """Saves average contact network as a gephi-file (.gexf)."""
        graph = self.get_infection_network()

        if path is None:
            path = Path.joinpath(src.PATH, "data_output", "infection_network_" + utils.str_time() + ".gexf")
        
        nx.write_gexf(graph, path)


    def create_daily_age_contact_matrices(self) -> dict:
        """Returns a dictionary containing a daily age-specific contact matrix for each simulated day."""
        if self.daily_age_contact_matrices is None:
            daily_age_contact_matrices = {}
            df_daily_edges = self.get_df_daily_edges()
            df_agents = self.get_df_agents()
            
            for i, row in df_daily_edges.iterrows():
    
                date = row["date"]
                weight = row["weight"]
                age_i = int(df_agents.loc[df_agents["id"]==row["id_i"], "age"])
                age_j = int(df_agents.loc[df_agents["id"]==row["id_j"], "age"])
                
                if not date in daily_age_contact_matrices.keys():
                    daily_age_contact_matrices[date] = np.array([[float(0)]*101]*101)

                daily_age_contact_matrices[date][age_j, age_i] = daily_age_contact_matrices[date][age_j, age_i] + weight
                daily_age_contact_matrices[date][age_i, age_j] = daily_age_contact_matrices[date][age_j, age_i] + weight
            
            self.daily_age_contact_matrices = daily_age_contact_matrices

        return self.daily_age_contact_matrices
    

    def create_average_age_contact_matrix(self) -> np.array:
        """Returns an average daily age-specific contact matrix averaged over all simulated days."""
        
        if self.average_age_contact_matrix is None:
            
            df_average_edges = self.get_df_average_edges()
            
            m = np.array([[float(0)]*101]*101)

            for i, row in df_average_edges.iterrows():
                
                weight = row["weight"]
                age_i = int(row["age_i"])
                age_j = int(row["age_j"])

                m[age_j, age_i] = m[age_j, age_i] + weight
                m[age_i, age_j] = m[age_i, age_j] + weight
            
            self.average_age_contact_matrix = m

        return self.average_age_contact_matrix
        

    def get_freqs_associated_agents(self) -> dict:
        """Return dictionary containing the frequency distribution of the number of agents per location category."""
        if self.freqs_associated_agents is None:
            data_dict = {}
            for location in self.locations:
                if location.category not in data_dict.keys():
                    data_dict[location.category] = []
                
                data_dict[location.category].append(location.n_associated_agents)
            
            self.freqs_associated_agents = data_dict
        
        return self.freqs_associated_agents
    

    def get_location_types(self) -> dict:
        if self.location_types is None:
            data_dict = {}
            for location in self.locations:
                if location.category not in data_dict.keys():
                    data_dict[location.category] = []
                data_dict[location.category].append(location.type)
            
            for category in data_dict.keys():
                data_dict[category] = sorted(data_dict[category])
            
            self.location_types = data_dict
        return self.location_types
    

    def get_df_infection_chain(self) -> pd.DataFrame:
        if self.df_infection_chain is None:
            list_agent_i = []
            list_agent_j = []

            for agent in self.agents:
                for infected_agent in agent.infected_agents:
                    list_agent_i.append(agent.id)
                    list_agent_j.append(infected_agent.id)
            
            self.df_infection_chain = pd.DataFrame({
                "id_i": list_agent_i,
                "id_j": list_agent_j,
            })
        
        return self.df_infection_chain
        

    def get_infection_network(self) -> nx.Graph:

        if self.infection_network is None:
            graph = nx.Graph()
            
            nodes = [(int(a.id), {
                "age": a.age, 
                "gender": a.gender, 
                "post_chain": a.len_post_infection_chain,
                "pre_chain": a.len_pre_infection_chain,
                }) for a in self.agents]
            
            edges = [(int(row["id_i"]), int(row["id_j"]), {"weight": 1}) for _,row in self.get_df_infection_chain().iterrows()]
            
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            
            self.infection_network = graph
        
        return self.infection_network