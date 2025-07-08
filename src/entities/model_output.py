from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
from bokehgraph import BokehGraph
import copy
import numpy as np
import datetime as dt
from pathlib import Path

import src
import src.entities.utils as utils

class ModelOutput:
    """Contains the data produced by a simulation run."""
    def __init__(
        self, 
        list_of_replication_output_dicts: List[dict],
        start_date: dt.date,
        end_date: dt.date,
        population_seed: int,
        n_replications: int,
        scaling_factor: float,
        ) -> None:
        
        self.list_of_replication_output_dicts: List[dict] = list_of_replication_output_dicts
        self.start_date: dt.date = start_date
        self.end_date: dt.date = end_date
        self.population_seed: int = population_seed
        self.all_agents_and_locations: List["AgentsAndLocations"] = [output_dict["agents_and_locations"] for output_dict in list_of_replication_output_dicts]
        self.location_categories = set([location.category for location in self.all_agents_and_locations[0].locations])
        self.df_average_degree_distributions: Optional[pd.DataFrame] = None
        self.df_agents: Optional[pd.DataFrame] = None
        self.agents: List["Agent"] = None
        self.df_infections: Optional[pd.DataFrame] = None
        self.df_cum_infections: Optional[pd.DataFrame] = None
        self.n_agents: Optional[int] = None
        self.n_replications = n_replications
        self.scaling_factor = scaling_factor
        self.df_dates = pd.concat([d["df_dates"] for d in self.list_of_replication_output_dicts])


    def get_agents(self) -> List["Agent"]:
        """Returns list which contains all agents from all model replications."""
        if self.agents is None:
            self.agents = []
            for aal in self.all_agents_and_locations:
                self.agents.extend(aal.agents)
        return self.agents


    def get_df_agents(
        self, 
        centrality: bool = False, 
        degree: bool = False,
        ) -> pd.DataFrame:
        """Creates dataframe containing all agents' attribute values over all replications."""
        
        self.df_agents = pd.concat(
            [aal.get_df_agents(centrality=(centrality if i == 0 else False), degree=(degree if i == 0 else False)) for i, aal in enumerate(self.all_agents_and_locations)]
            ).reset_index()
        
        return self.df_agents
    

    def eval_age_contact_matrix(
        self, 
        step_size: int = 5, 
        n_groups: int = 16,
        only_correlation: bool = False,
        locations: List[str] = [None],
        weighted_sim_contacts: bool = False,
        same_color_scale: bool = False,
        save_data=False,
        plot=True,
        ):

        def df_to_contact_matrix(
            n_groups: int, 
            df: pd.DataFrame, 
            col_contact_count: str = "weight", 
            col_age_group_i: str = "age_group_i", 
            col_age_group_j: str = "age_group_j",
            weighted_sim_contacts: bool = False,
            ) -> np.array:
            
            m = np.array([[float(0)]*n_groups]*n_groups)
            
            for i, row in df.iterrows():
                
                weight = row[col_contact_count]
                age_i = int(row[col_age_group_i])
                age_j = int(row[col_age_group_j])

                if weighted_sim_contacts:
                    m[age_j, age_i] += weight
                else:
                    m[age_j, age_i] += 1 
            
            m = m + m.transpose()

            return m
        

        def compare_contact_matrices(
            m_sim: np.array, 
            m_emp: np.array, 
            step_size: int = 1,
            only_correlation: bool = False,
            same_color_scale: bool = False,
            plot=True,
            
            ):
            flat_data = {
                "simulation": [m_sim[i, j] for i in range(len(m_sim[0])) for j in range(len(m_sim))],
                "polymod": [m_emp[i, j] for i in range(len(m_emp[0])) for j in range(len(m_emp))],
            }

            df_flat = pd.DataFrame(flat_data)
            r = np.corrcoef(df_flat["simulation"], df_flat["polymod"])[1,0]
            
            if only_correlation:
                return r
            
            else:
                print("pearson's R:", r)

                if save_data:
                    df_m_sim = pd.DataFrame(m_sim)
                    df_m_sim.to_csv(Path.joinpath(src.PATH, "data_output", "eval", "df_eval_contacts_m_sim.csv"))

                    df_m_emp = pd.DataFrame(m_emp)
                    df_m_emp.to_csv(Path.joinpath(src.PATH, "data_output", "eval", "df_eval_contacts_m_emp.csv"))

                    df_flat.to_csv(Path.joinpath(src.PATH, "data_output", "eval", "df_eval_contacts_m_flat.csv"))

                if plot:
                    fig, ax = plt.subplots(1,3)
                    fig.set_size_inches(14, 4)
                    fig.canvas.draw()

                    upper_bound = max([m_sim.max(), m_emp.max()])

                    sns.heatmap(m_sim, ax=ax[0], vmin=0, vmax=(upper_bound if same_color_scale else None))
                    sns.heatmap(m_emp, ax=ax[1], vmin=0, vmax=(upper_bound if same_color_scale else None))
                    sns.regplot(x="polymod", y="simulation", data=df_flat, ax=ax[2], ci=None, color="black")
                
                    ax[0].set(
                        title="Simulated age contact matrix",
                        xlabel="Age group of agent i",
                        ylabel="Age group of agent j",
                        ylim=[0, len(m_sim)],
                        )
                    ax[1].set(
                        title="Empirical age contact matrix",
                        xlabel="Age group of agent i",
                        ylabel="Age group of agent j",
                        ylim=[0, len(m_sim)],
                        )
                    ax[2].set(
                        title="Correlation of simulated and empirical contact matrix",
                        xlabel="Empirical number of contacts",
                        ylabel="Simulated number of contacts",
                        )
                    
                    # change tick-labels
                    for i in range(2):
                        labels = [item.get_text() for item in ax[i].get_xticklabels()]
                        labels = [int(tick) * step_size for tick in labels]
                        ax[i].set_xticklabels(labels)

                        labels = [item.get_text() for item in ax[i].get_yticklabels()]
                        labels = [int(tick) * step_size for tick in labels]
                        ax[i].set_yticklabels(labels)
            
                    plt.tight_layout()
                    plt.show()
            

        # Count people per age group
        all_df_average_edges = pd.concat(
                [aal.get_df_average_edges() for aal in self.all_agents_and_locations]
                )
        df_sim = all_df_average_edges.copy()
        df_sim["age_group_i"] = df_sim["age_i"].apply(lambda x: utils.group_it(x, 0, step_size, n_groups))
        df_sim["age_group_j"] = df_sim["age_j"].apply(lambda x: utils.group_it(x, 0, step_size, n_groups))
        
        df_sim_age_group_count = df_sim.groupby("id_i", as_index=False)["age_group_i"].first()
        df_sim_age_group_count = df_sim_age_group_count["age_group_i"].value_counts()

        if only_correlation:
            r_dict = {}

        for location in locations:
            str_location = ("all_locations" if location is None else location)

            all_df_average_edges = pd.concat(
                [aal.get_df_average_edges(location=location) for aal in self.all_agents_and_locations]
                )
            
            if len(all_df_average_edges) > 0:

                m_prem = np.array(
                    pd.read_excel(
                        Path.joinpath(
                            src.PATH, 
                            "data_input", 
                            "plos_prem_2017", 
                            "Germany", 
                            "MUestimates_" + str_location + "_1_ONLY_GERMANY.xlsx",
                            )))
                
                df_sim = all_df_average_edges.copy()
                df_sim = df_sim.loc[(df_sim["age_i"] < 80) & (df_sim["age_j"] < 80),:]
                df_sim["age_group_i"] = df_sim["age_i"].apply(lambda x: utils.group_it(x, 0, step_size, n_groups))
                df_sim["age_group_j"] = df_sim["age_j"].apply(lambda x: utils.group_it(x, 0, step_size, n_groups))
                
                # create contact matrices
                m_sim_5 = df_to_contact_matrix(n_groups=n_groups, df=df_sim, weighted_sim_contacts=weighted_sim_contacts)
                
                m_prem = (m_prem + m_prem.transpose()) / 2

                for i in range(len(m_sim_5)):
                    m_sim_5[i] = m_sim_5[i] / (df_sim_age_group_count[i] * self.n_replications)
                m_sim_5 = (m_sim_5 + m_sim_5.transpose()) / 2

                if not only_correlation:
                    utils.print_header("Location:" + str_location)
                r = compare_contact_matrices(m_sim_5, m_prem, step_size=step_size, only_correlation=only_correlation, same_color_scale=same_color_scale, plot=plot)
                if only_correlation:    
                    r_dict[str_location] = r

        if only_correlation:
            return r_dict
    

    def eval_incidence(self, sim_date_col="date_symptomatic", plot=True, per_replication=False) -> dict:
        g = None

        def prepare_df_sim_incidence(df_agents):
            df_sim_incidence = df_agents
            df_sim_incidence["counter"] = 1
            helper_df = pd.DataFrame({"date": utils.dates_between(self.start_date-dt.timedelta(days=8), self.end_date)})
            df_sim_incidence = df_sim_incidence.groupby([sim_date_col]).count()[["counter"]].reset_index()
            df_sim_incidence = helper_df.merge(df_sim_incidence, left_on="date", right_on=sim_date_col, how="left")
            df_sim_incidence["incidence_7_days_sim"] = np.nan

            for i, row in df_sim_incidence.iterrows():
                incidence_7_days = (df_sim_incidence.loc[i-8:i-1,"counter"].sum() / len(df_agents)) * 100000
                df_sim_incidence.loc[i, "incidence_7_days_sim"] = incidence_7_days
                
            df_sim_incidence["date"] = pd.to_datetime(df_sim_incidence["date"])
            df_sim_incidence = df_sim_incidence.reset_index().groupby(pd.Grouper(key="date", freq="W-MON", closed='left')).mean(numeric_only=True).reset_index()
            df_sim_incidence["date"] = pd.to_datetime(df_sim_incidence["date"]).apply(lambda x: x.date())
            df_sim_incidence = df_sim_incidence.loc[(df_sim_incidence["date"] >= dt.date(2022, 6, 1)) & (df_sim_incidence["date"] < dt.date(2022, 8, 1)),:].reset_index() # insert start and end date?
            df_sim_incidence.date = df_sim_incidence.date.apply(lambda x: str(x))
            df_sim_incidence["incidence_7_days_sim"] = df_sim_incidence["incidence_7_days_sim"] * self.scaling_factor
            return df_sim_incidence

        def prepare_df_rki_incidence():
            df = pd.read_csv(Path.joinpath(src.PATH, "data_input", "covid_data", "COVID-19-Faelle_7-Tage-Inzidenz_Bundeslaender.csv"))
            df = df.loc[df["Bundesland_id"]==8,:]
            df = df.loc[df["Altersgruppe"] == "00+"]
            df["Meldedatum"] = pd.to_datetime(df["Meldedatum"]).apply(lambda x: x.date())
            df = df.loc[(df["Meldedatum"] >= dt.date(2022, 6, 1)) & (df["Meldedatum"] < dt.date(2022, 8, 1)),:].reset_index()
            df["incidence_7_days_emp"] = df["Inzidenz_7-Tage"]
            df["Meldedatum"] = pd.to_datetime(df["Meldedatum"])
            df = df.groupby(pd.Grouper(key="Meldedatum", freq="W-MON", closed='left')).mean(numeric_only=True).reset_index()
            df["Meldedatum"] = pd.to_datetime(df["Meldedatum"]).apply(lambda x: x.date())
            df = df.loc[(df["Meldedatum"] >= dt.date(2022, 6, 1)) & (df["Meldedatum"] < dt.date(2022, 8, 1)),:].reset_index()
            df.Meldedatum = df.Meldedatum.apply(lambda x: str(x))
            return df
        
        def calculate_fit(df_rki_incidence, df_sim_incidence):
            df_incidence = df_rki_incidence.merge(df_sim_incidence, left_on="Meldedatum", right_on="date", how="left")
            root_mean_squared_error = utils.get_rmse(df_incidence["incidence_7_days_emp"], df_incidence["incidence_7_days_sim"])
            df_incidence["rmse"] = root_mean_squared_error
            df_incidence["date"] = df_incidence["date"].apply(lambda x: x.split()[0])
            return df_incidence, root_mean_squared_error
        
        df_rki_incidence = prepare_df_rki_incidence()
        
        if not per_replication:
            df_agents = self.get_df_agents().copy()
            df_sim_incidence = prepare_df_sim_incidence(df_agents)
            df_incidence, root_mean_squared_error = calculate_fit(df_rki_incidence, df_sim_incidence)
            if plot:
                fig, ax = plt.subplots(1,1)
                ax.plot(df_incidence["date"], df_incidence["incidence_7_days_sim"], label="simulated")
                ax.plot(df_incidence["date"], df_incidence["incidence_7_days_emp"], label="empirical")
                ax.set_title("7-day incidence rate per 100,000")
                ax.set_xlabel("Date")
                ax.set_ylabel("Weekly averaged 7-day incidence rate per 100,000")
                plt.xticks(rotation=45, ha='right')
                ax.legend()
                plt.show()
        
        else:
            list_df = []
            for i, aal in enumerate(self.all_agents_and_locations):
                df_agents = aal.get_df_agents()
        
                df_sim_incidence = prepare_df_sim_incidence(df_agents)
                df_incidence, root_mean_squared_error = calculate_fit(df_rki_incidence, df_sim_incidence)
                df_incidence["replication"] = i
                list_df.append(df_incidence)
            df_incidence = pd.concat(list_df)
            
            df_incidence_sim = df_incidence[["date", "incidence_7_days_sim"]]
            df_incidence_sim.columns = ["date", "incidence"]
            df_incidence_sim["source"] = "Simulation"
            
            df_incidence_emp = df_incidence[["date", "incidence_7_days_emp"]]
            df_incidence_emp.columns = ["date", "incidence"]
            df_incidence_emp["source"] = "Empirical"

            df_incidence_long = pd.concat([df_incidence_emp, df_incidence_sim])
            
            if plot:
                fig, ax = plt.subplots(1,1)
                

                sns.lineplot(data=df_incidence_long, x="date", y="incidence", hue="source", ax=ax)

                ax.set_title("7-day incidence rate per 100,000")
                ax.set_xlabel("Date")
                ax.set_ylabel("Weekly averaged 7-day incidence rate per 100,000")
                plt.xticks(rotation=45, ha='right')
                plt.show()

                g = fig
                
        return {
            "df_incidence": df_incidence,
            "rmse": root_mean_squared_error,
            "plot": g,
        }
    
    def eval_cum_infections_by_age(
        self, 
        sim_infection_col="ever_infected_with_symptoms", 
        emp_start_date=None, 
        emp_end_date=None,
        plot=True,
        plot_4=True,
        age_specific=True,
        ) -> dict:

        df_output = None

        def prepare_df_rki():
            df = pd.read_csv(Path.joinpath(src.PATH, "data_input", "covid_data", "COVID-19-Faelle_7-Tage-Inzidenz_Bundeslaender.csv"))
            df = df.loc[df["Bundesland_id"]==8,:]
            df = df.loc[df["Altersgruppe"] != "00+"]

            df.loc[df["Altersgruppe"] == "00-04", "Altersgruppe"] = 0
            df.loc[df["Altersgruppe"] == "05-14", "Altersgruppe"] = 5
            df.loc[df["Altersgruppe"] == "15-34", "Altersgruppe"] = 15
            df.loc[df["Altersgruppe"] == "35-59", "Altersgruppe"] = 35
            df.loc[df["Altersgruppe"] == "60-79", "Altersgruppe"] = 60
            df.loc[df["Altersgruppe"] == "80+",   "Altersgruppe"] = 80
            
            df["Meldedatum"] = pd.to_datetime(df["Meldedatum"]).apply(lambda x: x.date())

            if emp_start_date is not None:
                df = df[df["Meldedatum"] >= emp_start_date].reset_index(drop=True)
            
            if emp_end_date is not None:
                df = df[df["Meldedatum"] <= emp_end_date].reset_index(drop=True)

            df["Bevoelkerung"] = df["Bevoelkerung"] / len(df.Meldedatum.unique())
            df = df.groupby("Altersgruppe")[["Faelle_neu", "Bevoelkerung"]].sum().reset_index()
            df.columns = ["age_group", "count_emp", "population_emp"]
            return df
        
        def prepare_df_sim():
            # get simulated data
            df_agents = self.get_df_agents()
            df_agents["population_sim"] = 1
            df_sim_age_sum = df_agents.groupby("age_group_rki")[[sim_infection_col, "population_sim"]].sum().reset_index()
            df_sim_age_sum.columns = ["age_group", "count_sim", "population_sim"]
            df_sim_age_sum["count_sim"] = df_sim_age_sum["count_sim"] * self.scaling_factor
            return df_sim_age_sum

        df_rki_age_sum = prepare_df_rki()
        df_sim_age_sum = prepare_df_sim()
        
        # join both datasets
        df_both_age_sum = df_sim_age_sum.merge(df_rki_age_sum, on="age_group")
        
        # calculate age-relative freqs
        if age_specific:
            df_both_age_sum["rel_count_emp"] = df_both_age_sum["count_emp"] / df_both_age_sum["population_emp"] 
            df_both_age_sum["rel_count_sim"] = df_both_age_sum["count_sim"] / df_both_age_sum["population_sim"] 
        else:
            # calculate relative freqs
            df_both_age_sum["rel_count_emp"] = df_both_age_sum["count_emp"] / df_both_age_sum["count_emp"].sum() 
            df_both_age_sum["rel_count_sim"] = df_both_age_sum["count_sim"] / df_both_age_sum["count_sim"].sum()
        
        # calculate correlation
        corr = np.corrcoef(df_both_age_sum["rel_count_emp"], df_both_age_sum["rel_count_sim"])[1,0]

        # calculate RMSE
        rmse = utils.get_rmse(df_both_age_sum["rel_count_emp"], df_both_age_sum["rel_count_sim"])

        df_output = df_both_age_sum

        g = None

        if plot:
            if plot_4:

                # create plots
                fig, ax = plt.subplots(2,2)
                fig.set_size_inches(10, 10)
                fig.suptitle("Age-specific infection frequencies", fontsize=16)
            
                y_max = max([df_both_age_sum["rel_count_emp"].max(), df_both_age_sum["rel_count_sim"].max()])

                sns.barplot(data = df_both_age_sum, x = "age_group", y = "rel_count_emp", color="steelblue", ax=ax[0,0])
                ax[0,0].set(
                    title="Empirical age-specific infection count",
                    xlabel="Age-group (lower boundary)",
                    ylabel="Relative frequency",
                    )
                
                sns.barplot(data = df_both_age_sum , x = "age_group", y = "rel_count_sim", color="steelblue", ax=ax[0,1])
                ax[0,1].set(
                    title="Simulated age-specific infection count",
                    xlabel="Age-group (lower boundary)",
                    ylabel="Relative frequency",
                    )
                
                ax[1,0].plot(df_both_age_sum["age_group"], df_both_age_sum["rel_count_emp"], label="empirical")
                ax[1,0].scatter(df_both_age_sum["age_group"], df_both_age_sum["rel_count_emp"], label="empirical")
                ax[1,0].plot(df_both_age_sum["age_group"], df_both_age_sum["rel_count_sim"], label="simulation")
                ax[1,0].scatter(df_both_age_sum["age_group"], df_both_age_sum["rel_count_sim"], label="simulation")
                ax[1,0].legend()
                ax[1,0].annotate("RMSE:" + str(round(rmse, 2)), xy = (30, 0))
                ax[1,0].set(
                    title="Fit of empirical and simulated data",
                    xlabel="Age-group (lower boundary)",
                    ylabel="Relative frequency",
                    )
                
                ax[1,1].plot([0,1000000], [0,1000000], color="grey")
                sns.regplot(data = df_both_age_sum, x="rel_count_emp", y="rel_count_sim", ax=ax[1,1], color="black", ci=None)
                ax[1,1].set_ylim(0-0.01, y_max+0.01)
                ax[1,1].set_xlim(0-0.01, y_max+0.01)
                ax[1,1].annotate("Pearson's R:" + str(round(corr,2)), xy = (0.2, 0))
                ax[1,1].set(
                    title="Linear relationship between emp. and sim. frequencies",
                    xlabel="Emp. relative frequency",
                    ylabel="Sim. relative frequency",
                    )
                plt.tight_layout()
                plt.show()

                # print correlation
                print("Pearson's R:", corr)
                print("RMSE:", rmse)

                g = fig
            
            else:
                list_df = []
                for i, aal in enumerate(self.all_agents_and_locations):
                    df_agents = aal.get_df_agents()
                    df_agents["population_sim"] = 1
                    df_sim_age_sum = df_agents.groupby("age_group_rki")[[sim_infection_col, "population_sim"]].sum().reset_index()
                    
                    df_sim_age_sum.columns = ["age_group", "freq", "population_sim"]
                    df_sim_age_sum["freq"] = df_sim_age_sum["freq"] * self.scaling_factor


                    df_sim_age_sum["rel_freq"] = df_sim_age_sum["freq"] / df_sim_age_sum["freq"].sum()
                    if age_specific:
                        df_sim_age_sum["rel_freq"] = df_sim_age_sum["freq"] / df_sim_age_sum["population_sim"]
                    
                    df_sim_age_sum["source"] = "Simulation"
                    list_df.append(df_sim_age_sum)
                

                df_sim_age_sum = pd.concat(list_df)
                
                df_rki_age_sum = prepare_df_rki()
                df_rki_age_sum.columns = ["age_group", "freq", "population_emp"]
                df_rki_age_sum["rel_freq"] = df_rki_age_sum["freq"] / df_rki_age_sum["freq"].sum()
                if age_specific:
                    df_rki_age_sum["rel_freq"] = df_rki_age_sum["freq"] / df_rki_age_sum["population_emp"]
                df_rki_age_sum["source"] = "Empirical"

                df_both_age_sum = pd.concat([df_rki_age_sum, df_sim_age_sum])
                df_output = df_both_age_sum
                
                g = sns.catplot(
                    data=df_both_age_sum, 
                    kind="bar",
                    x="age_group", 
                    y="rel_freq", 
                    hue="source",
                    errorbar="ci", 
                    )
                plt.title("Infections by age")
                plt.xlabel("Age group (lower bound)")
                plt.ylabel("Share of all infections")
                plt.show()

        return {
            "corr": corr,
            "rmse": rmse,
            "plot": g,
            "df": df_output,
        }
