import src
from src.entities.location import LocationDeclarant
import src.entities.utils as utils
from src.entities.agent import ICU, ILL_SYMPTOMATIC

from pathlib import Path
import os
import random
import datetime as dt

import numpy as np
import pandas as pd

# load homeoffice data
df_homeoffice = pd.read_csv(Path.joinpath(src.PATH, "data_input", "homeoffice", "Alipouretal_WFH_Germany-master", "Alipouretal_WFH_Germany-master", "wfh_nace2.csv"))
df_homeoffice = df_homeoffice.rename({"nace2": "nace2_division"}, axis=1)
df_homeoffice["wfh_freq_prob"] = df_homeoffice["wfh_freq"] / 100
dict_homeoffice = df_homeoffice[["nace2_division", "wfh_freq_prob"]].set_index("nace2_division").to_dict()["wfh_freq_prob"]

# load SOEP
def get_soep4sim(year: int, area:str = "DE"):
    nace2_table = pd.read_csv(Path.joinpath(src.PATH, "data_input", "nace", "nace2_codes.csv"))
    nace2_division_to_section = nace2_table.set_index("nace2_division").to_dict()['nace2_section']

    # path to folder where the partial SOEP-datasets are
    soep_folder_path = Path.joinpath(src.PATH, "data_input", "soep", "SOEP-CORE.v36eu_STATA", "Stata")
    soep4sim_file_path = Path.joinpath(src.PATH, "data_input", "soep4sim", f"soep4sim_{area}.csv")

    if os.path.exists(soep4sim_file_path):
        df = pd.read_csv(soep4sim_file_path)

    else:
        df_pequiv = utils.load_partial_soep_dataset(
            path=Path.joinpath(soep_folder_path, "pequiv.dta"),
            rename_columns={
                "d11101": "age",
                "d11102ll": "gender",
                "l11101": "federal_state", 
                "e11101": "work_hours_year",
            }, 
            filter_year=year,
            drop_duplicates=True,
            )

        df_pl = utils.load_partial_soep_dataset(
            path=Path.joinpath(soep_folder_path, "pl.dta"),
            rename_columns={
                "plb0183_h": "work_hours_day",
                "plb0186_h": "work_hours_week",
                "pli0040"  : "hours_shopping",
            },
            keep_columns=["hid"],
            filter_year=year,
            drop_duplicates=True,
            )

        df_pgen = utils.load_partial_soep_dataset(
            path=Path.joinpath(soep_folder_path, "pgen.dta"),
            rename_columns={"pgnace2": "nace2_division"},
            filter_year=year,
            drop_duplicates=True,
            )
        
        df_bjp = utils.load_partial_soep_dataset(
            path=Path.joinpath(soep_folder_path, "raw", "bip.dta"),
            rename_columns={
                "bip_23_02": "higher_education",
            },
            keep_columns=["hid"],
            filter_year=year,
            drop_duplicates=True,
            )

        df_hhrf = utils.load_partial_soep_dataset(
            path=Path.joinpath(soep_folder_path, "raw", "hhrf.dta"),
            keep_columns=["hid", "bihhrf"], # bihhrf = 2018 cross-sectional hh weight
            drop_duplicates=False,
            )
        
        df_hl = utils.load_partial_soep_dataset(
            path=Path.joinpath(soep_folder_path, "hl.dta"),
            rename_columns={"hlc0005_h": "hh_income"},
            keep_columns=["hid", "hh_income"],
            filter_year=year,
            drop_duplicates=False,
            )
        df_hl = df_hl.drop("syear", axis=1)
        
        # join individual data
        dfs = [
            df_pl, 
            df_pequiv, 
            df_pgen,
            df_bjp,
            ]
        dfs = [df.set_index(["hid", "syear", "pid" ]) for df in dfs]
        df = pd.concat(dfs, axis=1).reset_index()

        # join household data
        df_hh = pd.merge(df_hhrf, df_hl, on="hid")
        df_hh = df_hh.set_index("hid")
        df = pd.merge(df, df_hh, on="hid", how="left")
        
        # keep only Baden-Wuerttemberg
        if area == "BW":
            df = df[df["federal_state"] == 8]
        
        # recode age
        df.loc[(df["age"] < 0) | (df["age"] > 100), "age"] = np.nan
        
        # recode higher education
        df["student"] = df["higher_education"]
        df["student"] = df["student"].apply(lambda x: (1 if x > 0 else 0))
        
        # recode work hours
        df.loc[df["work_hours_week"] < 0, "work_hours_week"] = np.nan
        
        # people with 0 on alternative work hour measure get valid 0
        df.loc[(df["work_hours_week"].isna()) & (df["work_hours_year"] == 0), "work_hours_week"] = 0 
        
        # people older than 75 and missing work hours get valid 0 work hours
        df.loc[(df["work_hours_week"].isna()) & (df["age"] > 75), "work_hours_week"] = 0 

        # People younger than 20 get 0 work hours
        df.loc[(df["work_hours_week"].isna()) & (df["age"] < 20), "work_hours_week"] = 0 

        # impute work hours
        df["age_gender_whw_mean"] = df.groupby(["age"])["work_hours_week"].transform(np.mean)
        df.loc[df["work_hours_week"].isna(), "work_hours_week"] = df["age_gender_whw_mean"]

        # calculate daily work hours
        df["work_hours_day"] = df["work_hours_week"] / 5

        # impute shopping hours
        df.loc[df["hours_shopping"] < 0, "hours_shopping"] = np.nan
        df.loc[df["hours_shopping"].isna(), "hours_shopping"] = df["hours_shopping"].mean()
        
        #recode nace
        df.loc[df["nace2_division"].isna(), "nace2_division"] = -1
        df.loc[df["nace2_division"] <= 0, "nace2_division"] = -1
        
        # impute nace
        df["nace2_division"] = df["nace2_division"].apply(lambda x: x if x > 0 else 0)

        # recode nace2-division to nace2-section
        df["nace2_section"] = df["nace2_division"].apply(lambda x: (nace2_division_to_section[x] if x > 0 else 0))

        # recode hh_income
        df.loc[df["hh_income"] < 0, "hh_income"] = np.nan

        # create new weight
        df["weight"] = df["bihhrf"]
        
        # adjust weight for students
        df["student_in_household"] = df.groupby("hid")["student"].transform(max)
        df["weight"] = df["weight"] + df["weight"] * df["student_in_household"]  * 0.5

        # adjust weight for people of age 25 - 44
        df["25_45"] = df["age"].apply(lambda age: (1 if 25 <= age < 45 else 0))
        df["25_45_hh_sum"] = df.groupby("hid")["25_45"].transform(sum)
        df["weight"] = df["weight"] + df["weight"] * df["25_45_hh_sum"] * 0.5

        # adjust weight for people of age 1 - 9
        df["1_10"] = df["age"].apply(lambda age: (1 if age < 10 else 0))
        df["1_10_hh_sum"] = df.groupby("hid")["1_10"].transform(sum)
        df["weight"] = df["weight"] - df["weight"] * df["1_10_hh_sum"] * 0.2

        # adjust weight for people of age 70 - 80
        df["70_80"] = df["age"].apply(lambda age: (1 if 70 <= age < 80 else 0))
        df["70_80_hh_sum"] = df.groupby("hid")["70_80"].transform(sum)
        df["weight"] = df["weight"] - df["weight"] * df["70_80_hh_sum"] * 0.1

        # adjust weight for people of age 80+
        df["80+"] = df["age"].apply(lambda age: (1 if age >= 80 else 0))
        df["80+_hh_sum"] = df.groupby("hid")["80+"].transform(sum)
        df["weight"] = df["weight"] + df["weight"] * df["80+_hh_sum"] * 0.2

        # drop households where members have missing values
        df = utils.drop_incomplete_households(
            df=df,
            columns=[
                "age",
                "student",
                "nace2_division",
                ],
            household_level=True,
            )
        
        # recode age
        df["age_group_5"] = df["age"].apply(lambda x: utils.group_it(x, 0, 5, 18, summarize_highest=True, return_value="lower_bound"))
        df["age_group_rki"] = df["age"].apply(lambda x: utils.group_age_like_rki(x))

        # save file on disk
        df.to_csv(soep4sim_file_path, index=False)
    
    return df


# define locations
def def_locations(
    
    # kingergarten 1
    n_contacts_kindergarten_1=2,
    n_hours_kindergarten_1=6,
    n_agents_kindergarten_1=15,

    # kindergarten 2
    n_contacts_kindergarten_2=2,
    n_hours_kindergarten_2=6,
    n_agents_kindergarten_2=15,

    # school
    n_contacts_school=2,
    n_hours_school=6,
    n_agents_school=25,
    
    # university
    n_contacts_university=2,
    n_hours_university=4,
    n_agents_university=25,

    # work
    n_contacts_work=2,
    n_agents_work=10,
    
    # friends 1
    n_contacts_friends_1=2,
    n_hours_friends_1=0.5,
    n_agents_friends_1=10,

    # random contacts 1
    n_contacts_random_1=4,
    n_hours_random_1=2,
    n_agents_random_1=10,
    
    # random contacts 3 (all)
    n_contacts_random_3=6,
    n_hours_random_3=3,
    n_agents_random_3=10,

    # random 15-contacts 
    n_contacts_random_4=2,
    n_hours_random_4=1,
    n_agents_random_4=10,

    # shop
    n_contacts_shop=2,
    n_agents_shop=1000,
    ):
    
    location_declarants = [

        # kindergarten 1
        LocationDeclarant(
            category="kindergarten_group_1",
            type="kindergarten_group_1",
            association_condition=lambda agent: ((0 <= agent.age <= 2) and random.random() < 0.295),
            n_hours_per_visit=n_hours_kindergarten_1,
            n_agents_per_location=n_agents_kindergarten_1,
            visit_condition = lambda environment, agent: (
                environment.weekday < 5 
                and 
                agent.infection_status not in [ICU, ILL_SYMPTOMATIC]
                and 
                not agent.lockdown and not agent.contact_traced
            ),
            appointment=True,
            n_contacts=n_contacts_kindergarten_1,
        ),

        # kindergarten 2
        LocationDeclarant(
            category="kindergarten_group_2",
            type="kindergarten_group_2",
            association_condition=lambda agent: ((3 <= agent.age <= 5) and random.random() < 0.945),
            n_hours_per_visit=n_hours_kindergarten_2,
            n_agents_per_location=n_agents_kindergarten_2,
            visit_condition = lambda environment, agent: (
                environment.weekday < 5 
                and 
                agent.infection_status not in [ICU, ILL_SYMPTOMATIC]
                and 
                not agent.lockdown and not agent.contact_traced
            ),
            appointment=True,
            n_contacts=n_contacts_kindergarten_2,
        ),

        # age-specific school classes
        LocationDeclarant(
            association_condition=lambda agent: 6 <= agent.age <= 20,
            category="school",
            type=lambda agent: agent.age,
            n_hours_per_visit=n_hours_school,
            n_agents_per_location=n_agents_school,
            visit_condition = lambda environment, agent: (
                environment.weekday < 5 
                and 
                agent.infection_status not in [ICU, ILL_SYMPTOMATIC]
                and
                (environment.current_interventions["open_school"] == 1 if environment.current_interventions is not None else True)
                and 
                not agent.lockdown and not agent.contact_traced
                ),
            appointment=True,
            n_contacts=n_contacts_school,
        ),

        # university
        LocationDeclarant(
            association_condition=lambda agent: (
                agent.student == 1 
                and 
                agent.age > 20
            ),
            category="university",
            type="university",
            n_hours_per_visit=n_hours_university,
            n_agents_per_location=n_agents_university,
            visit_condition = lambda environment, agent: (
                environment.weekday < 5 
                and 
                agent.infection_status not in [ICU, ILL_SYMPTOMATIC]
                and
                (random.random() < environment.current_interventions["open_university"] if environment.current_interventions is not None else True)
                and 
                not agent.lockdown and not agent.contact_traced
            ),
            appointment=True,
            n_contacts=n_contacts_university,
        ),

        # work place
        LocationDeclarant(
            category="work",
            type=lambda agent: agent.nace2_section,
            association_condition=lambda agent: (
                agent.work_hours_day > 0
                and
                agent.age > 20
            ),
            n_hours_per_visit= lambda agent: agent.work_hours_day,
            n_agents_per_location=n_agents_work,
            visit_condition=lambda environment, agent: (
                environment.weekday < 5 
                and 
                random.random() < dict_homeoffice[agent.nace2_division]
                and
                agent.infection_status not in [ICU, ILL_SYMPTOMATIC] and not agent.lockdown and not agent.contact_traced
                ,
            ),
            appointment=True,
            n_contacts = n_contacts_work,
        ),

        # shop
        LocationDeclarant(
            category="shop",
            type="shop",
            n_hours_per_visit=lambda agent: agent.hours_shopping,
            n_agents_per_location=n_agents_shop,
            visit_condition=lambda environment, agent: environment.weekday < 6 and agent.infection_status not in [ICU, ILL_SYMPTOMATIC] and not agent.lockdown and not agent.contact_traced,
            n_contacts = n_contacts_shop,
        ),

        # friends
        LocationDeclarant(
            category="friends_1",
            type=lambda agent: agent.age + random.randint(-5, 5),
            n_hours_per_visit=n_hours_friends_1,
            n_agents_per_location=n_agents_friends_1,
            appointment=True,
            visit_condition=lambda environment, agent: environment.weekday >= 5 and agent.infection_status not in [ICU, ILL_SYMPTOMATIC] and not agent.lockdown and not agent.contact_traced,
            n_contacts = n_contacts_friends_1,
        ),

        # random contacts middle aged
        LocationDeclarant(
            category="random_1",
            type="random_1",
            n_hours_per_visit=n_hours_random_1,
            n_agents_per_location=n_agents_random_1,
            appointment=True,
            association_condition=lambda agent: 15 <= agent.age <= 60,
            visit_condition=lambda environment, agent: agent.infection_status not in [ICU, ILL_SYMPTOMATIC] and not agent.lockdown and not agent.contact_traced,
            n_contacts=n_contacts_random_1,
        ),

         # random contacts all
        LocationDeclarant(
            category="random_3",
            type="random_3",
            n_hours_per_visit=n_hours_random_3,
            n_agents_per_location=n_agents_random_3,
            appointment=True,
            visit_condition=lambda environment, agent: agent.infection_status not in [ICU, ILL_SYMPTOMATIC] and not agent.lockdown and not agent.contact_traced,
            n_contacts=n_contacts_random_3,
        ),

        # random contacts 15
        LocationDeclarant(
            category="random_4",
            type="random_4",
            n_hours_per_visit=n_hours_random_4,
            n_agents_per_location=n_agents_random_4,
            appointment=True,
            association_condition=lambda agent: 15 <= agent.age <= 20,
            visit_condition=lambda environment, agent: agent.infection_status not in [ICU, ILL_SYMPTOMATIC] and not agent.lockdown and not agent.contact_traced,
            n_contacts=n_contacts_random_4,
        ),
    ]
    return location_declarants

if __name__ == "__main__":
    get_soep4sim(
        area="BW", 
        year=2018,
        )