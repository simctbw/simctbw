prep_data <- function(df, aggregate_agents=FALSE, add_sens_x_list=FALSE){
  df$male <- ifelse(df$gender==1, 1, 0)
  df$ever_infected <- ifelse(df$ever_infected == TRUE, 1, 0)
  df[is.na(df$tick_infection), "tick_infection"] <- max(df$tick_infection, na.rm = T) + 1
  df[is.na(df$n_ever_infected_when_infected), "n_ever_infected_when_infected"] <- max(df$n_ever_infected_when_infected, na.rm = T) + 1
  df$ever_infected <- ifelse(df$ever_infected > 0, 1, 0)
  df$ever_infected_with_symptoms <- ifelse(df$ever_infected_with_symptoms, 1, 0)
  
  ### Aggregate on individual level
  if (aggregate_agents == TRUE){
    df_agg1 <- aggregate(
      x=list(
        "male"=df$male,
        "len_post_infection_chain" = df$len_post_infection_chain,
        "len_post_infection_chain_60" = df$len_post_infection_chain_60,
        "len_post_infection_chain_80" = df$len_post_infection_chain_80,
        "len_pre_infection_chain" = df$len_pre_infection_chain,
        "age" = df$age,
        "household_size" = df$household_size,
        "degree" = df$degree,
        "nace2_section" = df$nace2_section,
        "in_kindergarten_group_1" = df$in_kindergarten_group_1,
        "in_kindergarten_group_2" = df$in_kindergarten_group_2,
        "in_school" = df$in_school,
        "in_work" = df$in_work,
        "in_university" = df$in_university,
        "n_agents_ever_met" = df$n_agents_ever_met,
        "work_hours_day" = df$work_hours_day,
        "ever_infected" = df$ever_infected,
        "hh_income" = df$hh_income,
        "wfh_freq" = df$wfh_freq,
        "tick_infection" = df$tick_infection,
        "n_ever_infected_when_infected" = df$n_ever_infected_when_infected,
        "ever_infected_with_symptoms" = df$ever_infected_with_symptoms,
        "age_group_rki" = df$age_group_rki
      ),
      by=list("unique_agent_id"=df$unique_agent_id),
      FUN=mean,
      na.rm=T,
    )
    
    df_agg2 <- df[
      !duplicated(df$unique_agent_id), 
      c(
        "syear", 
        "pid", 
        "hid", 
        "household_id",
        "unique_agent_id", 
        "unique_household_id", 
        "master_model_index"
      )
    ]
    df <- merge(df_agg1, df_agg2, by="unique_agent_id")
  }
  
  df$len_post_infection_chain <- ceiling(df$len_post_infection_chain)
  df$len_post_infection_chain_60 <- ceiling(df$len_post_infection_chain_60)
  df$len_post_infection_chain_80 <- ceiling(df$len_post_infection_chain_80)
  df$log_len_post_infection_chain <- log(df$len_post_infection_chain + 1)
  df$len_pre_infection_chain <- round(df$len_pre_infection_chain)
  df$tick_infection = round(df$tick_infection)
  
  ### Define/Recode life stages
  df$occ_status = NA
  
  # children
  df[df$in_kindergarten_group_1 == 1, "occ_status"] = "Child"
  df[df$in_kindergarten_group_2 == 1, "occ_status"] = "Child"
  df[df$in_school == 1, "occ_status"] = "Child"
  df[df$in_kindergarten_group_1 == 0 & df$in_kindergarten_group_2 == 0 & df$age < 6, "occ_status"] = "Child"
  
  # working adults
  df[df$in_work == 1, "occ_status"] = "Work"
  df[df$in_university == 1, "occ_status"] = "Work"
  
  # non-working adults
  df[df$in_work == 0 & df$in_university == 0 & df$age >= 20, "occ_status"] = "No Work"
  
  df$occ_status <- factor(df$occ_status)
  df$occ_status <- relevel(df$occ_status, "No Work")
  table(df$occ_status, useNA = "always")

  ### Measure household context
  df$work <- ifelse(df$occ_status=="Work", 1, 0)
  df$no_work <- ifelse(df$occ_status=="No Work", 1, 0)
  df$child <- ifelse(df$occ_status=="Child", 1, 0)
  
  df_hh <- aggregate(
    x=list(
      "work" = df$work,
      "no_work" = df$no_work,
      "child" = df$child
    ),
    by=list("unique_household_id"=df$unique_household_id),
    FUN=sum,
    na.rm=T
  )
  
  names(df_hh) <- c(
    "unique_household_id",
    "work_hhsum", 
    "no_work_hhsum",
    "child_hhsum"
  )
  
  df <- merge(df, df_hh, by="unique_household_id", all.x = TRUE)
  
  df$add_child <- df$child_hhsum - df$child
  df$add_work <- df$work_hhsum - df$work
  df$add_no_work <- df$no_work_hhsum - df$no_work
  
  # recode occ_status into LifeStage
  df$LifeStage <- df$occ_status
  levels(df$LifeStage) <- c("Unemployed adult", "Child", "Employed adult")
  
  # calculate equivalent household income
  df$eq_hh_income <- df$hh_income / (1 + 0.7 * (df$add_work + df$add_no_work -1) + 0.5 * df$add_child)
  
  output_list <- list("df"=df)
  
  if (aggregate_agents==TRUE){
    output_list$df_agg1 <- df_agg1
  }
  
  return(output_list)
}


get_replications <- function(df, f, col){
  set.seed(1)
  
  vec_results <- vector(length=nrow(df))
  
  for (i in 1:nrow(df)){
    df_sample <- df[sample(nrow(df), size=i),]
    result <- f(df_sample[,col])
    vec_results[[i]] <- result
  }
  
  df_results <- data.frame("n"=1:nrow(df), "estimate"=vec_results)
  
  return(df_results)
}
