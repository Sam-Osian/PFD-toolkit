# install.packages(c("readxl","dplyr","tidyr","purrr","irr"))
library(readxl)
library(dplyr)
library(tidyr)
library(purrr)
library(irr)

# irr doesn’t return a confidence interval for k.
# It gives k and its z-statistic (z = k / SE), but not SE or variance.
# We therefore recover SE as 'k / z' and build a 95% Wald CI: k +/- 1.96*SE.


# Read data
# ...data is an Excel spreadsheet with multiple tabs, but 'Randomised w 
#    Annotations' is the only tab we need.

file_path  <- "Downloads/PFD Toolkit Consensus Annotations.xlsx"
sheet_name <- "Consensus annotations"

raw <- read_excel(file_path, sheet = sheet_name)

# Keep only the annotation cols + Toolkit output and put them in a tidier form 

annotation_cols <- c("Dan: child & suicide", 
                     "Arpan: child & suicide", 
                     "Sahil: child & suicide")

df <- raw %>% 
  select(`PFD Toolkit: child suicide?`,
         `Post-consensus verdict: Is this a child suicide case? (Yes or No)`,
         all_of(annotation_cols)) %>% 
  # trim leading/trailing spaces in all selected columns (handles stray spaces)
  mutate(across(everything(), ~ trimws(as.character(.x)))) %>% 
  mutate(across(everything(),
                ~factor(.x, levels = c("No","Yes")))) %>% 
  # short, convenient aliases for analysis while preserving original names above
  rename(Toolkit  = `PFD Toolkit: child suicide?`,
         Consensus = `Post-consensus verdict: Is this a child suicide case? (Yes or No)`)


# Calculations!

## For each clin, raw agreement & Cohen's k vs. Toolkit 

### Documentation for kappa2(): https://www.rdocumentation.org/packages/irr/versions/0.84.1/topics/kappa2

agreement_vs_toolkit <- map_dfr(
  annotation_cols,
  function(cln){
    rater_col <- cln
    k_out     <- kappa2(df[, c(rater_col, "Toolkit")], weight = "unweighted")
    se        <- abs(k_out$value / k_out$statistic)
    ci        <- k_out$value + c(-1.96, 1.96) * se
    tibble(
      Clinician       = cln,
      Raw_Agreement   = mean(df[[rater_col]] == df$Toolkit, na.rm = TRUE),
      Cohen_Kappa     = k_out$value,
      CI_lower        = ci[1],
      CI_upper        = ci[2]
    )
  }
)

## Consensus vs. Toolkit: raw agreement & Cohen's kappa

### Documentation for kappa2(): https://www.rdocumentation.org/packages/irr/versions/0.84.1/topics/kappa2

consensus_out <- kappa2(df[, c("Consensus", "Toolkit")], weight = "unweighted")
consensus_se  <- abs(consensus_out$value / consensus_out$statistic)
consensus_ci  <- consensus_out$value + c(-1.96, 1.96) * consensus_se

consensus_vs_toolkit <- tibble(
  Raw_Agreement = mean(df$Consensus == df$Toolkit, na.rm = TRUE),
  Cohen_Kappa   = consensus_out$value,
  CI_lower      = consensus_ci[1],
  CI_upper      = consensus_ci[2]
)

## Overall three-rater Fleiss' kappa (inter-rater agreement)

### Docs for kappam.fleiss(): https://www.rdocumentation.org/packages/irr/versions/0.84.1/topics/kappam.fleiss

fleiss_out <- kappam.fleiss(df %>% select(all_of(annotation_cols)), 
                            detail = TRUE)

fleiss_se <- abs(fleiss_out$value / fleiss_out$statistic)
fleiss_ci <- fleiss_out$value + c(-1.96, 1.96) * fleiss_se

fleiss_stats <- tibble(
  Fleiss_Kappa = fleiss_out$value,
  CI_lower     = fleiss_ci[1],
  CI_upper     = fleiss_ci[2]
)

# Display results, and hope they're good :) 

cat("\n--- Individual Raw Agreement & Cohen's kappa vs. PFD Toolkit
    ...we're not reporting on this in the paper, but it's useful to see\n")
print(agreement_vs_toolkit)

cat("\n--- Inter-rater agreement (Fleiss' kappa across all annotation_cols\n")
print(fleiss_stats)

cat("\n--- Raw Agreement & Cohen's kappa: Toolkit vs. Consensus\n")
print(consensus_vs_toolkit)



# -------------------------------------------------------------------------
# Publication window tabulation for POSITIVE Toolkit cases only


# Preconditions
if(!"Date" %in% names(raw)){
  stop("Expected a 'Date' column in the sheet (YYYY-MM-DD).")
}
if(!"PFD Toolkit: child suicide?" %in% names(raw)){
  stop("Expected a 'PFD Toolkit: child suicide?' column in the sheet.")
}

# Filter to positive Toolkit calls (robust to stray spaces/case)
pos_raw <- raw %>%
  mutate(`PFD Toolkit: child suicide?` = trimws(as.character(`PFD Toolkit: child suicide?`))) %>%
  filter(tolower(`PFD Toolkit: child suicide?`) == "yes")

if(nrow(pos_raw) == 0){
  cat("\n--- Publication window counts for positive Toolkit cases\n")
  cat("No positive Toolkit cases found.\n")
} else {
  # Enforce ISO date format and parse
  dates_df <- pos_raw %>%
    mutate(
      Date_raw    = as.character(Date),
      Date_trim   = trimws(Date_raw),
      Date_is_iso = grepl("^\\d{4}-\\d{2}-\\d{2}$", Date_trim),
      Date_parsed = ifelse(Date_is_iso, as.character(Date_trim), NA_character_),
      Date_parsed = as.Date(Date_parsed, format = "%Y-%m-%d")
    )
  
  # Define window: inclusive 2015-01-01 through 2023-11-30
  start_date <- as.Date("2015-01-01")
  end_date   <- as.Date("2023-11-30")
  
  dates_df <- dates_df %>%
    mutate(
      Publication_Window = dplyr::case_when(
        is.na(Date_parsed)            ~ "Invalid/Unknown",
        Date_parsed <  start_date     ~ "Before Jan 2015",
        Date_parsed <= end_date       ~ "Jan 2015–Nov 2023",
        Date_parsed >  end_date       ~ "After Nov 2023"
      )
    )
  
  # Counts among positive cases
  pub_window_counts <- dates_df %>%
    count(Publication_Window, name = "n") %>%
    arrange(match(Publication_Window,
                  c("Before Jan 2015","Jan 2015–Nov 2023","After Nov 2023","Invalid/Unknown")))
  
  # Pull specific buckets (default 0 if absent)
  in_n      <- pub_window_counts$n[match("Jan 2015–Nov 2023", pub_window_counts$Publication_Window)]; if(is.na(in_n)) in_n <- 0
  before_n  <- pub_window_counts$n[match("Before Jan 2015",   pub_window_counts$Publication_Window)]; if(is.na(before_n)) before_n <- 0
  invalid_n <- pub_window_counts$n[match("Invalid/Unknown",   pub_window_counts$Publication_Window)]; if(is.na(invalid_n)) invalid_n <- 0
  
  cat("\n--- Publication window counts for positive Toolkit cases (YYYY-MM-DD enforced)\n")
  cat(sprintf("Within Jan 2015–Nov 2023: %d\n", in_n))
  cat(sprintf("Before Jan 2015: %d\n", before_n))
  if(invalid_n > 0){
    cat(sprintf("\n*** WARNING: %d 'Date' entries not in strict YYYY-MM-DD format and were excluded from windowing.\n", invalid_n))
  }
}

