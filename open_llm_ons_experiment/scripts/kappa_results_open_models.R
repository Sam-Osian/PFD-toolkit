library(dplyr)
library(readr)
library(irr)

# Mirror ONS replication approach:
# recover SE from kappa/z and compute 95% Wald CI.

predictions_path <- "open_llm_ons_experiment/artifacts/report_level_predictions.csv"
results_out_path <- "open_llm_ons_experiment/artifacts/kappa_by_model.csv"

if (!file.exists(predictions_path)) {
  stop("Missing predictions file: ", predictions_path)
}

pred <- read_csv(predictions_path, show_col_types = FALSE) %>%
  filter(status == "completed") %>%
  mutate(
    consensus = as.logical(consensus),
    model_pred = as.logical(model_pred)
  ) %>%
  filter(!is.na(consensus), !is.na(model_pred))

if (nrow(pred) == 0) {
  stop("No completed prediction rows found.")
}

safe_kappa <- function(df) {
  a <- ifelse(df$consensus, "Yes", "No")
  b <- ifelse(df$model_pred, "Yes", "No")
  pair_df <- data.frame(
    consensus = factor(a, levels = c("No", "Yes")),
    model = factor(b, levels = c("No", "Yes"))
  )

  k_out <- kappa2(pair_df, weight = "unweighted")
  se <- abs(k_out$value / k_out$statistic)
  ci <- k_out$value + c(-1.96, 1.96) * se

  tibble(
    n_reports = nrow(pair_df),
    raw_agreement = mean(pair_df$consensus == pair_df$model, na.rm = TRUE),
    cohen_kappa = k_out$value,
    ci_lower = ci[1],
    ci_upper = ci[2]
  )
}

kappa_by_model <- pred %>%
  group_by(model, tag, family) %>%
  group_modify(~ safe_kappa(.x)) %>%
  ungroup() %>%
  arrange(desc(cohen_kappa))

write_csv(kappa_by_model, results_out_path)
cat("Wrote:", results_out_path, "\n")
