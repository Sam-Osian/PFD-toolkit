library(readr)
library(irr)

# Usage:
# Rscript kappa_single_model.R <input_csv>
# input_csv columns: consensus, model_pred (values: Yes/No)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript kappa_single_model.R <input_csv>")
}

in_path <- args[[1]]
df <- read_csv(in_path, show_col_types = FALSE)

df$consensus <- factor(df$consensus, levels = c("No", "Yes"))
df$model_pred <- factor(df$model_pred, levels = c("No", "Yes"))

# Same approach as original kappa_results.R:
# recover SE from z and build 95% Wald CI.
k_out <- kappa2(df[, c("consensus", "model_pred")], weight = "unweighted")
se <- abs(k_out$value / k_out$statistic)
ci <- k_out$value + c(-1.96, 1.96) * se

cat(sprintf("%0.12f,%0.12f,%0.12f", k_out$value, ci[1], ci[2]))
