library(modEvA)
library(FNN)
library(FSA)

# ---- Settings ----------------------------------------------------------
base_path        <- "/marbec-data/RLS-Australia/malpolon"
inductive_run    <- "42_ind_pa_seedtest-2026-03-12_17-19"
transductive_run <- "42_td_pa_seedtest-2026-03-11_14-31"
seeds            <- 1:5
min_presences    <- 1
matched_species  <- TRUE   # TRUE: only species with a gain in all three bins
distance_bins    <- c("near", "mid", "far")

# ---- Data --------------------------------------------------------------
database <- read.csv(file.path(base_path, "inputs/australia/database_common.csv"),
                     row.names = 1, check.names = FALSE)
env      <- read.csv(file.path(base_path, "xgb/X_compound_values_mm.csv"),
                     row.names = 1, check.names = FALSE)

species  <- names(database)[(ncol(database) - 817):(ncol(database) - 1)]
is_test  <- database[rownames(env), "subset"] == "test"
test_ids <- rownames(env)[is_test]
presence <- 1L * (as.matrix(database[test_ids, species]) > 0)   # surveys x species

# ---- Distance to labelled surveys, location terciles -------------------
X_trainval <- as.matrix(env[!is_test, ])
X_test     <- as.matrix(env[is_test, ])
medians    <- apply(X_trainval, 2, median, na.rm = TRUE)
impute     <- function(x) { x[is.na(x)] <- medians[col(x)[is.na(x)]]; x }
X_trainval <- impute(X_trainval)
X_test     <- impute(X_test)

keep       <- which(apply(X_trainval, 2, sd) > 0)
Z_trainval <- scale(X_trainval[, keep, drop = FALSE])
Z_test     <- scale(X_test[, keep, drop = FALSE],
                    center = attr(Z_trainval, "scaled:center"),
                    scale  = attr(Z_trainval, "scaled:scale"))

nearest        <- knnx.dist(Z_trainval, Z_test, k = 1)[, 1]
test_locations <- as.character(database[test_ids, "location"])
loc_distance   <- tapply(nearest, test_locations, median)
loc_bin <- setNames(
  cut(loc_distance, quantile(loc_distance, c(0, 1/3, 2/3, 1)),
      labels = distance_bins, include.lowest = TRUE),
  names(loc_distance))

loc_by_bin <- split(names(loc_bin), loc_bin)[distance_bins]
ids_by_loc <- split(test_ids, test_locations)
ids_of     <- function(locs) unlist(ids_by_loc[locs], use.names = FALSE)

print(t(sapply(loc_by_bin, function(l)
  c(n_locations = length(l), min = min(loc_distance[l]), max = max(loc_distance[l])))))

# ---- Predictions -------------------------------------------------------
load_predictions <- function(run) lapply(seeds, function(s) {
  path <- file.path(base_path, "outputs", run, paste0("Seed-", s), "predictions-probs.csv")
  p <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE)[test_ids, species])
  storage.mode(p) <- "numeric"
  p
})
predictions <- list(inductive    = load_predictions(inductive_run),
                    transductive = load_predictions(transductive_run))

# ---- CBI and gain ------------------------------------------------------
cbi <- function(obs, pred) {
  if (sum(obs == 1) < min_presences || !any(obs == 0)) return(NA_real_)
  tryCatch(Boyce(obs = obs, pred = pred, plot = FALSE)$Boyce,
           error = function(e) NA_real_)
}

# Transductive minus inductive CBI for a set of surveys: seeds x species matrix.
# Seed k of each model is paired with seed k of the other; the pairing is arbitrary.
gain_by_seed <- function(ids) {
  rows <- match(ids, test_ids)
  sapply(species, function(sp) {
    obs <- as.numeric(presence[rows, sp])
    # sapply(seq_along(seeds), function(k)
      cbi(obs, predictions$transductive[[k]][rows, sp]) -
        cbi(obs, predictions$inductive[[k]][rows, sp]))
  })
}

seed_gains <- lapply(loc_by_bin, \(locs) gain_by_seed(ids_of(locs)))

# Seed-averaged gain per species (NA if any seed failed)
gains <- lapply(seed_gains, function(m) { g <- colMeans(m); g[!is.na(g)] })
if (matched_species) {
  common <- Reduce(intersect, lapply(gains, names))
  gains  <- lapply(gains, \(g) g[common])
}

gain_summary <- data.frame(bin = distance_bins,
                           n_locations = lengths(loc_by_bin),
                           n_species   = lengths(gains),
                           mean_gain   = sapply(gains, mean),
                           median_gain = sapply(gains, median),
                           row.names = NULL)
print(gain_summary, digits = 3)

# ---- Do the terciles differ? Kruskal-Wallis + Dunn post-hoc ------------
use_seeds <- TRUE

long <- do.call(rbind, lapply(distance_bins, function(bin) {
  m <- seed_gains[[bin]][, names(gains[[bin]]), drop = FALSE]   # species kept in the summary
  if (!use_seeds) m <- t(colMeans(m))
  data.frame(bin     = factor(bin, levels = distance_bins),
             species = rep(colnames(m), each = nrow(m)),
             gain    = as.vector(m))
}))
long <- long[!is.na(long$gain), ]

dunn <- FSA::dunnTest(gain ~ bin, data = long, method = "holm")
print(dunn$res, digits = 3)   # check the sign of Z against the median gains above
