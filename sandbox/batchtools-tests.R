library(future)
plan(list(
  tweak(
    future.batchtools::batchtools_slurm, 
    workers = "ccipl", 
    resources = list(
      ncpus = 20L, 
      walltime = 3600,
      partition = "SMP-short",
      omp.threads = 20L
    )
  ),
  tweak(multisession, workers = 20L)
))

furrr::future_walk(1:1000, \(.x) {
  Sys.sleep(1)
})

plan(sequential)
