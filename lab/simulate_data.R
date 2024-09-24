library(purrr)
library(roahd)
library(withr)

N <- 100
P <- 200
L <- 3
time_grid <- seq( 0, 1, length.out = P )
C1 <- exp_cov_function(time_grid, alpha = 0.1, beta = 0.2)
C2 <- exp_cov_function(time_grid, alpha = 0.2, beta = 0.5)
C3 <- exp_cov_function(time_grid, alpha = 0.3, beta = 1)
centerline <- matrix(c(
  sin(2 * pi * time_grid),
  sqrt(time_grid),
  10 * (time_grid - 0.5) * time_grid),
  nrow = 3, byrow = TRUE
)

withr::with_seed(1234, {
  dat <- generate_gauss_mfdata(
    N, L, centerline,
    correlations = c(0.5, 0.5, 0.5),
    listCov = list(C1, C2, C3)
  )
})

mfdat <- mfData(time_grid, dat)
saveRDS(mfdat, file = "lab/mfdat.rds")

dat <- purrr::map(1:N, \(n) {
  rbind(dat[[1]][n, ], dat[[2]][n, ], dat[[3]][n, ])
})
saveRDS(dat, file = "lab/dat.rds")
