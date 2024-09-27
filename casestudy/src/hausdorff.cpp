#include <Rcpp.h>

// // [[Rcpp::plugins(openmp)]] // Uncomment on Windows and Linux

// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

// [[Rcpp::export]]
double hausdorff_distance_cpp(Rcpp::NumericVector x, Rcpp::NumericVector y,
                              unsigned int dimension = 1)
{
  unsigned int numDimensions = dimension;
  unsigned int numPoints = x.size() / numDimensions;

  double dX = 0.0;
  double dY = 0.0;

  for (unsigned int i = 0;i < numPoints;++i)
  {
    double min_dist_x = std::numeric_limits<double>::infinity();
    double min_dist_y = std::numeric_limits<double>::infinity();

    for (unsigned int j = 0;j < numPoints;++j)
    {
      double dist_x = 0.0;
      double dist_y = 0.0;

      for (unsigned int k = 0;k < numDimensions;++k)
      {
        unsigned int index_i = k * numPoints + i;
        unsigned int index_j = k * numPoints + j;
        dist_x += std::pow(x[index_i] - y[index_j], 2);
        dist_y += std::pow(y[index_i] - x[index_j], 2);
      }
      min_dist_x = std::min(min_dist_x, dist_x);
      min_dist_y = std::min(min_dist_y, dist_y);
    }

    dX = std::max(dX, min_dist_x);
    dY = std::max(dY, min_dist_y);
  }

  return std::sqrt(std::max(dX, dY));
}

double hausdorff_distance_cpp(RcppParallel::RMatrix<double>::Row x,
                              RcppParallel::RMatrix<double>::Row y,
                              unsigned int dimension = 1)
{
  unsigned int numDimensions = dimension;
  unsigned int numPoints = x.size() / numDimensions;

  double dX = 0.0;
  double dY = 0.0;

  for (unsigned int i = 0;i < numPoints;++i)
  {
    double min_dist_x = std::numeric_limits<double>::infinity();
    double min_dist_y = std::numeric_limits<double>::infinity();

    for (unsigned int j = 0;j < numPoints;++j)
    {
      double dist_x = 0.0;
      double dist_y = 0.0;

      for (unsigned int k = 0;k < numDimensions;++k)
      {
        unsigned index_i = k * numPoints + i;
        unsigned index_j = k * numPoints + j;
        dist_x += std::pow(x[index_i] - y[index_j], 2);
        dist_y += std::pow(y[index_i] - x[index_j], 2);
      }

      min_dist_x = std::min(min_dist_x, dist_x);
      min_dist_y = std::min(min_dist_y, dist_y);
    }

    dX = std::max(dX, min_dist_x);
    dY = std::max(dY, min_dist_y);
  }

  return std::sqrt(std::max(dX, dY));
}

Rcpp::NumericMatrix listToMatrix(Rcpp::List x)
{
  unsigned int nrows = x.size();
  unsigned int ncols = Rcpp::as<Rcpp::NumericVector>(x[0]).size();
  Rcpp::NumericMatrix out(nrows, ncols);
  Rcpp::NumericMatrix workMatrix;
  Rcpp::NumericVector row;

  for (unsigned int i = 0;i < nrows;++i)
  {
    workMatrix = Rcpp::as<Rcpp::NumericMatrix>(x[i]);
    workMatrix = Rcpp::transpose(workMatrix);
    row = Rcpp::as<Rcpp::NumericVector>(workMatrix);
    std::copy(row.begin(), row.end(), out.row(i).begin());
  }

  return out;
}

Rcpp::NumericVector dist_cpp_omp(Rcpp::NumericMatrix x,
                                 unsigned int dimension = 1,
                                 unsigned int ncores = 1)
{
  unsigned int N = x.nrow();
  unsigned int K = N * (N - 1) / 2;
  Rcpp::NumericVector out(K);
  RcppParallel::RMatrix<double> xSafe(x);
  RcppParallel::RVector<double> outSafe(out);

#ifdef _OPENMP
  #pragma omp parallel for num_threads(ncores)
#endif
  for (unsigned int k = 0;k < K;++k)
  {
    unsigned int i = N - 2 - std::floor(std::sqrt(-8 * k + 4 * N * (N - 1) - 7) / 2.0 - 0.5);
    unsigned int j = k + i + 1 - N * (N - 1) / 2 + (N - i) * ((N - i) - 1) / 2;
    outSafe[k] = hausdorff_distance_cpp(xSafe.row(i), xSafe.row(j), dimension);
  }

  out.attr("Size") = N;
  out.attr("Labels") = Rcpp::seq(1, N);
  out.attr("Diag") = false;
  out.attr("Upper") = false;
  out.attr("method") = "hausdorff";
  out.attr("class") = "dist";
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector dist_cpp_omp(Rcpp::List x,
                                 unsigned int dimension = 1,
                                 unsigned int ncores = 1)
{
  Rcpp::NumericMatrix xMatrix = listToMatrix(x);
  return dist_cpp_omp(xMatrix, dimension, ncores);
}

struct HausdorffDistanceComputer : public RcppParallel::Worker
{
  const RcppParallel::RMatrix<double> m_SafeInput;
  RcppParallel::RVector<double> m_SafeOutput;
  unsigned int m_Dimension;

  HausdorffDistanceComputer(const Rcpp::NumericMatrix x,
                    Rcpp::NumericVector out,
                    unsigned int dimension)
    : m_SafeInput(x), m_SafeOutput(out), m_Dimension(dimension) {}

  void operator()(std::size_t begin, std::size_t end)
  {
    unsigned int N = m_SafeInput.nrow();
    for (std::size_t k = begin;k < end;++k)
    {
      unsigned int i = N - 2 - std::floor(std::sqrt(-8 * k + 4 * N * (N - 1) - 7) / 2.0 - 0.5);
      unsigned int j = k + i + 1 - N * (N - 1) / 2 + (N - i) * ((N - i) - 1) / 2;
      m_SafeOutput[k] = hausdorff_distance_cpp(m_SafeInput.row(i), m_SafeInput.row(j), m_Dimension);
    }
  }
};

Rcpp::NumericVector dist_rcppparallel(Rcpp::NumericMatrix x,
                                      unsigned int dimension = 1,
                                      unsigned int ncores = 1)
{
  unsigned int N = x.nrow();
  unsigned int K = N * (N - 1) / 2;
  Rcpp::NumericVector out(K);
  HausdorffDistanceComputer hausdorffDistance(x, out, dimension);
  RcppParallel::parallelFor(0, K, hausdorffDistance, 1, ncores);
  out.attr("Size") = x.nrow();
  out.attr("Labels") = Rcpp::seq(1, x.nrow());
  out.attr("Diag") = false;
  out.attr("Upper") = false;
  out.attr("method") = "hausdorff";
  out.attr("class") = "dist";
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector dist_rcppparallel(Rcpp::List x,
                                      unsigned int dimension = 1,
                                      unsigned int ncores = 1)
{
  Rcpp::NumericMatrix xMatrix = listToMatrix(x);
  return dist_rcppparallel(xMatrix, dimension, ncores);
}
