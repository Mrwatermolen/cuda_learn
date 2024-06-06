#include "ffz/array.cuh"
#include <cstdio>

using ffz::cuda::Array;
using ffz::cuda::ArrayHD;
using Size = std::size_t;
using Real = float;

constexpr Real c_0 = 3e8;
constexpr Size num_cells = 200;
constexpr Size num_time_steps = 1000;
constexpr Real dz = 0.01;
constexpr Real cfl = 1.0;
constexpr Real dt = cfl * dz / c_0;
constexpr Size tfsf_start = 90;
constexpr Size tfsf_end = 160;
constexpr Size reflect_record_index = 85;
constexpr Size transmit_record_index = 165;

template <typename T, std::size_t TimeSteps, std::size_t NumCells>
__global__ void
fdtd1DFixedKernel(Array<T, NumCells + 1> *ex, Array<T, NumCells> *hy,
                  Array<T, NumCells + 1> *cexe, Array<T, NumCells + 1> *cexhy,
                  Array<T, NumCells> *chyh, Array<T, NumCells> *chye,
                  Array<T, TimeSteps> *source,
                  Array<T, TimeSteps> *ex_reflect_monitor,
                  Array<T, TimeSteps> *ex_transmit_monitor,
                  Array<T, TimeSteps *(NumCells + 1)> *ex_line_monitor) {
  // Absorbing boundary condition
  constexpr auto abc_coeff_0 = (c_0 * dt - dz) / (c_0 * dt + dz);
  constexpr auto abc_coeff_1 = 2 * dz / (c_0 * dt + dz);
  Real abc_x = 0;
  Real abc_y = 0;
  Real abc_a = 0;
  Real abc_b = 0;

  // (*ex)[tfsf_start]

  const auto id = blockIdx.x * blockDim.x + threadIdx.x;
  const auto total_threads = gridDim.x * blockDim.x;
  printf("total_threads: %d\n", total_threads);
  if (id == 0) {
    printf("GridDim: %d, BlockDim: %d\n", gridDim.x, blockDim.x);
    printf("NumCells: %lu, NumTimeSteps: %lu\n", NumCells, TimeSteps);
  }

  if (0 <= id && id < NumCells + 1) {
    struct {
      T _start;
      T _end;
    } task;

    task._start = id;
    task._end = id + 1;

    for (Size t = 0; t < TimeSteps; ++t) {
      if (id != 0) {
        for (Size i = task._start; i < task._end; ++i) {
          (*ex)[i] =
              (*cexe)[i] * (*ex)[i] + (*cexhy)[i] * ((*hy)[i] - (*hy)[i - 1]);
        }
      }
      if (id == tfsf_start) {
        (*ex)[tfsf_start] += (*cexhy)[tfsf_start] * (*source)[t];
      }

      for (Size i = tfsf_start; i < tfsf_end; ++i) {
        (*hy)[i] =
            (*chyh)[i] * (*hy)[i] + (*chye)[i] * ((*ex)[i + 1] - (*ex)[i]);
      }

      for (Size i = tfsf_start; i < tfsf_end; ++i) {
        (*ex_line_monitor)[t * (NumCells + 1) + i] = (*ex)[i];
      }

      if (id == reflect_record_index) {
        (*ex_reflect_monitor)[t] = (*ex)[reflect_record_index];
      }

      if (id == transmit_record_index) {
        (*ex_transmit_monitor)[t] = (*ex)[transmit_record_index];
      }

      if (id == 0) {
        printf("\rTime: %lu / %lu\n", t + 1, TimeSteps);
      }

      __syncthreads();
    }
  }
};

auto fdtd1DFixed() -> void {

  constexpr Real l_min = 20 * dz;
  constexpr Real f_max = c_0 / (l_min);
  constexpr Real eps_0 = 8.854e-12;
  constexpr Real mu_0 = 4 * M_PI * 1e-7;
  constexpr Real sigma_e_zero = 1e-15;
  constexpr Real sigma_m_zero = 1e-15;
  constexpr Size slab_start = 100;
  constexpr Size slab_end = 150;

  auto ex_hd = ArrayHD<Real, num_cells + 1>();
  auto hy_hd = ArrayHD<Real, num_cells>();
  ex_hd.allocateHost();
  hy_hd.allocateHost();
  for (auto &&i : *ex_hd.host()) {
    i = 0.0;
  }
  for (auto &&i : *hy_hd.host()) {
    i = 0.0;
  }
  ex_hd.copyHostToDevice();
  hy_hd.copyHostToDevice();

  // material parameters
  auto eps_hd = ArrayHD<Real, num_cells + 1>();
  auto mu_hd = ArrayHD<Real, num_cells>();
  auto sigma_e_hd = ArrayHD<Real, num_cells + 1>();
  auto sigma_m_hd = ArrayHD<Real, num_cells>();
  eps_hd.allocateHost();
  mu_hd.allocateHost();
  sigma_e_hd.allocateHost();
  sigma_m_hd.allocateHost();
  auto &eps = *eps_hd.host();
  auto &mu = *mu_hd.host();
  auto &sigma_e = *sigma_e_hd.host();
  auto &sigma_m = *sigma_m_hd.host();

  // initialize material parameters
  constexpr Real air_eps_r = 1.0;
  constexpr Real air_mu_r = 1.0;
  constexpr Real slab_eps_r = 4.0;
  constexpr Real slab_mu_r = 1.0;
  for (Size i = 0; i < num_cells + 1; ++i) {
    eps[i] = air_eps_r * eps_0;
    sigma_e[i] = sigma_e_zero;
  }
  for (Size i = 0; i < num_cells; ++i) {
    mu[i] = air_mu_r * mu_0;
    sigma_m[i] = sigma_m_zero;
  }
  // fill slab
  for (Size i = slab_start; i < slab_end + 1; ++i) {
    eps[i] = slab_eps_r * eps_0;
    mu[i] = slab_mu_r * mu_0;
    sigma_e[i] = sigma_e_zero;
    sigma_m[i] = sigma_m_zero;
  }

  // coefficients
  auto cexe_hd = ArrayHD<Real, num_cells + 1>();
  auto cexhy_hd = ArrayHD<Real, num_cells + 1>();
  auto chyh_hd = ArrayHD<Real, num_cells>();
  auto chye_hd = ArrayHD<Real, num_cells>();
  cexe_hd.allocateHost();
  cexhy_hd.allocateHost();
  chyh_hd.allocateHost();
  chye_hd.allocateHost();
  auto &cexe = *cexe_hd.host();
  auto &cexhy = *cexhy_hd.host();
  auto &chyh = *chyh_hd.host();
  auto &chye = *chye_hd.host();

  // initialize coefficients
  for (Size i = 0; i < num_cells + 1; ++i) {
    cexe[i] = (2 * eps[i] - dt * sigma_e[i]) / (2 * eps[i] + dt * sigma_e[i]);
    cexhy[i] = -dt / (eps[i] + dt * sigma_e[i]) / dz;
  }
  for (Size i = 0; i < num_cells; ++i) {
    chyh[i] = (2 * mu[i] - dt * sigma_m[i]) / (2 * mu[i] + dt * sigma_m[i]);
    chye[i] = -dt / (mu[i] + dt * sigma_m[i]) / dz;
  }
  cexe_hd.copyHostToDevice();
  cexhy_hd.copyHostToDevice();
  chyh_hd.copyHostToDevice();
  chye_hd.copyHostToDevice();

  // source
  auto source_hd = ArrayHD<Real, num_time_steps>();
  source_hd.allocateHost();
  auto &source = *source_hd.host();

  // initialize source
  auto source_func = [l_min](const Real t) {
    constexpr Real tau = l_min / 6e8;
    constexpr Real t_0 = 4.5 * tau;
    return std::exp(-std::pow((t - t_0) / tau, 2));
  };
  for (Size i = 0; i < num_time_steps; ++i) {
    source[i] = source_func((i + 0.5) * dt);
  }
  source_hd.copyHostToDevice();

  // record
  auto ex_reflect_monitor_hd = ArrayHD<Real, num_time_steps>();
  auto ex_transmit_monitor_hd = ArrayHD<Real, num_time_steps>();
  auto ex_line_monitor_hd = ArrayHD<Real, num_time_steps *(num_cells + 1)>();
  ex_reflect_monitor_hd.allocateHost();
  ex_transmit_monitor_hd.allocateHost();
  ex_line_monitor_hd.allocateHost();
  auto &ex_reflect_monitor = *ex_reflect_monitor_hd.host();
  auto &ex_transmit_monitor = *ex_transmit_monitor_hd.host();
  auto &ex_line_monitor = *ex_line_monitor_hd.host();

  // initialize record
  for (Size i = 0; i < num_time_steps; ++i) {
    ex_reflect_monitor[i] = 0.0;
    ex_transmit_monitor[i] = 0.0;
  }
  for (Size i = 0; i < num_time_steps * (num_cells + 1); ++i) {
    ex_line_monitor[i] = 0.0;
  }
  ex_reflect_monitor_hd.copyHostToDevice();
  ex_transmit_monitor_hd.copyHostToDevice();
  ex_line_monitor_hd.copyHostToDevice();

  // kernel
  constexpr Size block_size = 256;
  const Size num_blocks = (num_cells + block_size - 1) / block_size;
  printf("Kenel launch: num_blocks: %zu, block_size: %zu\n", num_blocks,
         block_size);
  fdtd1DFixedKernel<Real, num_time_steps, num_cells>
      <<<num_blocks, block_size>>>(
          ex_hd.device(), hy_hd.device(), cexe_hd.device(), cexhy_hd.device(),
          chyh_hd.device(), chye_hd.device(), source_hd.device(),
          ex_reflect_monitor_hd.device(), ex_transmit_monitor_hd.device(),
          ex_line_monitor_hd.device());

  // copy back
  printf("Copy back\n");
  ex_hd.copyDeviceToHost();
  hy_hd.copyDeviceToHost();
  ex_reflect_monitor_hd.copyDeviceToHost();
  ex_transmit_monitor_hd.copyDeviceToHost();
  ex_line_monitor_hd.copyDeviceToHost();
}

int main() {
  fdtd1DFixed();
  return 0;
}
