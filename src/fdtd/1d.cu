#include <chrono>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "fdtd_update_scheme.cuh"
#include "fz/tensor.cuh"

using fz::cuda::Tensor;
using fz::cuda::TensorHD;
using Size = std::size_t;
using Real = float;

constexpr Real c_0 = 3e8;
constexpr Real dz = 0.01;
constexpr Real cfl = 1.0;
constexpr Real dt = cfl * dz / c_0;
constexpr Size tfsf_start = 50;
constexpr Size reflect_record_index = 45;
constexpr Size transmit_record_index = 75;

constexpr Real l_min = 20 * dz;
constexpr Real eps_0 = 8.854e-12;
constexpr Real mu_0 = 4 * 3.141593 * 1e-7;
constexpr Real sigma_e_zero = 1e-15;
constexpr Real sigma_m_zero = 1e-15;
constexpr Size slab_start = 60;

using Array1D = Tensor<Real, 1>;
using Array2D = Tensor<Real, 2>;

__global__ void fdtd1DFixedKernel(Array1D *ex, Array1D *hy, Array1D *cexe,
                                  Array1D *cexhy, Array1D *chyh, Array1D *chye,
                                  Array1D *source, Array1D *ex_reflect_monitor,
                                  Array1D *ex_transmit_monitor,
                                  Array2D *ex_line_monitor,
                                  std::size_t gif_step) {
  // Absorbing boundary condition
  constexpr auto abc_coeff_0 = (c_0 * dt - dz) / (c_0 * dt + dz);
  constexpr auto abc_coeff_1 = 2 * dz / (c_0 * dt + dz);
  constexpr auto abc_coeff_2 =
      (0.5 * c_0 * dt - dz) / (0.5 * c_0 * dt + dz);  // slab epsilon_r = 4
  constexpr auto abc_coeff_3 = 2 * dz / (0.5 * c_0 * dt + dz);

  Real abc_a = 0;
  Real abc_b = 0;
  Real abc_c = 0;
  Real abc_d = 0;

  const auto id = threadIdx.x;
  if (id == 0) {
    printf("GridDim: %d, BlockDim: %d\n", gridDim.x, blockDim.x);
    printf("NumCells: %lu, TimeSteps: %lu\n", ex->size() - 1, source->size());
  }
  const auto num_cells = ex->size() - 1;
  const auto time_steps = source->size();

  if (id < num_cells) {
    struct {
      Size _start;
      Size _end;
    } task;

    // task._start = id;
    // task._end = id + 1;
    auto remainder = num_cells % blockDim.x;
    auto quotient = num_cells / blockDim.x;
    if (id < remainder) {
      task._start = id * (quotient + 1);
      task._end = (id + 1) * (quotient + 1);
    } else {
      task._start = id * quotient + remainder;
      task._end = (id + 1) * quotient + remainder;
    }
    // printf("Thread: %d, Start: %lu, End: %lu\n", id, task._start, task._end);

    for (Size t = 0; t < time_steps; ++t) {
      Real abc_p = (*ex)[num_cells - 1];
      Real abc_q = (*ex)[num_cells];
      Real abc_r = (*ex)[1];
      Real abc_s = (*ex)[0];

      // Update E
      one_dimensional::updateE(*cexe, *cexhy, *ex, *hy, task._start, task._end);

      one_dimensional::updateELeft(*cexe, *cexhy, *ex, *hy, task._start);

      __syncthreads();

      if (id == 0) {
        // absorbing boundary condition
        (*ex)[0] = -abc_c + abc_coeff_0 * ((*ex)[1] + abc_d) +
                   abc_coeff_1 * (abc_r + abc_s);
        (*ex)[num_cells] = -abc_a +
                           abc_coeff_2 * ((*ex)[num_cells - 1] + abc_b) +
                           abc_coeff_3 * (abc_p + abc_q);
        abc_a = abc_p;
        abc_b = abc_q;
        abc_c = abc_r;
        abc_d = abc_s;

        // TF/SF
        Real coeff_e = dt / (dz * 8.854e-12);
        Real h_i = (*source)[t] / (377.0);

        (*ex)[tfsf_start] += coeff_e * h_i;
      }

      __syncthreads();

      // Update H
      one_dimensional::updateH(*chyh, *chye, *hy, *ex, task._start, task._end);

      __syncthreads();

      if (id == 0) {
        // TF/SF
        Real coeff_h = dt / (dz * 4 * 3.141593 * 1e-7);
        Real e_i = (*source)[t];

        (*hy)[tfsf_start - 1] += coeff_h * e_i;
      }

      __syncthreads();
      if (t % gif_step == 0) {
        auto k = t / gif_step;
        for (Size i = 0; i < num_cells + 1; ++i) {
          (*ex_line_monitor)(k, i) = (*ex)[i];
        }
      }

      if (id == reflect_record_index) {
        (*ex_reflect_monitor)[t] = (*ex)[reflect_record_index];
      }

      if (id == transmit_record_index) {
        (*ex_transmit_monitor)[t] = (*ex)[transmit_record_index];
      }

      if (id == 0) {
        printf("\rTime: %lu / %lu\n", t + 1, time_steps);
      }

      __syncthreads();
    }
  }
};

auto fdtd1D(std::size_t num_cells, std::size_t num_time_steps) -> void {
  auto ex_hd = TensorHD<Real, 1>({num_cells + 1});
  auto hy_hd = TensorHD<Real, 1>({num_cells});
  for (auto &&i : *ex_hd.host()) {
    i = 0.0;
  }
  for (auto &&i : *hy_hd.host()) {
    i = 0.0;
  }
  ex_hd.copyHostToDevice();
  hy_hd.copyHostToDevice();

  // material parameters
  auto eps_hd = TensorHD<Real, 1>({num_cells + 1});
  auto mu_hd = TensorHD<Real, 1>({num_cells});
  auto sigma_e_hd = TensorHD<Real, 1>({num_cells + 1});
  auto sigma_m_hd = TensorHD<Real, 1>({num_cells});
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
  for (Size i = slab_start; i < num_cells + 1; ++i) {
    eps[i] = slab_eps_r * eps_0;
    sigma_e[i] = sigma_e_zero;
  }
  for (Size i = slab_start; i < num_cells; ++i) {
    mu[i] = slab_mu_r * mu_0;
    sigma_m[i] = sigma_m_zero;
  }

  // coefficients
  auto cexe_hd = TensorHD<Real, 1>({num_cells + 1});
  auto cexhy_hd = TensorHD<Real, 1>({num_cells + 1});
  auto chyh_hd = TensorHD<Real, 1>({num_cells});
  auto chye_hd = TensorHD<Real, 1>({num_cells});
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
  auto source_hd = TensorHD<Real, 1>({num_time_steps});
  auto &source = *source_hd.host();

  // initialize source
  auto source_func = [](const Real t) {
    constexpr Real tau = l_min / 6e8;
    constexpr Real t_0 = 4.5 * tau;
    return std::exp(-std::pow((t - t_0) / tau, 2));
  };
  for (Size i = 0; i < num_time_steps; ++i) {
    source[i] = source_func((i + 0.5) * dt);
  }
  source_hd.copyHostToDevice();

  // record
  auto ex_reflect_monitor_hd = TensorHD<Real, 1>({num_time_steps});
  auto ex_transmit_monitor_hd = TensorHD<Real, 1>({num_time_steps});
  const auto gif_step = 3;
  auto ex_line_monitor_hd =
      TensorHD<Real, 2>({(num_time_steps / gif_step), num_cells + 1});
  auto &ex_reflect_monitor = *ex_reflect_monitor_hd.host();
  auto &ex_transmit_monitor = *ex_transmit_monitor_hd.host();
  auto &ex_line_monitor = *ex_line_monitor_hd.host();

  // initialize record
  for (Size i = 0; i < num_time_steps; ++i) {
    ex_reflect_monitor[i] = 0.0;
    ex_transmit_monitor[i] = 0.0;
  }
  for (Size i = 0; i < (num_time_steps / gif_step); ++i) {
    for (Size j = 0; j < num_cells + 1; ++j) {
      ex_line_monitor(i, j) = 0.0;
    }
  }
  ex_reflect_monitor_hd.copyHostToDevice();
  ex_transmit_monitor_hd.copyHostToDevice();
  ex_line_monitor_hd.copyHostToDevice();

  // kernel
  constexpr Size num_blocks = 1;  // don't support multi-block
  constexpr Size block_size = 128;
  printf("Kenel launch: num_blocks: %zu, block_size: %zu\n", num_blocks,
         block_size);
  fdtd1DFixedKernel<<<num_blocks, block_size>>>(
      ex_hd.device(), hy_hd.device(), cexe_hd.device(), cexhy_hd.device(),
      chyh_hd.device(), chye_hd.device(), source_hd.device(),
      ex_reflect_monitor_hd.device(), ex_transmit_monitor_hd.device(),
      ex_line_monitor_hd.device(), gif_step);
  cudaDeviceSynchronize();

  // copy back
  printf("Kernel finished\n");
  ex_hd.copyDeviceToHost();
  hy_hd.copyDeviceToHost();
  ex_reflect_monitor_hd.copyDeviceToHost();
  ex_transmit_monitor_hd.copyDeviceToHost();
  ex_line_monitor_hd.copyDeviceToHost();

  // output
  printf("Output\n");
  std::filesystem::path dir("tmp");
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directory(dir);
  }
  printf("Output Directory: %s\n", (std::filesystem::absolute(dir)).c_str());

  std::fstream file((dir / "ex_line_monitor.dat").string(), std::ios::out);
  for (Size t = 0; t < (num_time_steps / gif_step); ++t) {
    for (Size i = 0; i < num_cells + 1; ++i) {
      file << ex_line_monitor.at(t, i) << " ";
    }
    file << "\n";
  }

  auto output_func = [](const auto &data, const auto filename) {
    std::fstream file(filename, std::ios::out);
    for (Size i = 0; i < data.size(); ++i) {
      file << data[i] << "\n";
    }
    file.close();
  };

  output_func(ex_reflect_monitor, (dir / "ex_reflect_monitor.dat").string());
  output_func(ex_transmit_monitor, (dir / "ex_transmit_monitor.dat").string());
  output_func(source, (dir / "incident_wave.dat").string());
  auto time = Tensor<Real, 1>({num_time_steps});
  for (Size i = 0; i < num_time_steps; ++i) {
    time[i] = i * dt;
  }
  output_func(time, (dir / "time.dat").string());

  printf("Finished\n");
}

int main(int argc, char *argv[]) {
  std::size_t num_cells = 100;
  std::size_t num_time_steps = 500;
  if (argc > 2) {
    num_cells = std::stoul(argv[1]);
    num_time_steps = std::stoul(argv[2]);
  }

  std::chrono::high_resolution_clock::time_point start_time =
      std::chrono::high_resolution_clock::now();
  fdtd1D(num_cells, num_time_steps);
  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  printf("Duration: %ld ms\n", duration.count());
  return 0;
}
