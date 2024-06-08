#include <cassert>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <fz/common.cuh>
#include <fz/tensor.cuh>

#include "fdtd_update_scheme.cuh"

using Real = float;
using SizeType = std::size_t;
using Array2D = fz::cuda::Tensor<Real, 2>;

using namespace utl;

template <Field F, Axis A>
FZ_CUDA_DUAL static auto constructFieldArray2D(const SizeType& nx,
                                               const SizeType& ny) {
  if constexpr (F == Field::E) {
    if constexpr (A == Axis::Z) {
      return Array2D({nx + 1, ny + 1});
    }
  } else if constexpr (F == Field::H) {
    if constexpr (A == Axis::X) {
      return Array2D({nx, ny + 1});
    } else if constexpr (A == Axis::Y) {
      return Array2D({nx + 1, ny});
    }
  }
}

__global__ void kernel() {
  auto nx = SizeType{10};
  auto ny = SizeType{10};
  constexpr auto field_e = Field::E;
  constexpr auto axis_z = Axis::Z;
  auto ec = constructFieldArray2D<field_e, axis_z>(nx, ny);  // ez
  auto ha =
      constructFieldArray2D<dualField(field_e), axisA(axis_z)>(nx, ny);  // hx
  auto hb =
      constructFieldArray2D<dualField(field_e), axisB(axis_z)>(nx, ny);  // hy

  assert(ec.shape()[0] == nx + 1);
  assert(ec.shape()[1] == ny + 1);
  assert(ha.shape()[0] == nx);
  assert(ha.shape()[1] == ny + 1);
  assert(hb.shape()[0] == nx + 1);
  assert(hb.shape()[1] == ny);

  printf("Success\n");
  printf("ec.shape() = {%lu, %lu}\n", ec.shape()[0], ec.shape()[1]);
  printf("ha.shape() = {%lu, %lu}\n", ha.shape()[0], ha.shape()[1]);
  printf("hb.shape() = {%lu, %lu}\n", hb.shape()[0], hb.shape()[1]);
}

void constructMaterialSPace() {}

void FDTD2d() {
  const SizeType nx = 180;
  const SizeType ny = 180;
  const SizeType nt = 800;
  constexpr Real center_frequency{12e9};
  constexpr Real max_frequency{20e9};
  constexpr Real min_lambda{3e8 / max_frequency};
  constexpr Real bandwidth{2 * center_frequency};
  constexpr Real dx{min_lambda / 20};
  constexpr Real dy{dx};
  constexpr Real dt = dx / (1.414 * 3e8);
  constexpr Real tau{1.7 / (max_frequency - center_frequency)};
  constexpr Real t_0{0.8 * tau};
  constexpr Real cylinder_radius{0.03};
  constexpr SizeType cylinder_radius_in_cells{
      static_cast<SizeType>(cylinder_radius / dx)};

  constexpr Real EPSILON_0 = 8.854187817e-12;
  constexpr Real MU_0 = 4 * 3.14159265 * 1e-7;
  constexpr Real C_0 = 3e8;

  // construct Domain
  auto coord_x = fz::cuda::Tensor<Real, 1>({nx + 1});
  auto coord_y = fz::cuda::Tensor<Real, 1>({ny + 1});
  auto min_x = -dx * (nx / 2.0);
  auto min_y = -dy * (ny / 2.0);
  for (SizeType i = 0; i < nx + 1; ++i) {
    coord_x(i) = min_x + i * dx;
  }
  for (SizeType i = 0; i < ny + 1; ++i) {
    coord_y(i) = min_y + i * dy;
  }
  printf("Domain Size: %lu x %lu\n", nx, ny);
  printf("dx = %f, dy = %f\n", dx, dy);
  printf("min_x = %f, max_x = %f\n", coord_x(0), coord_x(nx));
  printf("min_y = %f, max_y = %f\n", coord_y(0), coord_y(ny));

  // construct Material Space
  auto epsilon_z_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny + 1});
  {
    auto&& epsilon = *epsilon_z_hd.host();
    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        epsilon(i, j) = 1.0 * EPSILON_0;
      }
    }
  }
  auto mu_x_hd = fz::cuda::TensorHD<Real, 2>({nx, ny + 1});
  {
    auto&& mu = *mu_x_hd.host();
    for (SizeType i = 0; i < nx; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        mu(i, j) = 1.0 * MU_0;
      }
    }
  }
  auto mu_y_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny});
  {
    auto&& mu = *mu_y_hd.host();
    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny; ++j) {
        mu(i, j) = 1.0 * MU_0;
      }
    }
  }
  auto sigma_e_z_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny + 1});
  {
    auto&& sigma = *sigma_e_z_hd.host();
    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        sigma(i, j) = 0.0;
      }
    }
  }
  auto sigma_mu_x_hd = fz::cuda::TensorHD<Real, 2>({nx, ny + 1});
  {
    auto&& sigma = *sigma_mu_x_hd.host();
    for (SizeType i = 0; i < nx; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        sigma(i, j) = 0.0;
      }
    }
  }
  auto sigma_mu_y_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny});
  {
    auto&& sigma = *sigma_mu_y_hd.host();
    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny; ++j) {
        sigma(i, j) = 0.0;
      }
    }
  }

  // construct Cylinder
  constexpr Real cylinder_center_x{0.0};
  constexpr Real cylinder_center_y{0.0};
  constexpr Real cylinder_sigma_e = 1e10;
  auto cylinder_shape_func = [cylinder_center_x, cylinder_center_y,
                              cylinder_radius](Real x, Real y,
                                               Real eps = 1e-6) {
    auto dis = (x - cylinder_center_x) * (x - cylinder_center_x) +
               (y - cylinder_center_y) * (y - cylinder_center_y);

    return dis < (cylinder_radius * cylinder_radius + eps);
  };
  // correct the sigma_e in the cylinder
  {
    auto&& sigma = *sigma_e_z_hd.host();
    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        if (cylinder_shape_func(coord_x(i), coord_y(j))) {
          sigma(i, j) = cylinder_sigma_e;
          printf("Cylinder: (%f, %f)\n", coord_x(i), coord_y(j));
        }
      }
    }
  }

  // output material space
  {
    auto epsilon = *epsilon_z_hd.host();
    auto sigma_e = *sigma_e_z_hd.host();
    auto outdir = std::filesystem::path("tmp");
    if (!std::filesystem::exists(outdir)) {
      std::filesystem::create_directory(outdir);
    }
    printf("Output Dir: %s\n", std::filesystem::absolute(outdir).c_str());

    auto sigma_e_out = outdir / "sigma_e_z.dat";
    auto coord_x_out = outdir / "coord_x.dat";
    auto coord_y_out = outdir / "coord_y.dat";
    std::fstream sigma_e_out_file(sigma_e_out, std::ios::out);
    std::fstream coord_x_out_file(coord_x_out, std::ios::out);
    std::fstream coord_y_out_file(coord_y_out, std::ios::out);
    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        sigma_e_out_file << sigma_e(i, j) << " ";
      }
      sigma_e_out_file << "\n";
    }
    for (SizeType i = 0; i < nx + 1; ++i) {
      coord_x_out_file << coord_x(i) << " ";
    }
    for (SizeType i = 0; i < ny + 1; ++i) {
      coord_y_out_file << coord_y(i) << " ";
    }
  }

  // coefficients
  auto ceze_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny + 1});
  auto cezhx_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny + 1});
  auto cezhy_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny + 1});
  {
    auto&& ceze = *ceze_hd.host();
    auto&& cezhx = *cezhx_hd.host();
    auto&& cezhy = *cezhy_hd.host();
    auto&& epsilon = *epsilon_z_hd.host();
    auto&& sigma_e = *sigma_e_z_hd.host();

    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        ceze(i, j) = (2 * epsilon(i, j) - dt * sigma_e(i, j)) /
                     (2 * epsilon(i, j) + dt * sigma_e(i, j));
        cezhx(i, j) = -(2 * dt / dy) / (2 * epsilon(i, j) + dt * sigma_e(i, j));
        cezhy(i, j) = (2 * dt / dx) / (2 * epsilon(i, j) + dt * sigma_e(i, j));
      }
    }
  }
  auto chxh_hd = fz::cuda::TensorHD<Real, 2>({nx, ny + 1});
  auto chxez_hd = fz::cuda::TensorHD<Real, 2>({nx, ny + 1});
  {
    auto&& chxh = *chxh_hd.host();
    auto&& chxez = *chxez_hd.host();
    auto&& mu = *mu_x_hd.host();
    auto&& sigma = *sigma_mu_x_hd.host();

    for (SizeType i = 0; i < nx; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        chxh(i, j) = (2 * mu(i, j) - dt * sigma(i, j)) /
                     (2 * mu(i, j) + dt * sigma(i, j));
        chxez(i, j) = -(2 * dt / dy) / (2 * mu(i, j) + dt * sigma(i, j));
      }
    }
  }
  auto chyh_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny});
  auto chyez_hd = fz::cuda::TensorHD<Real, 2>({nx + 1, ny});
  {
    auto&& chyh = *chyh_hd.host();
    auto&& chyez = *chyez_hd.host();
    auto&& mu = *mu_y_hd.host();
    auto&& sigma = *sigma_mu_y_hd.host();

    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny; ++j) {
        chyh(i, j) = (2 * mu(i, j) - dt * sigma(i, j)) /
                     (2 * mu(i, j) + dt * sigma(i, j));
        chyez(i, j) = (2 * dt / dx) / (2 * mu(i, j) + dt * sigma(i, j));
      }
    }
  }

  // source
  auto source_hd = fz::cuda::TensorHD<Real, 1>({nt});
  {
    auto&& src = *source_hd.host();
    for (SizeType i = 0; i < nt; ++i) {
      src(i) = std::exp(-0.5 * std::pow((t_0 - (i + 0.5) * dt) / tau, 2));
    }
  }

  // TF/SF
  constexpr SizeType tfsf_margin_x = 30;
  constexpr SizeType tfsf_margin_y = 30;
  const auto tfsf_size_x = nx - 2 * tfsf_margin_x;
  const auto tfsf_size_y = ny - 2 * tfsf_margin_y;
  const SizeType aux_size_arr =
      static_cast<SizeType>(std::ceil(
          std::sqrt(tfsf_size_x * tfsf_size_x + tfsf_size_y * tfsf_size_y))) +
      4 + 1;
  
}

int main() {
  // kernel<<<1, 1>>>();
  // cudaDeviceSynchronize();
  FDTD2d();
  return 0;
}
