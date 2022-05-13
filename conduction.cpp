#include <boost/program_options.hpp>
#include <cmath>
#include "conduction.h"
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

using namespace dolfinx;
using namespace std;
namespace po = boost::program_options;

namespace linalg
{
/// Compute vector r = alpha*x + y
/// @param[out] r Result
/// @param[in] alpha
/// @param[in] x
/// @param[in] y
template <typename U>
void axpy(la::Vector<U>& r, U alpha, const la::Vector<U>& x,
          const la::Vector<U>& y)
{
  std::transform(x.array().cbegin(), x.array().cend(), y.array().cbegin(),
                 r.mutable_array().begin(),
                 [alpha](auto x, auto y) { return alpha * x + y; });
}

/// Solve problem A.x = b using the Conjugate Gradient method
/// @tparam U The scalar type
/// @tparam ApplyFunction Type of the function object "action"
/// @param[in, out] x Solution vector, may be set to an initial guess
/// @param[in] b RHS Vector
/// @param[in] action Function that provides the action of the linear operator
/// @param[in] kmax Maximum number of iterations
/// @param[in] rtol Relative tolerances for convergence
/// @return The number if iterations
/// @pre It is required that the ghost values of `x` and `b` have been
/// updated before this function is called
template <typename U, typename ApplyFunction>
int cg(la::Vector<U>& x, const la::Vector<U>& b, ApplyFunction&& action,
       int kmax = 50, double rtol = 1e-8)
{
  // Create working vectors
  la::Vector<U> r(b), y(b);

  // Compute initial residual r0 = b - Ax0
  action(x, y);
  axpy(r, U(-1), y, b);

  // Create p work vector
  la::Vector<U> p(r);

  // Iterations of CG
  double rnorm0 = r.squared_norm();
  const double rtol2 = rtol * rtol;
  double rnorm = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // Compute y = A p
    action(p, y);

    // Compute alpha = r.r/p.y
    const U alpha = rnorm / la::inner_product(p, y);

    // Update x (x <- x + alpha*p)
    axpy(x, alpha, p, x);

    // Update r (r <- r - alpha*y)
    axpy(r, -alpha, y, r);

    // Update residual norm
    // Note: we use U for beta to support float, double, etc. U can be
    // complex, even though the value will always be real
    const double rnorm_new = r.squared_norm();
    const U beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p (p <- beta*p + r)
    axpy(p, beta, p, r);
  }

  return k;
}
} // namespace linalg

int main(int argc, char* argv[])
{
  // std::string grid_info;
  // std::string origin;
  // po::options_description desc("Options");
  // desc.add_options()
  // ("help", "how to pass arguments")
  // ("grid_info", po::value<std::string>(&grid_info), "grid information: Nx-Ny-Nz")
  // ("origin", po::value<std::string>(&origin), "location of origin: x,y,z");
  // po::variables_map vm;
  // po::store(po::parse_command_line(argc, argv, desc), vm);
  // po::notify(vm);
  // if (vm.count("help")){
  //   std::cout << desc << endl;
  //   return 1;
  // }
  
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  {
    using T = PetscScalar;

    MPI_Comm comm = MPI_COMM_WORLD;

    // Create mesh and function space
    io::XDMFFile file_sigma(comm, "mesh/s51-51-51o0_0_0_tetr.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
        comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10}, mesh::CellType::tetrahedron,
        mesh::GhostMode::none));
    // file_sigma.read_mesh(mesh, mesh::GhostMode::none, "Grid");
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_conduction_M, "ui", mesh));

    // Prepare and set Constants for the bilinear form
    auto f = std::make_shared<fem::Constant<T>>(0);
    auto g = std::make_shared<fem::Constant<T>>(0);

    // Define variational forms
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_conduction_L, {V}, {}, {{"f", f}, {"g", g}}, {}));

    // Action of the bilinear form "a" on a function ui
    auto ui = std::make_shared<fem::Function<T>>(V);
    auto M = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_conduction_M, {V}, {{"ui", ui}}, {{}}, {}));

    // Create Dirichlet boundary conditions
    auto x0facet = mesh::locate_entities_boundary(*mesh, 0, [](auto&& x) -> xt::xtensor<bool, 1> {
                                         return xt::isclose(xt::row(x, 0), 0.0);});
    auto x1facet = mesh::locate_entities_boundary(*mesh, 0, [](auto&& x) -> xt::xtensor<bool, 1> {
                                         return xt::isclose(xt::row(x, 0), 1.0);});
    auto u0 = std::make_shared<fem::Function<T>>(V);
    u0->interpolate(
        [](auto&& x) {
          return 1 + xt::square(xt::row(x, 0)) - xt::square(xt::row(x, 0));
        });
    auto u1 = std::make_shared<fem::Function<T>>(V);
    u1->interpolate(
        [](auto&& x) {
          return xt::square(xt::row(x, 0)) - xt::square(xt::row(x, 0));
        });
    std::vector<std::int32_t> x0bdofs = fem::locate_dofs_topological({*V}, 0, x0facet);
    auto x0bc = std::make_shared<const fem::DirichletBC<T>>(u0, x0bdofs);
    std::vector<std::int32_t> x1bdofs = fem::locate_dofs_topological({*V}, 0, x1facet);
    auto x1bc = std::make_shared<const fem::DirichletBC<T>>(u1, x1bdofs);

    // Assemble RHS vector
    la::Vector<T> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());
    fem::assemble_vector(b.mutable_array(), *L);

    // Apply lifting to account for Dirichlet boundary condition
    // b <- b - A * x_bc
    fem::set_bc(ui->x()->mutable_array(), {x0bc, x1bc}, -1.0);
    dolfinx::fem::assemble_vector(b.mutable_array(), *M);

    // Communicate ghost values
    b.scatter_rev(common::IndexMap::Mode::add);

    // Set BC dofs to zero (effectively zeroes columns of A)
    fem::set_bc(b.mutable_array(), {x0bc}, 0.0);
    fem::set_bc(b.mutable_array(), {x1bc}, 0.0);

    b.scatter_fwd();

    // Pack coefficients and constants
    auto coeff = fem::allocate_coefficient_storage(*M);
    const std::vector<T> constants = fem::pack_constants(*M);

    // Create function for computing the action of A on x (y = Ax)
    std::function<void(la::Vector<T>&, la::Vector<T>&)> action
        = [&M, &ui, &x0bc, &x1bc, &coeff, &constants](la::Vector<T>& x, la::Vector<T>& y)
    {
      // Zero y
      y.set(0.0);

      // Update coefficient ui (just copy data from x to ui)
      std::copy(x.array().begin(), x.array().end(),
                ui->x()->mutable_array().begin());

      // Compute action of A on x
      fem::pack_coefficients(*M, coeff);
      fem::assemble_vector(y.mutable_array(), *M, xtl::span<const T>(constants),
                           fem::make_coefficients_span(coeff));

      // Set BC dofs to zero (effectively zeroes rows of A)
      fem::set_bc(y.mutable_array(), {x0bc}, 0.0);
      fem::set_bc(y.mutable_array(), {x1bc}, 0.0);

      // Accumulate ghost values
      y.scatter_rev(common::IndexMap::Mode::add);

      // Update ghost values
      y.scatter_fwd();
    };

    // Compute solution using the conjugate gradient method
    auto u = std::make_shared<fem::Function<T>>(V);
    int num_it = linalg::cg(*u->x(), b, action, 200, 1e-6);

    // Set BC values in the solution vectors
    fem::set_bc(u->x()->mutable_array(), {x0bc}, 1.0);
    fem::set_bc(u->x()->mutable_array(), {x1bc}, 0.0);

    if (dolfinx::MPI::rank(comm) == 0)
    {
      // Save solution in XDMF format
      io::XDMFFile file(MPI_COMM_WORLD, "u.xdmf", "w");
      file.write_mesh(*mesh);
      file.write_function({*u}, 0.0);
      std::cout << "Number of CG iterations " << num_it << std::endl;
    }
  }

  MPI_Finalize();

  return 0;
}
