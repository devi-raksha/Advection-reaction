#ifndef BURGER_EQUATION_H
#define BURGER_EQUATION_H

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>

namespace dealii
{
  // Forward declarations
  template <int dim, int spacedim>
  class BurgerEquation;

  // Exact solution class for Burger's equation
  template <int spacedim>
  class ExactSolutionBurger : public Function<spacedim>
  {
  public:
    ExactSolutionBurger()
      : Function<spacedim>(1) // scalar function
    {}

    virtual double
    value(const Point<spacedim> &p,
          const unsigned int     component = 0) const override;

    virtual Tensor<1, spacedim>
    gradient(const Point<spacedim> &p,
             const unsigned int     component = 0) const override;
  };

  // Right-hand side function for Burger's equation
  template <int spacedim>
  class RHS_Burger : public Function<spacedim>
  {
  public:
    RHS_Burger()
      : Function<spacedim>(1)
    {}

    virtual double
    value(const Point<spacedim> &p,
          const unsigned int     component = 0) const override;
  };

  // Scratch data for assembly
  template <int dim, int spacedim>
  struct BurgerScratchData
  {
    BurgerScratchData(const FiniteElement<dim, spacedim> &fe,
                      const Quadrature<dim>              &quadrature,
                      const Quadrature<dim - 1>          &quadrature_face);

    BurgerScratchData(const BurgerScratchData<dim, spacedim> &scratch_data);

    FEValues<dim, spacedim>          fe_values;
    FEInterfaceValues<dim, spacedim> fe_interface_values;
  };

  // Copy data for assembly
  struct BurgerCopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    struct FaceData
    {
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> joint_dof_indices;
    };
    std::vector<FaceData> face_data;

    template <class Iterator>
    void
    reinit(const Iterator &cell, const unsigned int dofs_per_cell);
  };

  // .........Flux computation functions................
  template <int dim, int spacedim>
  double
  compute_burger_lax_friedrichs_flux(const double u_left,
                                     const double u_right,
                                     const double b_dot_n);

  template <int dim, int spacedim>
  double
  compute_tangent_normal_product_burger(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell,
    const Tensor<1, spacedim>                                      &normal);

  // Main class
  template <int dim, int spacedim>
  class BurgerEquation : public ParameterAcceptor
  {
  public:
    BurgerEquation();

    void
    initialize_params(const std::string &filename);
    void
    run_convergence_study();

  private:
    void
    setup_system();
    void
    assemble_mass_matrix();
    void
    assemble_system();
    void
    solve();
    void
    output_results(const unsigned int cycle) const;
    void
    compute_errors(unsigned int k);

    // Mesh and DOF management
    Triangulation<dim, spacedim>                  triangulation;
    DoFHandler<dim, spacedim>                     dof_handler;
    std::unique_ptr<FiniteElement<dim, spacedim>> fe;

    // Linear algebra objects
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> system_matrix_time;
    Vector<double>       solution;
    Vector<double>       solution_old;
    Vector<double>       right_hand_side;
    Vector<double>       tmp_vector;

    // Parameters
    unsigned int fe_degree            = 1;
    std::string  output_filename      = "solution";
    bool         use_direct_solver    = true;
    unsigned int n_refinement_cycles  = 4;
    unsigned int n_global_refinements = 4;
    double       time_step            = 0.01;
    double       final_time           = 1.0;
    double       theta                = 1.0; // penalty parameter
    double       time                 = 0.0;
    unsigned int n_time_steps         = 0;

    // Function parsers
    FunctionParser<spacedim> initial_condition;
    std::string              initial_expression = "sin(pi*x)";

    // RHS function
    std::unique_ptr<RHS_Burger<spacedim>> rhs_function;
  };

} // namespace dealii

#endif // BURGER_EQUATION_