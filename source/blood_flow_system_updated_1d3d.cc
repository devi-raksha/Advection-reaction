/* --------------------------------------------------------------------------
 */
#include "../include/blood_flow_system_updated_1d3d.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <iomanip>

namespace dealii
{
  // Main class implementation
  template <int dim, int spacedim>
  BloodFlowSystem<dim, spacedim>::BloodFlowSystem()
    : triangulation()
    , dof_handler(triangulation)
    , fe(nullptr)
    , time_step(1.0)
    , time(0.0)
    , n_time_steps(0)
  {
    add_parameter("Finite element degree", fe_degree);
    add_parameter("Problem constants", constants);
    add_parameter("Output filename", output_filename);
    add_parameter("Use direct solver", use_direct_solver);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Number of global refinement", n_global_refinements);
    add_parameter("Time step", time_step);
    add_parameter("Final time", final_time);
    add_parameter("Density (rho)", rho);
    add_parameter("Viscosity coefficient (c)", viscosity_c);
    add_parameter("Reference area", reference_area);
    add_parameter("Elastic modulus", elastic_modulus);
    add_parameter("Reference pressure", reference_pressure);
    add_parameter("Theta (penalty parameter)", theta);
    add_parameter("Eta (stability parameter)", eta);
    add_parameter("Initial condition A expression", initial_A_expression);
    add_parameter("Initial condition U expression", initial_U_expression);
    add_parameter("Pressure boundary expression", pressure_bc_expression);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::initialize_params(const std::string &filename)
  {
    ParameterAcceptor::initialize(filename,
                                  "last_used_parameters.prm",
                                  ParameterHandler::Short);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::setup_system()
  {
    if (!fe)
      {
        // Two-component system: A (area) and U (velocity)
        fe = std::make_unique<FESystem<dim, spacedim>>(
          FE_DGQ<dim, spacedim>(fe_degree), 2);

        // Initialize RHS functions with current parameters
        rhs_A_function = std::make_unique<RHS_A_BloodFlow<spacedim>>();
        rhs_U_function = std::make_unique<RHS_U_BloodFlow<spacedim>>();

        // Update RHS function parameters
        rhs_U_function->set_rho(rho);
        rhs_U_function->set_elastic_modulus(elastic_modulus);
        rhs_U_function->set_viscosity_c(viscosity_c);

        std::string vars;
        if (spacedim == 1)
          vars = "x";
        else if (spacedim == 2)
          vars = "x,y";
        else
          vars = "x,y,z";
        std::map<std::string, double> const_map;
        initial_A.initialize(vars, initial_A_expression, const_map);
        initial_U.initialize(vars, initial_U_expression, const_map);
        pressure_bc.initialize(vars, pressure_bc_expression, const_map);
      }

    dof_handler.distribute_dofs(*fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    solution_old.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
    pressure.reinit(dof_handler.n_dofs());
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_mass_matrix()
  {
    mass_matrix = 0;
    Vector<double> dummy_rhs(dof_handler.n_dofs());

    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    auto cell_worker = [&](const auto                          &cell,
                           BloodFlowScratchData<dim, spacedim> &scratch,
                           BloodFlowCopyData                   &copy) {
      const unsigned int n_dofs = scratch.fe_values.get_fe().n_dofs_per_cell();
      copy.reinit(cell, n_dofs);
      scratch.fe_values.reinit(cell);

      const auto &fe_v = scratch.fe_values;
      const auto &JxW  = fe_v.get_JxW_values();

      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        {
          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  // Mass matrix for area component
                  copy.cell_matrix(i, j) += fe_v[area_extractor].value(i, q) *
                                            fe_v[area_extractor].value(j, q) *
                                            JxW[q];

                  // Mass matrix for velocity component
                  copy.cell_matrix(i, j) +=
                    fe_v[velocity_extractor].value(i, q) *
                    fe_v[velocity_extractor].value(j, q) * JxW[q];
                }
            }
        }
    };

    // Execute assembly
    const QGauss<dim>     quadrature(fe->tensor_degree() + 1);
    const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

    BloodFlowScratchData<dim, spacedim> scratch_data(*fe,
                                                     quadrature,
                                                     quadrature_face);
    BloodFlowCopyData                   copy_data;

    const AffineConstraints<double> constraints;
    auto                            copier = [&](const BloodFlowCopyData &c) {
      constraints.distribute_local_to_global(
        c.cell_matrix, c.cell_rhs, c.local_dof_indices, mass_matrix, dummy_rhs);
    };

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

    system_matrix   = 0;
    right_hand_side = 0;

    // Define extractors
    const FEValuesExtractors::Scalar area_extractor(0);
    const FEValuesExtractors::Scalar velocity_extractor(1);

    // Cell worker - handles volume integral terms
    auto cell_worker = [&](const Iterator                      &cell,
                           BloodFlowScratchData<dim, spacedim> &scratch_data,
                           BloodFlowCopyData                   &copy_data) {
      const unsigned int n_dofs =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);

      const auto &fe_v     = scratch_data.fe_values;
      const auto &JxW      = fe_v.get_JxW_values();
      const auto &q_points = fe_v.get_quadrature_points();

      // Get old solution values using extractors
      std::vector<double> old_area_values(fe_v.n_quadrature_points);
      std::vector<double> old_velocity_values(fe_v.n_quadrature_points);

      fe_v[area_extractor].get_function_values(solution_old, old_area_values);
      fe_v[velocity_extractor].get_function_values(solution_old,
                                                   old_velocity_values);

      for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
        {
          const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                             cell->vertex(1).distance(cell->vertex(0));
          const double A_old = old_area_values[point];
          const double U_old = old_velocity_values[point];

          // Set time for RHS functions
          rhs_A_function->set_time(time);
          rhs_U_function->set_time(time);

          const double rhs_A_value = rhs_A_function->value(q_points[point]);
          const double rhs_U_value = rhs_U_function->value(q_points[point]);

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              for (unsigned int j = 0; j < n_dofs; ++j)
                {
                  // A-U block: semi-implicit advection term
                  const double b_gradA =
                    b_vec * fe_v[area_extractor].gradient(i, point);
                    
                  copy_data.cell_matrix(i, j) -=
                    fe_v[area_extractor].value(j, point) * U_old * b_gradA *
                    JxW[point];

                  // U-U block: nonlinear convection + viscosity
                  const double nonlinear_conv =
                    - 0.5 * U_old *
                    (fe_v[velocity_extractor].value(j, point) *
                     (b_vec * fe_v[velocity_extractor].gradient(i, point)));

                  const double reaction =
                    viscosity_c * fe_v[velocity_extractor].value(i, point) *
                    fe_v[velocity_extractor].value(j, point);

                  copy_data.cell_matrix(i, j) +=
                    (nonlinear_conv + reaction) * JxW[point];
                }

              // Compute pressure using header function
              const double P_old = compute_pressure_value<dim, spacedim>(
                A_old, reference_area, elastic_modulus, reference_pressure);

              // pressure  RHS
              copy_data.cell_rhs(i) +=
                (1.0 / rho) * P_old * b_vec *
                fe_v[velocity_extractor].gradient(i, point) * JxW[point];

              // Right-hand side terms
              copy_data.cell_rhs(i) +=
                rhs_A_value * fe_v[area_extractor].value(i, point) * JxW[point];
              copy_data.cell_rhs(i) +=
                rhs_U_value * fe_v[velocity_extractor].value(i, point) *
                JxW[point];
            }
        }
    };

    // Face worker - handles interior face integrals using header flux functions
    auto face_worker = [&](const Iterator                      &cell,
                           const unsigned int                   f,
                           const unsigned int                   sf,
                           const Iterator                      &ncell,
                           const unsigned int                   nf,
                           const unsigned int                   nsf,
                           BloodFlowScratchData<dim, spacedim> &scratch,
                           BloodFlowCopyData                   &copy) {
      FEInterfaceValues<dim, spacedim> &fe_iv = scratch.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

      const auto &JxW        = fe_iv.get_JxW_values();
      const auto &normals    = fe_iv.get_fe_face_values(0).get_normal_vectors();
      const unsigned int n_q = fe_iv.get_fe_face_values(0).n_quadrature_points;

      // Extract solution values
      std::vector<double> A_L(n_q), U_L(n_q), A_R(n_q), U_R(n_q);
      std::vector<double> A_L_im(n_q), U_L_im(n_q), A_R_im(n_q), U_R_im(n_q);

      fe_iv.get_fe_face_values(0)[area_extractor].get_function_values(
        solution_old, A_L);
      fe_iv.get_fe_face_values(0)[velocity_extractor].get_function_values(
        solution_old, U_L);
      fe_iv.get_fe_face_values(1)[area_extractor].get_function_values(
        solution_old, A_R);
      fe_iv.get_fe_face_values(1)[velocity_extractor].get_function_values(
        solution_old, U_R);

      copy.face_data.emplace_back();
      auto              &face = copy.face_data.back();
      const unsigned int nd   = fe_iv.n_current_interface_dofs();
      face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
      face.cell_matrix.reinit(nd, nd);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const auto   normal = normals[q];
          const double b_dot_n =
            compute_tangent_normal_product<dim, spacedim>(cell, normal);

          // Left and right states
          const double AL = A_L[q], UL = U_L[q];
          const double AR = A_R[q], UR = U_R[q];

          // Compute pressures and wave speeds using header functions
          const double PL = compute_pressure_value<dim, spacedim>(
            AL, reference_area, elastic_modulus, reference_pressure);
          const double PR = compute_pressure_value<dim, spacedim>(
            AR, reference_area, elastic_modulus, reference_pressure);

          const double cL = compute_wave_speed<dim, spacedim>(AL,
                                                              reference_area,
                                                              elastic_modulus,
                                                              rho);
          const double cR = compute_wave_speed<dim, spacedim>(AR,
                                                              reference_area,
                                                              elastic_modulus,
                                                              rho);


          // Penalty parameter
          const double h_face = cell->measure();
          std::cout << "h_face: " << h_face << std::endl;
          // const double sigma =
          //   compute_penalty_parameter<dim, spacedim>(cL, cR, h_face, theta);
          const double sigma = theta*h_face / (2 * time_step);


          for (unsigned int j = 0; j < nd; ++j)
            {
              auto UL_im = fe_iv[velocity_extractor].value(0, j, q);
              auto UR_im = fe_iv[velocity_extractor].value(1, j, q);
              auto AL_im = fe_iv[area_extractor].value(0, j, q);
              auto AR_im = fe_iv[area_extractor].value(1, j, q);

              const double F_au_avg = 0.5 * b_dot_n * (UL * AL_im + UR * AR_im);
              const double F_u_avg = 0.25 * b_dot_n * (UL * UL_im + UR * UR_im);
              const double F_p_avg = 0.5 * (1.0 / rho) * b_dot_n * (PL + PR);
              // check here
              for (unsigned int i = 0; i < nd; ++i)
                {
                  face.cell_matrix(i, j) +=
                    F_au_avg * fe_iv[area_extractor].jump_in_values(i, q) *
                    JxW[q];

                  face.cell_matrix(i, j) +=
                    F_u_avg * fe_iv[velocity_extractor].jump_in_values(i, q) *
                    JxW[q];

                  face.cell_matrix(i, j) +=
                    F_p_avg * fe_iv[velocity_extractor].jump_in_values(j, q) *
                    fe_iv[velocity_extractor].jump_in_values(i, q) * JxW[q];

                  // Penalty/stabilization terms
                  face.cell_matrix(i, j) -=
                    sigma * fe_iv[area_extractor].jump_in_values(j, q) *
                    fe_iv[area_extractor].jump_in_values(i, q) * JxW[q];

                  face.cell_matrix(i, j) -=
                    sigma * fe_iv[velocity_extractor].jump_in_values(j, q) *
                    fe_iv[velocity_extractor].jump_in_values(i, q) * JxW[q];
                }
            }
        }
    };


    // Boundary worker using header exact solution
    ExactSolutionBloodFlow<spacedim> exact_solution;
    exact_solution.set_time(time);

    auto boundary_worker = [&](const Iterator                      &cell,
                               const unsigned int                   face_no,
                               BloodFlowScratchData<dim, spacedim> &scratch,
                               BloodFlowCopyData                   &copy) {
      scratch.fe_interface_values.reinit(cell, face_no);
      const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

      const auto        &JxW     = fe_face.get_JxW_values();
      const auto        &normals = fe_face.get_normal_vectors();
      const unsigned int n_q     = fe_face.n_quadrature_points;

      // Interior trace at t^n
      std::vector<double> A_L(n_q), U_L(n_q);
      fe_face[area_extractor].get_function_values(solution_old, A_L);
      fe_face[velocity_extractor].get_function_values(solution_old, U_L);

      copy.cell_matrix.reinit(fe_face.get_fe().dofs_per_cell,
                              fe_face.get_fe().dofs_per_cell);
      copy.cell_rhs.reinit(fe_face.get_fe().dofs_per_cell);

      // Boundary data at t^n (old time)
      exact_solution.set_time(time - time_step);
      std::vector<Vector<double>> bc_old(n_q, Vector<double>(2));
      exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc_old);

      // Boundary data at t^{n+1} (new time)
      exact_solution.set_time(time);
      std::vector<Vector<double>> bc_new(n_q, Vector<double>(2));
      exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc_new);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double b_dot_n =
            compute_tangent_normal_product<dim, spacedim>(cell, normals[q]);

          // Extract boundary values at t^n
          const double A_ext_old = bc_old[q][0]; // Area component
          const double U_ext_old = bc_old[q][1]; // Velocity component

          // Extract boundary values at t^{n+1}
          const double A_ext_new = bc_new[q][0]; // Area component
          const double U_ext_new = bc_new[q][1]; // Velocity component

          // Compute boundary pressure at t^n
          const double P_ext_old = compute_pressure_value<dim, spacedim>(
            A_ext_old, reference_area, elastic_modulus, reference_pressure);

          // Compute wave speed at boundary
          const double c_ext = compute_wave_speed<dim, spacedim>(
            A_ext_old, reference_area, elastic_modulus, rho);

          const double h     = cell->diameter();
          const double alpha = theta * h / (2 * time_step);

          for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
            {
              // Semi-implicit flux contributions
              const double A_in_new = fe_face[area_extractor].value(j, q);
              const double U_in_new = fe_face[velocity_extractor].value(j, q);

              for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                {
                  // Advective matrix terms (one-sided flux with exterior state)
                  copy.cell_matrix(i, j) +=
                    (U_ext_old * b_dot_n) * A_in_new *
                    fe_face[area_extractor].value(i, q) * JxW[q];

                  copy.cell_matrix(i, j) +=
                    0.5 * (U_ext_old * b_dot_n) * U_in_new *
                    fe_face[velocity_extractor].value(i, q) * JxW[q];

                  // Penalty matrix terms
                  copy.cell_matrix(i, j) -=
                    alpha * fe_face[area_extractor].value(j, q) *
                    fe_face[area_extractor].value(i, q) * JxW[q];

                  copy.cell_matrix(i, j) -=
                    alpha * fe_face[velocity_extractor].value(j, q) *
                    fe_face[velocity_extractor].value(i, q) * JxW[q];
                }

              // RHS forcing from boundary data
              copy.cell_rhs(j) -= (U_ext_old * A_ext_new * b_dot_n) *
                                  fe_face[area_extractor].value(j, q) * JxW[q];

              copy.cell_rhs(j) -= 0.5 * (U_ext_old * U_ext_new * b_dot_n) *
                                  fe_face[velocity_extractor].value(j, q) *
                                  JxW[q];

              copy.cell_rhs(j) -= (1.0 / rho) * P_ext_old * b_dot_n *
                                  fe_face[velocity_extractor].value(j, q) *
                                  JxW[q];
            }
        }
    };


    // Copier lambda
    const AffineConstraints<double> constraints;
    auto                            copier = [&](const BloodFlowCopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             right_hand_side);
      for (auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 system_matrix);
        }
    };

    // Execute mesh loop
    const QGauss<dim>     quadrature(fe->tensor_degree() + 1);
    const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

    BloodFlowScratchData<dim, spacedim> scratch_data(*fe,
                                                     quadrature,
                                                     quadrature_face);
    BloodFlowCopyData                   copy_data;

    auto null_boundary = [](const auto        &cell,
                            const unsigned int face_no,
                            auto              &scratch,
                            auto              &copy) {};

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::solve()
  {
    if (use_direct_solver)
      {
        SparseDirectUMFPACK system_matrix_inverse;
        system_matrix_inverse.initialize(system_matrix);
        system_matrix_inverse.vmult(solution, right_hand_side);
      }
    else
      {
        SolverControl               solver_control(1000, 1e-14);
        SolverGMRES<Vector<double>> solver(solver_control);
        PreconditionSSOR<>          preconditioner;
        const double                omega = 1.4;
        preconditioner.initialize(system_matrix, omega);
        solver.solve(system_matrix, solution, right_hand_side, preconditioner);
        std::cout << "  Solver converged in " << solver_control.last_step()
                  << " iterations." << std::endl;
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::output_results(const unsigned int cycle) const
  {
    const std::string filename =
      output_filename + "-" + std::to_string(cycle) + ".vtu";
    std::cout << "  Writing solution to <" << filename << ">" << std::endl;
    std::ofstream output(filename);

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names(2);
    solution_names[0] = "area";
    solution_names[1] = "velocity";

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        2, DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim, spacedim>::type_dof_data,
                             data_component_interpretation);
    solution_names[0] = "pressure";
    solution_names[1] = "unused";
    data_out.add_data_vector(pressure,
                             solution_names,
                             DataOut<dim, spacedim>::type_dof_data,
                             data_component_interpretation);

    data_out.build_patches();
    data_out.write_vtu(output);

    // Also write the pvd record
    static std::vector<std::pair<double, std::string>> pvd_output_records;
    pvd_output_records.push_back(std::make_pair(time, filename));
    std::ofstream pvd_output(output_filename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, pvd_output_records);
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_pressure()
  {
    pressure = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        std::vector<types::global_dof_index> dof_indices(fe->n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);

        for (unsigned int i = 0; i < fe->n_dofs_per_cell(); ++i)
          {
            const unsigned int component =
              fe->system_to_component_index(i).first;
            if (component == 0) // Area component
              {
                const double area = solution[dof_indices[i]];
                pressure[dof_indices[i]] =
                  compute_pressure_value<dim, spacedim>(area,
                                                        reference_area,
                                                        elastic_modulus,
                                                        reference_pressure);
              }
          }
      }
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::compute_errors(unsigned int k)
  {
    const ComponentSelectFunction<spacedim> area_mask(0, 1.0, 2);
    const ComponentSelectFunction<spacedim> velocity_mask(1, 1.0, 2);

    Vector<float> difference_per_cell(triangulation.n_active_cells());

    // Create exact solution at current time
    ExactSolutionBloodFlow<spacedim> exact_solution;
    exact_solution.set_time(time);

    // Area L2 error
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 3),
                                      VectorTools::L2_norm,
                                      &area_mask);
    const double Area_L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    // Area H1 error
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 3),
                                      VectorTools::H1_seminorm,
                                      &area_mask);
    const double Area_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_seminorm);

    // Velocity L2 error
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 3),
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double Velocity_L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    // Velocity H1 error
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      difference_per_cell,
                                      QGauss<dim>(fe_degree + 3),
                                      VectorTools::H1_seminorm,
                                      &velocity_mask);
    const double Velocity_H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_seminorm);

    // Variables to store previous errors for convergence rate calculation
    static double last_Area_L2_error     = 0;
    static double last_Area_H1_error     = 0;
    static double last_Velocity_L2_error = 0;
    static double last_Velocity_H1_error = 0;

    // Output results with convergence rates
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "=== Error Analysis at Time t = " << time
              << " (Refinement Level " << k + 1 << ") ===" << std::endl;

    std::cout << " Area L2 error:      " << std::setw(12) << Area_L2_error
              << "   Conv_rate: " << std::setw(6)
              << (k == 0 ? 0.0 :
                           std::log(last_Area_L2_error / Area_L2_error) /
                             std::log(2.0))
              << std::endl;

    std::cout << " Area H1 error:      " << std::setw(12) << Area_H1_error
              << "   Conv_rate: " << std::setw(6)
              << (k == 0 ? 0.0 :
                           std::log(last_Area_H1_error / Area_H1_error) /
                             std::log(2.0))
              << std::endl;

    std::cout << " Velocity L2 error:  " << std::setw(12) << Velocity_L2_error
              << "   Conv_rate: " << std::setw(6)
              << (k == 0 ?
                    0.0 :
                    std::log(last_Velocity_L2_error / Velocity_L2_error) /
                      std::log(2.0))
              << std::endl;

    std::cout << " Velocity H1 error:  " << std::setw(12) << Velocity_H1_error
              << "   Conv_rate: " << std::setw(6)
              << (k == 0 ?
                    0.0 :
                    std::log(last_Velocity_H1_error / Velocity_H1_error) /
                      std::log(2.0))
              << std::endl;

    std::cout << " DoFs: " << dof_handler.n_dofs() << "   h ≈ "
              << 1.0 / triangulation.n_active_cells() << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Update previous error values
    last_Area_L2_error     = Area_L2_error;
    last_Area_H1_error     = Area_H1_error;
    last_Velocity_L2_error = Velocity_L2_error;
    last_Velocity_H1_error = Velocity_H1_error;
  }

  template <int dim, int spacedim>
  void
  BloodFlowSystem<dim, spacedim>::run_convergence_study()
  {
    std::cout << "=== CONVERGENCE STUDY for DG" << fe_degree
              << " ===" << std::endl;

    for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
      {
        std::cout << "\n--- Refinement Cycle " << cycle << " ---" << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation);
            triangulation.refine_global(n_global_refinements);
          }
        else
          {
            triangulation.refine_global(1);
          }

        setup_system();

        ExactSolutionBloodFlow<spacedim> exact_solution;
        exact_solution.set_time(0.0);

        // Project initial conditions
        AffineConstraints<double> constraints;
        constraints.close();
        VectorTools::project(dof_handler,
                             constraints,
                             QGauss<dim>(fe_degree + 1),
                             exact_solution,
                             solution);

        solution_old = solution;

        // Run time stepping to final_time
        time = 0.0;
        assemble_mass_matrix();

        n_time_steps =
          static_cast<unsigned int>(std::round(final_time / time_step));
        system_matrix_time.reinit(sparsity_pattern);
        tmp_vector.reinit(dof_handler.n_dofs());

        for (unsigned int step = 1; step <= n_time_steps; ++step)
          {
            time += time_step;
            exact_solution.set_time(time);

            std::cout << "Step " << step << "  t=" << time << std::endl;

            // Assemble system matrix
            assemble_system();

            // Form time-stepping system matrix: M/dt + A
            system_matrix_time.copy_from(mass_matrix);
            system_matrix_time *= (1.0 / time_step);
            system_matrix_time.add(1.0, system_matrix);

            // Form right-hand side: M/dt * u_old + right_hand
            mass_matrix.vmult(tmp_vector, solution_old);
            tmp_vector *= (1.0 / time_step);
            tmp_vector += right_hand_side;

            // Solve time step
            if (use_direct_solver)
              {
                SparseDirectUMFPACK direct;
                direct.initialize(system_matrix_time);
                direct.vmult(solution, tmp_vector);
              }
            else
              {
                SolverControl               solver_control(1000, 1e-14);
                SolverGMRES<Vector<double>> gmres(solver_control);
                PreconditionSSOR<>          preconditioner;
                const double                omega =
                  1.4; // SSOR relaxation parameter (typical range: 1.0-2.0)
                preconditioner.initialize(system_matrix_time, omega);
                gmres.solve(system_matrix_time,
                            solution,
                            tmp_vector,
                            preconditioner);

                std::cout << "  GMRES converged in "
                          << solver_control.last_step() << " iterations."
                          << std::endl;
              }

            std::cout << "norm of vector " << solution.l2_norm() << std::endl;

            solution_old = solution;
            compute_pressure();
            output_results(step);
          }

        // Compute errors at final time
        compute_errors(cycle);
      }
  }

  // Explicit instantiation
  template class BloodFlowSystem<1, 3>;

} // namespace dealii