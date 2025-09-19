/* --------------------------------------------------------------------------
 */

#include "../include/blood-flow-system-1d3d.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/vector_tools.h>

namespace dealii
{

    // Physical parameters and constitutive relations
    template <int dim, int spacedim>
    double
    compute_wave_speed(const double area,
                       const double reference_area,
                       const double elastic_modulus,
                       const double density)
    {
        const double ratio = area / reference_area;
        const double m = 0.5; // tube law exponent
        const double dpda =
            elastic_modulus * m * std::pow(ratio, m - 1.0) / reference_area;
        return std::sqrt(area / density * dpda);
    }

    template <int dim, int spacedim>
    double
    compute_pressure(const double area,
                     const double reference_area,
                     const double elastic_modulus,
                     const double reference_pressure)
    {
        const double ratio = area / reference_area;
        const double m = 0.5; // tube law exponent
        return elastic_modulus * (std::pow(ratio, m) - 1.0) + reference_pressure;
    }

    //   // Advection field b (tangential to curve)
    //   template <int dim, int spacedim>
    //   Tensor<1, spacedim>
    //   advection_field_b(const Point<spacedim> &p)
    //   {
    //     Tensor<1, spacedim> b_field;
    //     // For 1D in 3D, b is tangential to the curve
    //     b_field[0] = 1.0; // tangential direction along x
    //     return b_field;
    //   }

    template <int dim, int spacedim>
    struct BloodFlowScratchData
    {
        BloodFlowScratchData(
            const FiniteElement<dim, spacedim> &fe,
            const Quadrature<dim> &quadrature,
            const Quadrature<dim - 1> &quadrature_face,
            const UpdateFlags update_flags = update_values | update_gradients |
                                             update_quadrature_points |
                                             update_JxW_values,
            const UpdateFlags interface_update_flags = update_values |
                                                       update_gradients |
                                                       update_quadrature_points |
                                                       update_JxW_values |
                                                       update_normal_vectors)
            : fe_values(fe, quadrature, update_flags), fe_interface_values(fe, quadrature_face, interface_update_flags)
        {
        }

        BloodFlowScratchData(
            const BloodFlowScratchData<dim, spacedim> &scratch_data)
            : fe_values(scratch_data.fe_values.get_fe(),
                        scratch_data.fe_values.get_quadrature(),
                        scratch_data.fe_values.get_update_flags()),
              fe_interface_values(scratch_data.fe_interface_values.get_fe(),
                                  scratch_data.fe_interface_values.get_quadrature(),
                                  scratch_data.fe_interface_values.get_update_flags())
        {
        }

        FEValues<dim, spacedim> fe_values;
        FEInterfaceValues<dim, spacedim> fe_interface_values;
    };

    // CopyData structures remain the same
    struct BloodFlowCopyDataFace
    {
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> joint_dof_indices;
    };

    struct BloodFlowCopyData
    {
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        std::vector<BloodFlowCopyDataFace> face_data;

        FullMatrix<double> cell_mass_matrix;
        Vector<double> cell_mass_rhs;

        template <class Iterator>
        void
        reinit(const Iterator &cell, unsigned int dofs_per_cell)
        {
            cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
            cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
            cell_rhs.reinit(dofs_per_cell);
            cell_mass_rhs.reinit(dofs_per_cell);
            local_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);
        }
    };

    // Main class implementation
    template <int dim, int spacedim>
    BloodFlowSystem<dim, spacedim>::BloodFlowSystem()
        : triangulation(), dof_handler(triangulation), fe(nullptr),
          // matrices and vectors default-construct
          time_step(1.0), time(0.0), n_time_steps(0)
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
        add_parameter("Right hand side expression", rhs_expression);
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

            std::string vars;
            if (spacedim == 1)
                vars = "x";
            else if (spacedim == 2)
                vars = "x,y";
            else
                vars = "x,y,z";

            std::map<std::string, double> const_map;
            rhs.initialize(vars, rhs_expression, const_map);
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
    void BloodFlowSystem<dim, spacedim>::assemble_mass_matrix()
    {
        mass_matrix = 0;
        Vector<double> dummy_rhs(dof_handler.n_dofs());

        const FEValuesExtractors::Scalar area_extractor(0);
        const FEValuesExtractors::Scalar velocity_extractor(1);

        auto cell_worker = [&](const auto &cell,
                               BloodFlowScratchData<dim, spacedim> &scratch,
                               BloodFlowCopyData &copy)
        {
            const unsigned int n_dofs = scratch.fe_values.get_fe().n_dofs_per_cell();
            copy.reinit(cell, n_dofs);
            scratch.fe_values.reinit(cell);

            const auto &fe_v = scratch.fe_values;
            const auto &JxW = fe_v.get_JxW_values();

            for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
            {
                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                        // Mass matrix for area component
                        copy.cell_matrix(i, j) +=
                            fe_v[area_extractor].value(i, q) *
                            fe_v[area_extractor].value(j, q) * JxW[q];

                        // Mass matrix for velocity component
                        copy.cell_matrix(i, j) +=
                            fe_v[velocity_extractor].value(i, q) *
                            fe_v[velocity_extractor].value(j, q) * JxW[q];
                    }
                }
            }
        };

        // Execute assembly
        const QGauss<dim> quadrature(fe->tensor_degree() + 1);
        const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

        BloodFlowScratchData<dim, spacedim> scratch_data(*fe, quadrature, quadrature_face);
        BloodFlowCopyData copy_data;

        const AffineConstraints<double> constraints;
        auto copier = [&](const BloodFlowCopyData &c)
        {
            constraints.distribute_local_to_global(c.cell_matrix, c.cell_rhs,
                                                   c.local_dof_indices,
                                                   mass_matrix, dummy_rhs);
        };

        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(),
                              cell_worker, copier, scratch_data, copy_data,
                              MeshWorker::assemble_own_cells);
    }

    template <int dim, int spacedim>
    void
    BloodFlowSystem<dim, spacedim>::assemble_system()
    {
        using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

        system_matrix = 0;
        right_hand_side = 0;

        // Define extractors
        const FEValuesExtractors::Scalar area_extractor(0);
        const FEValuesExtractors::Scalar velocity_extractor(1);

        auto penalty_parameter = [](const double degree,
                                    const double extent1,
                                    const double extent2) -> double
        {
            return 4.0 * degree * (degree + 1.0) * 0.5 * (1.0 / extent1 + 1.0 / extent2);
        };

        // Cell worker which handles volum integral terms

        auto cell_worker = [&](const Iterator &cell,
                               BloodFlowScratchData<dim, spacedim> &scratch_data,
                               BloodFlowCopyData &copy_data)
        {
            const unsigned int n_dofs = scratch_data.fe_values.get_fe().n_dofs_per_cell();
            copy_data.reinit(cell, n_dofs);
            scratch_data.fe_values.reinit(cell);

            const auto &fe_v = scratch_data.fe_values;
            const auto &JxW = fe_v.get_JxW_values();
            const auto &q_points = fe_v.get_quadrature_points();

            // Now get old solution values using extractors
            std::vector<double> old_area_values(fe_v.n_quadrature_points);
            std::vector<double> old_velocity_values(fe_v.n_quadrature_points);

            fe_v[area_extractor].get_function_values(solution_old, old_area_values);
            fe_v[velocity_extractor].get_function_values(solution_old, old_velocity_values);

            for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
            {
                //   const auto b_vec = advection_field_b<dim,
                //   spacedim>(q_points[point]);
                const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                                   cell->vertex(1).distance(cell->vertex(0));
                const double A_old =
                    old_area_values[point]; // Area at previous time
                const double U_old =
                    old_velocity_values[point]; // Velocity at previous time

                // TO obtain RHS value
                const double rhs_value = rhs.value(q_points[point]);

                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                        // Area equation : \partial A/ \partial t + \nabla_{\Gamma}\cdot (AU) =0

                        // A-A block
                        copy_data.cell_matrix(i, j) +=
                            fe_v[area_extractor].value(i, point) *
                            fe_v[area_extractor].value(j, point) * JxW[point]; // Mass term

                        // A-U block  like A_old*b\cdot \nabla_{\Gamma} U

                        const double b_gradU = b_vec * fe_v[velocity_extractor].gradient(j, point); // removes product error
                        copy_data.cell_matrix(i, j) +=
                            fe_v[area_extractor].value(i, point) * A_old *
                            b_gradU * JxW[point];
                        // copy_data.cell_matrix(i, j) +=
                        //     fe_v[area_extractor].value(i, point) * A_old *
                        //     (b_vec * fe_v[velocity_extractor].value(j, point)) * JxW[point];

                        // Velocity equation: \partial U/ \partial t + U\nabla_{\Gamma} U + (1/\rho)\nabla_{\Gamma} P + cU =0
                        // U-U block : nonlinear convection+ viscosity
                        const double nonlinear_conv = U_old *
                                                      (fe_v[velocity_extractor].gradient(i, point)[0] *
                                                       fe_v[velocity_extractor].value(j, point));
                        const double reaction = viscosity_c *
                                                fe_v[velocity_extractor].value(i, point) *
                                                fe_v[velocity_extractor].value(j, point);

                        copy_data.cell_matrix(i, j) += (nonlinear_conv + reaction) * JxW[point];

                        // U-A block : pressure gradient term  -(1/\rho)\nabla_{\Gamma} P = -(1/\rho)(dP/dA)\nabla_{\Gamma} A

                        if (A_old > 1e-12) // Avoid division by zero
                        {
                            const double dpda = elastic_modulus * 0.5 *
                                                std::pow(A_old / reference_area, -0.5) / reference_area;
                            const double pressure_grad = -(1.0 / rho) * dpda *
                                                         (fe_v[velocity_extractor].gradient(i, point)[0] *
                                                          fe_v[area_extractor].value(j, point));
                            copy_data.cell_matrix(i, j) += pressure_grad * JxW[point];
                        }
                    }

                    // Right-hand side

                    copy_data.cell_rhs(i) += rhs_value * (fe_v[area_extractor].value(i, point) + fe_v[velocity_extractor].value(i, point)) * JxW[point];
                }
            }
        };
        // Face worker lambda - handles interior face integrals
        auto face_worker = [&](const Iterator &cell,
                               const unsigned int &f,
                               const unsigned int &sf,
                               const Iterator &ncell,
                               const unsigned int &nf,
                               const unsigned int &nsf,
                               BloodFlowScratchData<dim, spacedim> &scratch_data,
                               BloodFlowCopyData &copy_data)
        {
            FEInterfaceValues<dim, spacedim> &fe_iv =
                scratch_data.fe_interface_values;
            fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

            // Create interface views using extractors - this is the key improvement!
            const auto &area_interface = fe_iv[area_extractor];
            const auto &velocity_interface = fe_iv[velocity_extractor];

            copy_data.face_data.emplace_back();
            BloodFlowCopyDataFace &copy_data_face = copy_data.face_data.back();

            const unsigned int n_dofs = fe_iv.n_current_interface_dofs();
            copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();
            copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

            const auto &JxW = fe_iv.get_JxW_values();
            const auto &normals = fe_iv.get_normal_vectors();
            const auto &q_points = fe_iv.get_quadrature_points();

            // Get old solution values at interface
            std::vector<double> old_area_values(q_points.size());
            std::vector<double> old_velocity_values(q_points.size());

            fe_iv.get_fe_face_values(0)[area_extractor].get_function_values(
                solution_old, old_area_values);
            fe_iv.get_fe_face_values(0)[velocity_extractor].get_function_values(
                solution_old, old_velocity_values);

            const double degree = std::max(1.0, static_cast<double>(fe_iv.get_fe().degree));
            const double extent1 = cell->measure();
            const double extent2 = ncell->measure();
            const double penalty = penalty_parameter(degree, extent1, extent2);

            for (unsigned int q = 0; q < q_points.size(); ++q)
            {
                const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                                   cell->vertex(1).distance(cell->vertex(0));
                const double b_dot_n = b_vec * normals[q];

                const double A_old = old_area_values[q];
                const double U_old = old_velocity_values[q];

                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                        // DG terms for area equation: -A_old * b\cdot n {U} [\phi_A] + penalty[A][\phi_A]
                        copy_data_face.cell_matrix(i, j) +=
                            (-A_old * b_dot_n *
                                 area_interface.average_of_values(j, q) *
                                 area_interface.jump_in_values(i, q) +
                             penalty * theta *
                                 area_interface.jump_in_values(j, q) *
                                 area_interface.jump_in_values(i, q)) *
                            JxW[q];

                        // DG terms for velocity equation: -U_old * b\cdot n {\phi_U} [U] + penalty[U][\phi_U]
                        copy_data_face.cell_matrix(i, j) +=
                            (-U_old * b_dot_n *
                                 velocity_interface.average_of_values(i, q) *
                                 velocity_interface.jump_in_values(j, q) +
                             penalty * theta *
                                 velocity_interface.jump_in_values(j, q) *
                                 velocity_interface.jump_in_values(i, q)) *
                            JxW[q];
                    }
                }
            }
        };

        // Boundary worker lambda - handles boundary face integrals
        auto boundary_worker = [&](const Iterator &cell,
                                   const unsigned int &face_no,
                                   BloodFlowScratchData<dim, spacedim> &scratch_data,
                                   BloodFlowCopyData &copy_data)
        {
            scratch_data.fe_interface_values.reinit(cell, face_no);
            const FEFaceValuesBase<dim, spacedim> &fe_face =
                scratch_data.fe_interface_values.get_fe_face_values(0);

            // Create face views using extractors
            const auto &area_face = fe_face[area_extractor];
            const auto &velocity_face = fe_face[velocity_extractor];

            const auto &q_points = fe_face.get_quadrature_points();
            const auto &JxW = fe_face.get_JxW_values();
            const auto &normals = fe_face.get_normal_vectors();

            const double degree = std::max(1.0, static_cast<double>(fe_face.get_fe().degree));
            const double extent1 = cell->measure(); // Fixed to call measure() as a function
            const double penalty = penalty_parameter(degree, extent1, extent1);

            for (unsigned int point = 0; point < q_points.size(); ++point)
            {
                const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                                   cell->vertex(1).distance(cell->vertex(0));
                const double b_dot_n = b_vec * normals[point];

                if (b_dot_n < 0) // Inflow boundary
                {
                    for (unsigned int i = 0; i < fe_face.get_fe().n_dofs_per_cell(); ++i)
                    {
                        for (unsigned int j = 0; j < fe_face.get_fe().n_dofs_per_cell(); ++j)
                        {
                            // Boundary stabilization
                            copy_data.cell_matrix(i, j) +=
                                theta * penalty * std::abs(b_dot_n) *
                                (area_face.value(i, point) * area_face.value(j, point) +
                                 velocity_face.value(i, point) * velocity_face.value(j, point)) *
                                JxW[point];
                        }

                        // Boundary conditions
                        const double area_bc = reference_area;
                        const double velocity_bc = 0.0;

                        copy_data.cell_rhs(i) +=
                            -theta * penalty * std::abs(b_dot_n) *
                            (area_bc * area_face.value(i, point) +
                             velocity_bc * velocity_face.value(i, point)) *
                            JxW[point];
                    }
                }
            }
        };

        // Copier lambda - assembles local contributions into global system
        const AffineConstraints<double> constraints;
        auto copier = [&](const BloodFlowCopyData &c)
        {
            constraints.distribute_local_to_global(c.cell_matrix, c.cell_rhs,
                                                   c.local_dof_indices,
                                                   system_matrix, right_hand_side);
            for (auto &cdf : c.face_data)
            {
                constraints.distribute_local_to_global(cdf.cell_matrix,
                                                       cdf.joint_dof_indices,
                                                       system_matrix);
            }
        };

        // Execute mesh loop
        const QGauss<dim> quadrature(fe->tensor_degree() + 1);
        const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

        BloodFlowScratchData<dim, spacedim> scratch_data(*fe, quadrature, quadrature_face);
        BloodFlowCopyData copy_data;

        MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(),
                              cell_worker, copier, scratch_data, copy_data,
                              MeshWorker::assemble_own_cells |
                                  MeshWorker::assemble_boundary_faces |
                                  MeshWorker::assemble_own_interior_faces_once,
                              boundary_worker, face_worker);
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::solve()
    {
        if (use_direct_solver)
        {
            SparseDirectUMFPACK system_matrix_inverse;
            system_matrix_inverse.initialize(system_matrix);
            system_matrix_inverse.vmult(solution, right_hand_side);
        }
        else
        {
            SolverControl solver_control(1000, 1e-12);
            SolverCG<Vector<double>> solver(solver_control);
            PreconditionSSOR<> preconditioner;
            preconditioner.initialize(system_matrix, fe->n_dofs_per_cell());
            solver.solve(system_matrix, solution, right_hand_side, preconditioner);
            std::cout << "  Solver converged in " << solver_control.last_step()
                      << " iterations." << std::endl;
        }
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::output_results(const unsigned int cycle) const
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

        // data_out.add_data_vector(pressure, "pressure");
        data_out.build_patches();
        data_out.write_vtu(output);

        // Also write the pvd record
        static std::vector<std::pair<double, std::string>> pvd_output_records;
        pvd_output_records.push_back(std::make_pair(time, filename));
        std::ofstream pvd_output(output_filename + ".pvd");
        DataOutBase::write_pvd_record(pvd_output, pvd_output_records);
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::compute_pressure()
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
                        dealii::compute_pressure<dim, spacedim>(area,
                                                                reference_area,
                                                                elastic_modulus,
                                                                reference_pressure);
                }
            }
        }
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::run()
    {
        // 1. Build initial mesh
        GridGenerator::hyper_cube(triangulation);
        triangulation.refine_global(n_global_refinements);

        // 2. Setup FE system
        setup_system();
        std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        // 3. Assemble mass matrix
        assemble_mass_matrix();

        // 4. Set initial conditions
        const std::string vars = (spacedim == 1 ? "x" : spacedim == 2 ? "x,y"
                                                                      : "x,y,z");

        // Project initial conditions
        AffineConstraints<double> constraints;
        constraints.close();

        // Create combined initial condition function
        // Create combined initial condition function
        std::vector<std::string> expressions(2);
        expressions[0] = initial_A_expression;
        expressions[1] = initial_U_expression;

        // Create a simple vector function for initial conditions
        class InitialCondition : public Function<spacedim>
        {
        public:
            InitialCondition()
                : Function<spacedim>(2)
            {
            } // 2 components

            virtual double
            value(const Point<spacedim> & /*p*/,
                  const unsigned int component = 0) const override

            {
                if (component == 0) // Area
                    return 1.0;     // or parse initial_A_expression
                else                // Velocity
                    return 0.0;     // or parse initial_U_expression
            }
        };

        InitialCondition initial_condition;

        VectorTools::project(dof_handler,
                             constraints,
                             QGauss<dim>(fe->tensor_degree() + 1),
                             initial_condition,
                             solution);

        solution_old = solution;
        compute_pressure();

        // 5. Time loop setup
        n_time_steps =
            static_cast<unsigned int>(std::round(final_time / time_step));
        system_matrix_time.reinit(sparsity_pattern);
        tmp_vector.reinit(dof_handler.n_dofs());

        std::cout << "  Number of time steps: " << n_time_steps
                  << " with time step size dt=" << time_step << std::endl;

        SparseDirectUMFPACK direct;
        if (use_direct_solver)
        {
            // For implicit time stepping: (M/dt + A) u^{n+1} = M/dt u^n + b
            system_matrix_time.copy_from(mass_matrix);
            system_matrix_time *= (1.0 / time_step);
            // Note: system_matrix will be assembled at each time step
        }

        time = 0.0;
        output_results(0);

        // 6. Time stepping
        for (unsigned int step = 1; step <= n_time_steps; ++step)
        {
            time += time_step;
            std::cout << "Step " << step << "  t=" << time << std::endl;

            // Assemble system matrix (depends on solution_old for semi-implicit
            // terms)
            assemble_system();

            // Form time-stepping system matrix: M/dt + A
            system_matrix_time.copy_from(system_matrix);
            system_matrix_time.add(1.0 / time_step, mass_matrix);

            // Form right-hand side: M/dt * u_old + b
            mass_matrix.vmult(tmp_vector, solution_old);
            tmp_vector *= (1.0 / time_step);
            tmp_vector += right_hand_side;

            // Solve time step
            if (use_direct_solver)
            {
                direct.initialize(system_matrix_time);
                direct.vmult(solution, tmp_vector);
            }
            else
            {
                SolverControl solver_control(1000, 1e-12);
                SolverCG<> cg(solver_control);

                PreconditionSSOR<> preconditioner;
                preconditioner.initialize(system_matrix_time, 1.0);
                cg.solve(system_matrix_time, solution, tmp_vector, preconditioner);
            }

            solution_old = solution;
            compute_pressure();
            output_results(step);
        }
    }

    // Explicit instantiation
    template class BloodFlowSystem<1, 3>;

} // namespace dealii