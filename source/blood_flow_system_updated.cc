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
    void
    BloodFlowSystem<dim, spacedim>::assemble_mass_matrix()
    {
        mass_matrix = 0;
        Vector<double> dummy_rhs(dof_handler.n_dofs());

        const QGauss<dim> quadrature(fe->tensor_degree() + 1);
        const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

        BloodFlowScratchData<dim, spacedim> scratch_data(*fe,
                                                         quadrature,
                                                         quadrature_face);
        BloodFlowCopyData copy_data;

        const auto cell_worker = [&](const auto &cell,
                                     BloodFlowScratchData<dim, spacedim> &scratch,
                                     BloodFlowCopyData &copy)
        {
            const unsigned int n_dofs = scratch.fe_values.get_fe().n_dofs_per_cell();
            copy.reinit(cell, n_dofs);
            scratch.fe_values.reinit(cell);
            const auto &fe_v = scratch.fe_values;
            const auto &JxW = fe_v.get_JxW_values();

            for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
                for (unsigned int i = 0; i < n_dofs; ++i)
                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                        const unsigned int component_i =
                            fe->system_to_component_index(i).first;
                        const unsigned int component_j =
                            fe->system_to_component_index(j).first;

                        if (component_i == component_j) // Only diagonal blocks
                            copy.cell_matrix(i, j) +=
                                fe_v.shape_value(i, q) * fe_v.shape_value(j, q) * JxW[q];
                    }
        };

        const AffineConstraints<double> constraints;
        const auto copier = [&](const BloodFlowCopyData &c)
        {
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

        system_matrix = 0;
        right_hand_side = 0;

        const QGauss<dim> quadrature(fe->tensor_degree() + 1);
        const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

        const auto cell_worker = [&](const Iterator &cell,
                                     BloodFlowScratchData<dim, spacedim>
                                         &scratch_data,
                                     BloodFlowCopyData &copy_data)
        {
            const unsigned int n_dofs =
                scratch_data.fe_values.get_fe().n_dofs_per_cell();
            copy_data.reinit(cell, n_dofs);
            scratch_data.fe_values.reinit(cell);

            const auto &q_points = scratch_data.fe_values.get_quadrature_points();
            const auto &fe_v = scratch_data.fe_values;
            const auto &JxW = fe_v.get_JxW_values();

            // Get old solution values at quadrature points
            std::vector<Vector<double>> old_solution_values(fe_v.n_quadrature_points,
                                                            Vector<double>(2));
            fe_v.get_function_values(solution_old, old_solution_values);

            for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
            {
                //   const auto b_vec = advection_field_b<dim,
                //   spacedim>(q_points[point]);
                const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                                   cell->vertex(1).distance(cell->vertex(0));
                const double A_old =
                    old_solution_values[point][0]; // Area at previous time
                const double U_old =
                    old_solution_values[point][1]; // Velocity at previous time

                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    const unsigned int component_i =
                        fe->system_to_component_index(i).first;

                    const auto grad_i = fe_v.shape_grad(i, point);

                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                        const unsigned int component_j =
                            fe->system_to_component_index(j).first;

                        // Equation 1: A_t + b·\nabla (AU) = 0
                        if (component_i == 0 && component_j == 0) // A-A block
                        {
                            // Semi-implicit: b·\nabla (A_old * U) where U is implicit
                            // This becomes: A_old * b·\nabla U (since A_old is
                            // constant)
                            const double adv_term = A_old * (b_vec * grad_i);
                            copy_data.cell_matrix(i, j) +=
                                adv_term * fe_v.shape_value(j, point) * JxW[point];
                        }
                        else if (component_i == 0 && component_j == 1) // A-U block
                        {
                            // Semi-implicit: A_old * b·\nabla U
                            const double adv_term =
                                A_old * (b_vec * fe_v.shape_grad(j, point));
                            copy_data.cell_matrix(i, j) +=
                                adv_term * fe_v.shape_value(i, point) * JxW[point];
                        }

                        // Equation 2: U_t + U\nabla  U + (1/\rho )\nabla  P + cU = 0

                        else if (component_i == 1 && component_j == 1) // U-U block
                        {
                            //   // Compute dpda once for this point
                            //   const double dpda =
                            //     elastic_modulus * 0.5 *
                            //     std::pow(A_old / reference_area, -0.5) /
                            //     reference_area;

                            // Semi-implicit nonlinear convection: U_old ∇U
                            const double nonlinear_conv =
                                U_old * (grad_i[0] * fe_v.shape_value(j, point));

                            // Reaction term: cU
                            const double reaction = viscosity_c *
                                                    fe_v.shape_value(i, point) *
                                                    fe_v.shape_value(j, point);

                            copy_data.cell_matrix(i, j) +=
                                (nonlinear_conv + reaction) * JxW[point];
                        }
                        else if (component_i == 1 &&
                                 component_j == 0) // U-A block (pressure term)
                        {
                            // Use the dpda computed above or recompute it
                            const double dpda =
                                elastic_modulus * 0.5 *
                                std::pow(A_old / reference_area, -0.5) / reference_area;
                            const double pressure_grad =
                                (1.0 / rho) * dpda *
                                (grad_i[0] *
                                 fe_v.shape_value(j, point)); // Use [0] component
                            copy_data.cell_matrix(i, j) += pressure_grad * JxW[point];
                        }
                    }

                    // Right-hand side
                    copy_data.cell_rhs(i) += rhs.value(q_points[point]) *
                                             fe_v.shape_value(i, point) * JxW[point];
                }
            }
        };

        const auto face_worker =
            [&](const Iterator &cell,
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
            const auto &q_points = fe_iv.get_quadrature_points();

            copy_data.face_data.emplace_back();
            BloodFlowCopyDataFace &copy_data_face = copy_data.face_data.back();

            const unsigned int n_dofs = fe_iv.n_current_interface_dofs();
            copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();
            copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

            const std::vector<double> &JxW = fe_iv.get_JxW_values();
            const auto &normals = fe_iv.get_normal_vectors();

            // Get old solution values at interface
            std::vector<Vector<double>> old_solution_face_values(q_points.size(),
                                                                 Vector<double>(2));
            fe_iv.get_fe_face_values(0).get_function_values(
                solution_old, old_solution_face_values);

            for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
            {
                //   const auto b_vec = advection_field_b<dim,
                //   spacedim>(q_points[qpoint]);
                const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                                   cell->vertex(1).distance(cell->vertex(0));
                const double b_dot_n = b_vec * normals[qpoint];
                const double abs_b_dot_n = std::abs(b_dot_n);

                const double A_old = old_solution_face_values[qpoint][0];
                const double U_old = old_solution_face_values[qpoint][1];

                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    const auto localdof_indices =
                        fe_iv.interface_dof_to_dof_indices(i);
                    const auto local_i =
                        localdof_indices[0] != numbers::invalid_unsigned_int ? localdof_indices[0] : localdof_indices[1];

                    const unsigned int component_i =
                        fe->system_to_component_index(local_i).first;

                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {
                        const auto localdof_indices =
                            fe_iv.interface_dof_to_dof_indices(j);
                        const auto local_j =
                            localdof_indices[0] != numbers::invalid_unsigned_int ? localdof_indices[0] : localdof_indices[1];

                        const unsigned int component_j =
                            fe->system_to_component_index(local_j).first;

                        // Interface terms for the coupled system
                        if (component_i == component_j)
                        {
                            // Upwind stabilization for both equations
                            const double stabilization =
                                theta * abs_b_dot_n *
                                fe_iv.jump_in_shape_values(j, qpoint) *
                                fe_iv.jump_in_shape_values(i, qpoint);

                            // Consistency terms (different for each equation)
                            double consistency = 0.0;
                            if (component_i == 0) // Area equation
                            {
                                consistency =
                                    A_old * b_dot_n *
                                    fe_iv.average_of_shape_values(j, qpoint) *
                                    fe_iv.jump_in_shape_values(i, qpoint);
                            }
                            else // Velocity equation
                            {
                                consistency =
                                    U_old * b_dot_n *
                                    fe_iv.average_of_shape_values(j, qpoint) *
                                    fe_iv.jump_in_shape_values(i, qpoint);
                            }

                            copy_data_face.cell_matrix(i, j) +=
                                (consistency + stabilization) * JxW[qpoint];
                        }
                    }
                }
            }
        };

        const auto boundary_worker =
            [&](const Iterator &cell,
                const unsigned int &face_no,
                BloodFlowScratchData<dim, spacedim> &scratch_data,
                BloodFlowCopyData &copy_data)
        {
            scratch_data.fe_interface_values.reinit(cell, face_no);
            const FEFaceValuesBase<dim, spacedim> &fe_face =
                scratch_data.fe_interface_values.get_fe_face_values(0);

            const auto &q_points = fe_face.get_quadrature_points();
            const unsigned int n_facet_dofs = fe_face.get_fe().n_dofs_per_cell();
            const std::vector<double> &JxW = fe_face.get_JxW_values();
            const auto &normals = fe_face.get_normal_vectors();

            for (unsigned int point = 0; point < q_points.size(); ++point)
            {
                const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                                   cell->vertex(1).distance(cell->vertex(0));
                // advection_field_b<dim, spacedim>(q_points[point]);
                const double b_dot_n = b_vec * normals[point];
                const double abs_b_dot_n = std::abs(b_dot_n);

                if (b_dot_n < 0) // Inflow boundary
                {
                    for (unsigned int i = 0; i < n_facet_dofs; ++i)
                    {
                        const unsigned int component_i =
                            fe->system_to_component_index(i).first;

                        for (unsigned int j = 0; j < n_facet_dofs; ++j)
                        {
                            const unsigned int component_j =
                                fe->system_to_component_index(j).first;

                            if (component_i == component_j)
                            {
                                copy_data.cell_matrix(i, j) +=
                                    theta * abs_b_dot_n *
                                    fe_face.shape_value(i, point) *
                                    fe_face.shape_value(j, point) * JxW[point];
                            }
                        }

                        // Apply boundary conditions
                        if (component_i == 0) // Area boundary condition
                        {
                            const double bc_value = reference_area; // Default
                            copy_data.cell_rhs(i) +=
                                -theta * abs_b_dot_n * bc_value *
                                fe_face.shape_value(i, point) * JxW[point];
                        }
                        else // Velocity boundary condition
                        {
                            const double bc_value = 0.0; // Default
                            copy_data.cell_rhs(i) +=
                                -theta * abs_b_dot_n * bc_value *
                                fe_face.shape_value(i, point) * JxW[point];
                        }
                    }
                }
            }
        };

        const AffineConstraints<double> constraints;
        const auto copier = [&](const BloodFlowCopyData &c)
        {
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

        BloodFlowScratchData<dim, spacedim> scratch_data(*fe,
                                                         quadrature,
                                                         quadrature_face);
        BloodFlowCopyData copy_data;

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
                        dealii::compute_pressure<dim, spacedim>(area,
                                                                reference_area,
                                                                elastic_modulus,
                                                                reference_pressure);
                }
            }
        }
    }

    template <int dim, int spacedim>
    void
    BloodFlowSystem<dim, spacedim>::run()
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