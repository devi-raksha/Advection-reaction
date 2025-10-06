/* --------------------------------------------------------------------------
 */
#include "../include/blood_flow_system_updated_1d3d.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/vector_tools.h>

#include <algorithm>

#include <iomanip>
#include <deal.II/numerics/error_estimator.h>

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
    compute_pressure_value(const double area,
                           const double reference_area,
                           const double elastic_modulus,
                           const double reference_pressure)
    {
        const double ratio = area / reference_area;
        const double m = 0.5; // tube law exponent
        return elastic_modulus * (std::pow(ratio, m) - 1.0) + reference_pressure;
    }

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
        add_parameter("Eta (stability parameter)", eta);
        // add_parameter("Right hand side A expression", rhs_A_expression);
        // add_parameter("Right hand side U expression", rhs_U_expression);
        add_parameter("Initial condition A expression", initial_A_expression);
        add_parameter("Initial condition U expression", initial_U_expression);
        add_parameter("Pressure boundary expression", pressure_bc_expression);
    }

    // rhs A and rhs U functions based on manufactured solution

    template <int dim, int spacedim>
    void
    BloodFlowSystem<dim, spacedim>::initialize_params(const std::string &filename)
    {
        ParameterAcceptor::initialize(filename,
                                      "last_used_parameters.prm",
                                      ParameterHandler::Short);
    }

    //...........................................................................................
    // Exact solution, its gradient, and vectorized versions with right hand side functions
    //...........................................................................................

    template <int dim, int spacedim>
    double
    BloodFlowSystem<dim, spacedim>::ExactSolution::value(
        const Point<spacedim> &p, const unsigned int component) const
    {
        // Parameters matching
        const double r0 = 9.99e-3;
        const double a0 = numbers::PI * r0 * r0;
        const double L = 1.0;
        const double T0 = 1.0;
        const double atilde = 0.1 * a0;
        const double qtilde = 0.0;

        const double x = p[0];
        const double t = this->get_time();

        if (component == 0) // area A
            return a0 + atilde * std::sin(2.0 * numbers::PI * x / L) *
                            std::cos(2.0 * numbers::PI * t / T0);
        else // velocity U
            return qtilde - (atilde * L / T0) *
                                std::cos(2.0 * numbers::PI * x / L) *
                                std::sin(2.0 * numbers::PI * t / T0);
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::ExactSolution::vector_value(
        const Point<spacedim> &p, Vector<double> &values) const
    {
        Assert(values.size() == 2, ExcDimensionMismatch(values.size(), 2));
        values[0] = value(p, 0);
        values[1] = value(p, 1);
    }

    template <int dim, int spacedim>
    Tensor<1, spacedim>
    BloodFlowSystem<dim, spacedim>::ExactSolution::gradient(
        const Point<spacedim> &p, const unsigned int component) const
    {
        // Parameters matching
        const double r0 = 9.99e-3;
        const double a0 = numbers::PI * r0 * r0;
        const double L = 1.0;
        const double T0 = 1.0;
        const double atilde = 0.1 * a0;
        const double qtilde = 0.0;

        const double x = p[0];
        const double t = this->get_time();

        Tensor<1, spacedim> grad;

        if (component == 0) // area A gradient
        {
            // dA/dx = atilde * (2\pi/L) * cos(2\pi x/L) * cos(2\pi t/T0)
            grad[0] = atilde * (2.0 * numbers::PI / L) *
                      std::cos(2.0 * numbers::PI * x / L) *
                      std::cos(2.0 * numbers::PI * t / T0);
        }
        else if (component == 1) // velocity U gradient
        {
            // dU/dx = (atilde * L / T0) * (2\pi /L) * sin(2\pi x/L) * sin(2\pi t/T0)
            grad[0] = (atilde * L / T0) * (2.0 * numbers::PI / L) *
                      std::sin(2.0 * numbers::PI * x / L) *
                      std::sin(2.0 * numbers::PI * t / T0);
        }

        return grad;
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::ExactSolution::vector_gradient(
        const Point<spacedim> &p,
        std::vector<Tensor<1, spacedim>> &gradients) const
    {
        Assert(gradients.size() == 2, ExcDimensionMismatch(gradients.size(), 2));

        gradients[0] = gradient(p, 0); // Area gradient
        gradients[1] = gradient(p, 1); // Velocity gradient
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::ExactSolution::vector_value_list(
        const std::vector<Point<spacedim>> &points,
        std::vector<Vector<double>> &value_list) const
    {
        const unsigned int n = points.size();
        Assert(value_list.size() == n, ExcDimensionMismatch(value_list.size(), n));
        for (unsigned int i = 0; i < n; ++i)
            vector_value(points[i], value_list[i]);
    }

    // RHS A‐forcing term f_a(x,t)
    template <int spacedim>
    class RHS_A : public Function<spacedim>
    {
    public:
        RHS_A()
            : Function<spacedim>(1)
        {
        }

        virtual double
        value(const Point<spacedim> &p,
              const unsigned int /*component*/ = 0) const override
        {
            const double r0 = 9.99e-3;
            const double a0 = numbers::PI * r0 * r0;
            const double L = 1.0;
            const double T0 = 1.0;
            const double atilde = 0.1 * a0;
            const double qtilde = 0.0;
            const double x = p[0];
            const double t = this->get_time();

            // forcing term from manufactured solution
            return std::sin(2.0 * numbers::PI * x / L) * std::sin(2.0 * numbers::PI * t / T0) *
                       (-2.0 * numbers::PI / T0 * atilde + (a0 + atilde * std::sin(2.0 * numbers::PI * x / L) * std::cos(2.0 * numbers::PI * t / T0)) * 2.0 * numbers::PI / T0 * atilde) +
                   atilde * std::cos(2.0 * numbers::PI * x / L) * std::cos(2.0 * numbers::PI * t / T0) * (2.0 * numbers::PI / L) * (qtilde - (atilde * L / T0) * std::cos(2.0 * numbers::PI * x / L) * std::sin(2.0 * numbers::PI * t / T0));
        }
    };

    // U‐forcing term f_u(x,t)
    template <int spacedim>
    class RHS_U : public Function<spacedim>
    {
    public:
        RHS_U()
            : Function<spacedim>(1)
        {
        }

        virtual double
        value(const Point<spacedim> &p,
              const unsigned int /*component*/ = 0) const override
        {
            const double r0 = 9.99e-3;
            const double a0 = numbers::PI * r0 * r0;
            const double L = 1.0;
            const double T0 = 1.0;
            const double atilde = 0.1 * a0;
            const double rho = 1.06;
            const double elastic_modulus = 1.0;
            const double viscosity_c = 1.0;
            const double m = 0.5;
            const double x = p[0];
            const double t = this->get_time();
            const double A = a0 + atilde * std::sin(2.0 * numbers::PI * x / L) * std::cos(2.0 * numbers::PI * t / T0);

            return std::cos(2.0 * numbers::PI * x / L) * std::cos(2.0 * numbers::PI * t / T0) * (-L * L / (T0 * T0) + elastic_modulus / (rho * std::pow(a0, m)) * std::pow(A, m - 1)) + (atilde - (atilde * L / T0) * std::cos(2.0 * numbers::PI * x / L) * std::sin(2.0 * numbers::PI * t / T0)) * ((2.0 * numbers::PI / T0) * atilde * std::sin(2.0 * numbers::PI * x / L) * std::sin(2.0 * numbers::PI * t / T0) + viscosity_c);
        }
    };

    //.............................................................................................
    // end of RHS functions
    //.............................................................................................

    template <int dim, int spacedim>
    void
    BloodFlowSystem<dim, spacedim>::setup_system()
    {
        if (!fe)
        {
            // Two-component system: A (area) and U (velocity)
            fe = std::make_unique<FESystem<dim, spacedim>>(
                FE_DGQ<dim, spacedim>(fe_degree), 2);

            rhs_A_function = std::make_unique<RHS_A<spacedim>>();
            rhs_U_function = std::make_unique<RHS_U<spacedim>>();

            std::string vars;
            if (spacedim == 1)
                vars = "x";
            else if (spacedim == 2)
                vars = "x,y";
            else
                vars = "x,y,z";
            std::map<std::string, double> const_map;
            // rhs_A.initialize(vars, rhs_A_expression, const_map);
            // rhs_U.initialize(vars, rhs_U_expression, const_map);
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

        // Cell worker which handles volume integral terms

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

            /*
             * Volume term assembly based on: equation :17 of the following paper
             * -------------------------------------------------------------------------
             * Computational modelling of 1D blood flow with variable mechanical properties
             * and its application to the simulation of wave propagation in the human
             * arterial system
             *
             * S. J. Sherwin, L. Formaggia, J. Peiró, V. Franke
             * ------------------------------------
             */

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

                // To obtain RHS values
                rhs_A_function->set_time(time);
                rhs_U_function->set_time(time);

                const double rhs_A_value = rhs_A_function->value(q_points[point]);
                const double rhs_U_value = rhs_U_function->value(q_points[point]);

                const unsigned int n_q_points = fe_v.n_quadrature_points;

                std::vector<Tensor<1, spacedim>> grad_U_old_values(n_q_points);
                fe_v[velocity_extractor].get_function_gradients(solution_old, grad_U_old_values);

                // std::vector<Tensor<1, spacedim>> grad_A_old_values(n_q_points);
                // fe_v[velocity_extractor].get_function_gradients(solution_old, grad_A_old_values);

                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                    for (unsigned int j = 0; j < n_dofs; ++j)
                    {

                        // A-U block : advection (-b.\nabla \phi_A , A_old U)
                        const double b_gradU = b_vec * fe_v[velocity_extractor].gradient(j, point); // removes product error
                        copy_data.cell_matrix(i, j) -=
                            fe_v[area_extractor].value(i, point) * A_old *
                            b_gradU * JxW[point];

                        // add h ^ { eta }(\nabla u ^ n, \nabla \phi_U) :  for stablizating oscillation based on step-33
                        const double h = std::sqrt(1.0 / triangulation.n_active_cells());

                        copy_data.cell_matrix(i, j) += std::pow(h, eta) * fe_v[velocity_extractor].gradient(i, point) * fe_v[velocity_extractor].gradient(j, point) * JxW[point];
                        copy_data.cell_matrix(i, j) += std::pow(h, eta) * fe_v[area_extractor].gradient(i, point) * fe_v[area_extractor].gradient(j, point) * JxW[point];

                        // Velocity equation U - U block : nonlinear convection + viscosity term
                        // \nabla_{\Gamma}(U^2/2) \phi_A = - (0.5* U_old* U, (b.\nabla \phi_U))

                        const double nonlinear_conv = -0.5 * U_old *
                                                      (fe_v[velocity_extractor].value(i, point) *
                                                       (b_vec * fe_v[velocity_extractor].gradient(j, point)));

                        // reaction term  (viscosity_c* U, \phi_U)
                        const double reaction = viscosity_c *
                                                fe_v[velocity_extractor].value(i, point) *
                                                fe_v[velocity_extractor].value(j, point);

                        copy_data.cell_matrix(i, j) += (nonlinear_conv + reaction) * JxW[point];

                        // U-A block : pressure gradient term (1/rho) \nabla_{\Gamma} P(A) \phi_U = - (1/rho) * (dP/dA(at A_old) * b.\nabla \phi_U, A)

                        if (A_old > 1e-12) // Avoid division by zero

                        {
                            const double dpda = elastic_modulus * 0.5 *
                                                std::pow(A_old / reference_area, -0.5) / reference_area;

                            const double pressure_grad = -(1.0 / rho) * dpda *
                                                         (b_vec * fe_v[velocity_extractor].gradient(j, point) *
                                                          fe_v[area_extractor].value(i, point));
                            copy_data.cell_matrix(i, j) += pressure_grad * JxW[point];
                        }
                    }

                    // Right-hand side

                    copy_data.cell_rhs(i) += rhs_A_value * fe_v[area_extractor].value(i, point) * JxW[point];
                    copy_data.cell_rhs(i) += rhs_U_value * fe_v[velocity_extractor].value(i, point) * JxW[point];
                }
            }
        };

        // Face worker lambda ‒ handles interior face integrals
        //   Based on rusanov flux / local Lax-Friedrichs flux f(a, b) = 0.5*(f(a) + f(b)) - 0.5*alpha*(b-a)
        //  where alpha = max(|u-c|, |u+c|) , c = wave speed

        auto face_worker = [&](const Iterator &cell,
                               const unsigned int f,
                               const unsigned int sf,
                               const Iterator &ncell,
                               const unsigned int nf,
                               const unsigned int nsf,
                               BloodFlowScratchData<dim, spacedim> &scratch,
                               BloodFlowCopyData &copy)
        {
            FEInterfaceValues<dim, spacedim> &fe_iv = scratch.fe_interface_values;
            fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

            const auto &JxW = fe_iv.get_JxW_values();
            const auto &normals = fe_iv.get_fe_face_values(0).get_normal_vectors();

            // Number of quadrature points on this face
            const unsigned int n_q = fe_iv.get_fe_face_values(0).n_quadrature_points;

            // Read old solution on both sides
            std::vector<double> A_L(n_q), U_L(n_q), A_R(n_q), U_R(n_q);
            fe_iv.get_fe_face_values(0)[area_extractor]
                .get_function_values(solution_old, A_L);
            fe_iv.get_fe_face_values(0)[velocity_extractor]
                .get_function_values(solution_old, U_L);
            fe_iv.get_fe_face_values(1)[area_extractor]
                .get_function_values(solution_old, A_R);
            fe_iv.get_fe_face_values(1)[velocity_extractor]
                .get_function_values(solution_old, U_R);

            copy.face_data.emplace_back();
            auto &face = copy.face_data.back();
            const unsigned int nd = fe_iv.n_current_interface_dofs();
            face.joint_dof_indices = fe_iv.get_interface_dof_indices();
            face.cell_matrix.reinit(nd, nd);

            for (unsigned int q = 0; q < n_q; ++q)
            {
                // Compute tangent-aligned b·n
                const auto normal = normals[q];
                const auto b_vec = (cell->vertex(1) - cell->vertex(0)) /
                                   cell->vertex(1).distance(cell->vertex(0));
                const double b_dot_n = b_vec * normal;

                // Left and right states
                const double AL = A_L[q], UL = U_L[q];
                const double AR = A_R[q], UR = U_R[q];

                // Pressures and wave speeds
                const double PL = compute_pressure_value<dim, spacedim>(
                    AL, reference_area, elastic_modulus, reference_pressure);
                const double PR = compute_pressure_value<dim, spacedim>(
                    AR, reference_area, elastic_modulus, reference_pressure);
                const double cL = compute_wave_speed<dim, spacedim>(
                    AL, reference_area, elastic_modulus, rho);
                const double cR = compute_wave_speed<dim, spacedim>(
                    AR, reference_area, elastic_modulus, rho);

                // Rusanov penalty
                const double s1 = std::abs(UL - cL);
                const double s2 = std::abs(UL + cL);
                const double s3 = std::abs(UR - cR);
                const double s4 = std::abs(UR + cR);
                const double alpha = 0.5 * std::max({s1, s2, s3, s4});

                // Physical fluxes
                const double FA_L = AL * UL * b_dot_n;
                const double FA_R = AR * UR * b_dot_n;
                const double FU_L = (0.5 * UL * UL + PL / rho) * b_dot_n;
                const double FU_R = (0.5 * UR * UR + PR / rho) * b_dot_n;

                // Local Lax-Friedrichs Fluxes (Rusanov Fluxes)
                const double flux_A = (FA_L + FA_R) - alpha * (AR - AL);
                const double flux_U = (FU_L + FU_R) - alpha * (UR - UL);

                const double h_face = cell->face(f)->measure();
                const double sigma = theta * std::max(std::abs(cL), std::abs(cR)) / h_face;

                for (unsigned int i = 0; i < nd; ++i)
                {
                    for (unsigned int j = 0; j < nd; ++j)
                    {
                        face.cell_matrix(i, j) +=
                            flux_A * fe_iv[area_extractor].jump_in_values(j, q) * JxW[q];

                        face.cell_matrix(i, j) +=
                            flux_U * fe_iv[velocity_extractor].jump_in_values(j, q) * JxW[q];

                        face.cell_matrix(i, j) +=
                            sigma * fe_iv[velocity_extractor].jump_in_values(i, q) * fe_iv[velocity_extractor].jump_in_values(j, q) * JxW[q];

                        // face.cell_matrix(i, j) +=
                        //     sigma * fe_iv[area_extractor].jump_in_values(i, q) * fe_iv[area_extractor].jump_in_values(j, q) * JxW[q];
                    }
                }
            }
        };

        // Boundary worker lambda ‒ handles boundary face integrals
        typename BloodFlowSystem<dim, spacedim>::ExactSolution exact_solution;
        exact_solution.set_time(time);

        auto boundary_worker = [&](const Iterator &cell,
                                   const unsigned int face_no,
                                   BloodFlowScratchData<dim, spacedim> &scratch,
                                   BloodFlowCopyData &copy)
        {
            scratch.fe_interface_values.reinit(cell, face_no);
            const auto &fe_face = scratch.fe_interface_values.get_fe_face_values(0);

            const auto &JxW = fe_face.get_JxW_values();
            const auto &normals = fe_face.get_normal_vectors();

            const unsigned int n_q = fe_face.n_quadrature_points;

            // Interior trace
            std::vector<double> A_L(n_q), U_L(n_q);
            fe_face[area_extractor]
                .get_function_values(solution_old, A_L);
            fe_face[velocity_extractor]
                .get_function_values(solution_old, U_L);

            // Dirichlet data from exact_solution
            std::vector<Vector<double>> bc(n_q, Vector<double>(2));
            exact_solution.vector_value_list(fe_face.get_quadrature_points(), bc);

            // Reinit local rhs
            copy.cell_matrix.reinit(fe_face.get_fe().dofs_per_cell,
                                    fe_face.get_fe().dofs_per_cell);
            copy.cell_rhs.reinit(fe_face.get_fe().dofs_per_cell);

            for (unsigned int q = 0; q < n_q; ++q)
            {
                const double b_dot_n = normals[q] *
                                       ((cell->vertex(1) - cell->vertex(0)) /
                                        cell->vertex(1).distance(cell->vertex(0)));

                // Interior and prescribed states
                const double AL = A_L[q], UL = U_L[q];
                const double AR = bc[q](0), UR = bc[q](1);

                // Pressure and wave speeds
                const double PL = compute_pressure_value<dim, spacedim>(
                    AL, reference_area, elastic_modulus, reference_pressure);
                const double PR = compute_pressure_value<dim, spacedim>(
                    AR, reference_area, elastic_modulus, reference_pressure);
                const double cL = compute_wave_speed<dim, spacedim>(
                    AL, reference_area, elastic_modulus, rho);
                const double cR = compute_wave_speed<dim, spacedim>(
                    AR, reference_area, elastic_modulus, rho);

                // Rusanov penalty
                const double s1 = std::abs(UL - cL);
                const double s2 = std::abs(UL + cL);
                const double s3 = std::abs(UR - cR);
                const double s4 = std::abs(UR + cR);
                const double alpha = 0.5 * std::max({s1, s2, s3, s4});

                // Physical fluxes
                const double FA_L = AL * UL * b_dot_n;
                const double FA_R = AR * UR * b_dot_n;
                const double FU_L = (0.5 * UL * UL + PL / rho) * b_dot_n;
                const double FU_R = (0.5 * UR * UR + PR / rho) * b_dot_n;

                // Rusanov flux
                const double flux_A = (FA_L + FA_R) - alpha * (AR - AL);
                const double flux_U = (FU_L + FU_R) - alpha * (UR - UL);

                // const double h_face = cell->face(face_no)->measure();
                // const double sigma = theta * std::max(cL, cR) / h_face;

                // charactieristc values

                const double lambda1 = UL * b_dot_n + cL;
                const double lambda2 = UL * b_dot_n - cL;

                if (lambda1 > 0)
                {
                    for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                    {
                        copy.cell_rhs(i) += -(FA_R * fe_face[area_extractor].value(i, q) + FU_R * fe_face[velocity_extractor].value(i, q)) * JxW[q];
                    }
                }
                else if (lambda2 < 0)
                {
                    for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                    {
                        // for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
                        // {
                        copy.cell_rhs(i) +=
                            FA_L * fe_face[area_extractor].value(i, q) * JxW[q] + FU_L * fe_face[velocity_extractor].value(i, q) * JxW[q];
                        // }
                    }
                }
                else
                {
                    for (unsigned int i = 0; i < fe_face.get_fe().dofs_per_cell; ++i)
                    {
                        // for (unsigned int j = 0; j < fe_face.get_fe().dofs_per_cell; ++j)
                        // {
                        copy.cell_rhs(i) +=
                            flux_A * fe_face[area_extractor].value(i, q) * JxW[q] + flux_U * fe_face[velocity_extractor].value(i, q) * JxW[q];
                        // }
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
                        compute_pressure_value<dim, spacedim>(area,
                                                              reference_area,
                                                              elastic_modulus,
                                                              reference_pressure);
                }
            }
        }
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::compute_errors(unsigned int k)
    {
        // Component selectors for area (0) and velocity (1)
        const ComponentSelectFunction<spacedim> area_mask(0, 1.0, 2);
        const ComponentSelectFunction<spacedim> velocity_mask(1, 1.0, 2);

        Vector<float> difference_per_cell(triangulation.n_active_cells());

        // Create exact solution at current time
        typename BloodFlowSystem<dim, spacedim>::ExactSolution exact_solution;
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

        // variables to store previous errors for convergence rate calculation
        static double last_Area_L2_error = 0;
        static double last_Area_H1_error = 0;
        static double last_Velocity_L2_error = 0;
        static double last_Velocity_H1_error = 0;

        // Output results with convergence rates
        std::cout << std::scientific << std::setprecision(3);
        std::cout << "=== Error Analysis at Time t = " << time
                  << " (Refinement Level " << k + 1 << ") ===" << std::endl;

        std::cout << " Area L2 error:      " << std::setw(12) << Area_L2_error
                  << "   Conv_rate: " << std::setw(6)
                  << (k == 0 ? 0.0 : std::log(last_Area_L2_error / Area_L2_error) / std::log(2.0))
                  << std::endl;

        std::cout << " Area H1 error:      " << std::setw(12) << Area_H1_error
                  << "   Conv_rate: " << std::setw(6)
                  << (k == 0 ? 0.0 : std::log(last_Area_H1_error / Area_H1_error) / std::log(2.0))
                  << std::endl;

        std::cout << " Velocity L2 error:  " << std::setw(12) << Velocity_L2_error
                  << "   Conv_rate: " << std::setw(6)
                  << (k == 0 ? 0.0 : std::log(last_Velocity_L2_error / Velocity_L2_error) / std::log(2.0))
                  << std::endl;

        std::cout << " Velocity H1 error:  " << std::setw(12) << Velocity_H1_error
                  << "   Conv_rate: " << std::setw(6)
                  << (k == 0 ? 0.0 : std::log(last_Velocity_H1_error / Velocity_H1_error) / std::log(2.0))
                  << std::endl;

        std::cout << " DoFs: " << dof_handler.n_dofs()
                  << "   h ≈ " << 1.0 / triangulation.n_active_cells() << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Update previous error values
        last_Area_L2_error = Area_L2_error;
        last_Area_H1_error = Area_H1_error;
        last_Velocity_L2_error = Velocity_L2_error;
        last_Velocity_H1_error = Velocity_H1_error;
    }

    template <int dim, int spacedim>
    void BloodFlowSystem<dim, spacedim>::run_convergence_study()
    {
        std::cout << "=== CONVERGENCE STUDY for DG" << fe_degree << " ===" << std::endl;

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

            typename BloodFlowSystem<dim, spacedim>::ExactSolution exact_solution;
            exact_solution.set_time(0.0);

            // Project initial conditions
            AffineConstraints<double> constraints;
            constraints.close();
            VectorTools::project(dof_handler, constraints,
                                 QGauss<dim>(fe_degree + 1),
                                 exact_solution, solution);

            solution_old = solution;

            // Run time stepping to final_time
            time = 0.0;
            assemble_mass_matrix();

            // Time loop

            n_time_steps =
                static_cast<unsigned int>(std::round(final_time / time_step));
            system_matrix_time.reinit(sparsity_pattern);
            tmp_vector.reinit(dof_handler.n_dofs());
            for (unsigned int step = 1; step <= n_time_steps; ++step)
            {
                time += time_step;
                // update bc time in functions
                exact_solution.set_time(time);

                std::cout << "Step " << step << "  t=" << time << std::endl;

                // Assemble system matrix (depends on solution_old for semi-implicit
                // terms)
                assemble_system();

                // Form time-stepping system matrix: M/dt + A
                system_matrix_time.copy_from(system_matrix);
                system_matrix_time.add(1.0 / time_step, mass_matrix);

                // Form right-hand side: M/dt * u_old + right_hand
                mass_matrix.vmult(tmp_vector, solution_old);

                tmp_vector *= (1.0 / time_step);
                tmp_vector += right_hand_side;

                SparseDirectUMFPACK direct;
                // Solve time step
                if (use_direct_solver)
                {
                    direct.initialize(system_matrix_time);
                    direct.vmult(solution, tmp_vector);
                }
                else
                {
                    SolverControl solver_control(1000, 1e-14);
                    SolverCG<> cg(solver_control);

                    PreconditionSSOR<> preconditioner;
                    preconditioner.initialize(system_matrix_time, 1.0);
                    cg.solve(system_matrix_time, solution, tmp_vector, preconditioner);
                }
                std::cout << "norm of vector " << solution.l2_norm() << std::endl;

                solution_old = solution;
                compute_pressure();
                output_results(step);
            }
            // Now compute errors at final time
            compute_errors(cycle);
        }
    }

    // Explicit instantiation
    template class BloodFlowSystem<1, 3>;

} // namespace dealii