
#include "../include/time_dependent_adv_reac_embed.h"

// wind field

template <int dim, int spacedim>
Tensor<1, spacedim>
beta(const Point<spacedim> &p)
{
    Tensor<1, spacedim> wind_field;
    for (unsigned int d = 0; d < spacedim; ++d)
        wind_field[d] = 1.0;
    return wind_field;
}

// scratchdata

template <int dim, int spacedim>
struct ScratchData
{
    ScratchData(const Mapping<dim, spacedim> &mapping,
                const FiniteElement<dim, spacedim> &fe,
                const Quadrature<dim> &quadrature,
                const Quadrature<dim - 1> &quadrature_face,
                const UpdateFlags update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags interface_update_flags =
                    update_values | update_gradients | update_quadrature_points |
                    update_JxW_values | update_normal_vectors)
        : fe_values(mapping, fe, quadrature, update_flags), fe_interface_values(mapping, fe, quadrature_face, interface_update_flags)
    {
    }

    ScratchData(const ScratchData<dim, spacedim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_mapping(),
                    scratch_data.fe_values.get_fe(),
                    scratch_data.fe_values.get_quadrature(),
                    scratch_data.fe_values.get_update_flags()),
          fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                              scratch_data.fe_interface_values.get_fe(),
                              scratch_data.fe_interface_values.get_quadrature(),
                              scratch_data.fe_interface_values.get_update_flags())
    {
    }

    FEValues<dim, spacedim> fe_values;
    FEInterfaceValues<dim, spacedim> fe_interface_values;
};

// CopyDataFace : no change when we change the spacedim

struct CopyDataFace
{
    FullMatrix<double> cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
    std::array<double, 2> values;
    std::array<unsigned int, 2> cell_indices;
};

// CopyData : no change when we change the spacedim

struct CopyData
{
    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace> face_data;

    double value;
    double value_estimator;
    unsigned int cell_index;

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

// auxillary function
template <int dim, int spacedim>
void get_function_jump(const FEInterfaceValues<dim, spacedim> &fe_iv,
                       const Vector<double> &solution,
                       std::vector<double> &jump)
{
    const unsigned int n_q = fe_iv.n_quadrature_points;
    std::array<std::vector<double>, 2> face_values;

    jump.resize(n_q);

    for (unsigned int i = 0; i < 2; ++i)
    {
        face_values[i].resize(n_q);
        fe_iv.get_fe_face_values(i).get_function_values(solution, face_values[i]);
    }

    for (unsigned int q = 0; q < n_q; ++q)
        jump[q] = face_values[0][q] - face_values[1][q];
}

template <int dim, int spacedim>
AdvectionReaction<dim, spacedim>::AdvectionReaction()
    : mapping(1), dof_handler(triangulation)
{
    // Assert(dim >= 1 && dim <= spacedim,
    //  ExcMessage("Invalid dim/spacedim combination.")); //  this line

    add_parameter("Finite element degree", fe_degree);
    add_parameter("Problem constants", constants);
    add_parameter("Output filename", output_filename);
    add_parameter("Use direct solver", use_direct_solver);
    add_parameter("Number of refinement cycles", n_refinement_cycles);
    add_parameter("Number of global refinement", n_global_refinements);
    add_parameter("Refinement", refinement);
    add_parameter("Exact solution expression", exact_solution_expression);
    add_parameter("Boundary conditions expression",
                  boundary_conditions_expression);
    add_parameter("Theta", theta);
    add_parameter("Advection coefficient expression",
                  advection_coefficient_expression);
    add_parameter("Right hand side expression", rhs_expression);

    this->prm.enter_subsection("Error table");
    error_table.add_parameters(this->prm);
    this->prm.leave_subsection();
}

template <int dim, int spacedim>
void AdvectionReaction<dim, spacedim>::initialize_params(const std::string &filename)
{
    ParameterAcceptor::initialize(filename,
                                  "last_used_parameters.prm",
                                  ParameterHandler::Short);

    if (theta < 0.0 || theta > 10.0 || std::abs(theta) < 1e-12)
    {
        throw theta_exc(
            "Theta parameter is not in a suitable range: see paper by "
            "Brezzi, Marini, Suli for an extended discussion");
    }
}

template <int dim, int spacedim>
void AdvectionReaction<dim, spacedim>::parse_string(const std::string &parameters)
{
    ParameterAcceptor::prm.parse_input_from_string(parameters);
    ParameterAcceptor::parse_all_parameters();
}

template <int dim, int spacedim>
void AdvectionReaction<dim, spacedim>::setup_system()
{
    if (!fe)
    {
        fe = std::make_unique<FE_DGQ<dim, spacedim>>(fe_degree);

        std::string vars;
        if (spacedim == 1)
            vars = "x";
        else if (spacedim == 2)
            vars = "x,y";
        else
            vars = "x,y,z";

        exact_solution.initialize(vars, exact_solution_expression, constants);
        rhs.initialize(vars, rhs_expression, constants);
        advection_coeff.initialize(vars,
                                   advection_coefficient_expression,
                                   constants);
        boundary_conditions.initialize(vars,
                                       boundary_conditions_expression,
                                       constants);
    }
    dof_handler.distribute_dofs(*fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp); // DG sparsity pattern generator
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
}

// this function for time dependet problem only to add mass matrix
template <int dim, int spacedim>
void AdvectionReaction<dim, spacedim>::assemble_mass_matrix()
{
    mass_matrix.reinit(sparsity_pattern);
    Vector<double> dummy_rhs(dof_handler.n_dofs());

    const QGauss<dim> quadrature(fe->tensor_degree() + 1);
    const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

    ScratchData<dim, spacedim> scratch_data(mapping, *fe, quadrature, quadrature_face);
    CopyData copy_data;

    const auto cell_worker = [&](const Iterator &cell,
                                 ScratchData<dim, spacedim> &scratch,
                                 CopyData &copy)
    {
        const unsigned int n_dofs = scratch.fe_values.get_fe().n_dofs_per_cell();
        copy.reinit(cell, n_dofs);
        scratch.fe_values.reinit(cell);
        const auto &fe_v = scratch.fe_values;
        const auto &JxW = fe_v.get_JxW_values();

        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
            for (unsigned int i = 0; i < n_dofs; ++i)
                for (unsigned int j = 0; j < n_dofs; ++j)
                    copy.cell_matrix(i, j) += fe_v.shape_value(i, q) *
                                              fe_v.shape_value(j, q) * JxW[q];
    };

    const AffineConstraints<double> constraints;
    const auto copier = [&](const CopyData &c)
    {
        constraints.distribute_local_to_global(c.cell_matrix,
                                               c.cell_rhs,
                                               c.local_dof_indices,
                                               mass_matrix,
                                               dummy_rhs);
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
void AdvectionReaction<dim, spacedim>::assemble_system()
{
    using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

    advec_react_matrix.reinit(sparsity_pattern);
    right_hand_side = 0;

    const QGauss<dim> quadrature(fe->tensor_degree() + 1);
    const QGauss<dim - 1> quadrature_face(fe->tensor_degree() + 1);

    const auto cell_worker = [&](const Iterator &cell,
                                 ScratchData<dim, spacedim> &scratch_data,
                                 CopyData &copy_data)
    {
        const unsigned int n_dofs =
            scratch_data.fe_values.get_fe().n_dofs_per_cell();
        copy_data.reinit(cell, n_dofs);
        scratch_data.fe_values.reinit(cell);

        const auto &q_points = scratch_data.fe_values.get_quadrature_points();
        const auto &fe_v = scratch_data.fe_values;
        const auto &JxW = fe_v.get_JxW_values();

        for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
        {
            // 1) Compute wind vector and dot with gradient → scalar
            const auto wind = beta<spacedim>(q_points[point]); // Tensor<1,spacedim>
            for (unsigned int i = 0; i < n_dofs; ++i)
            {
                const auto grad_i = fe_v.shape_grad(i, point); // Tensor<1,spacedim>
                const double adv_term_i = wind * grad_i;       // dot → double

                for (unsigned int j = 0; j < n_dofs; ++j)
                {
                    const double diff_term =
                        -adv_term_i * fe_v.shape_value(j, point) // advection
                        + advection_coeff.value(q_points[point]) *
                              fe_v.shape_value(i, point) *
                              fe_v.shape_value(j, point); // reaction

                    copy_data.cell_matrix(i, j) += diff_term * JxW[point];
                }

                // right-hand side
                copy_data.cell_rhs(i) += rhs.value(q_points[point]) *
                                         fe_v.shape_value(i, point) * JxW[point];
            }
        }
    };

    const auto boundary_worker = [&](const Iterator &cell,
                                     const unsigned int &face_no,
                                     ScratchData<dim, spacedim> &scratch_data,
                                     CopyData &copy_data)
    {
        scratch_data.fe_interface_values.reinit(cell, face_no);
        const FEFaceValuesBase<dim, spacedim> &fe_face =
            scratch_data.fe_interface_values.get_fe_face_values(0);

        const auto &q_points = fe_face.get_quadrature_points();

        const unsigned int n_facet_dofs = fe_face.get_fe().n_dofs_per_cell();
        const std::vector<double> &JxW = fe_face.get_JxW_values();
        const auto &normals = fe_face.get_normal_vectors();

        std::vector<double> g(q_points.size());
        exact_solution.value_list(q_points, g);

        for (unsigned int point = 0; point < q_points.size(); ++point)

        {
            const auto wind = beta<spacedim>(q_points[point]);
            const double w_dot_n = wind * normals[point]; // dot product
            const double abs_w_dot_n = std::abs(w_dot_n);
            const double beta_dot_n = theta * abs_w_dot_n;

            if (beta_dot_n > 0)

            {
                for (unsigned int i = 0; i < n_facet_dofs; ++i)
                    for (unsigned int j = 0; j < n_facet_dofs; ++j)
                        copy_data.cell_matrix(i, j) +=
                            fe_face.shape_value(i,
                                                point)      // \phi_i
                            * fe_face.shape_value(j, point) // \phi_j
                            * beta_dot_n                    // \beta . n
                            * JxW[point];                   // dx
            }
            else
                for (unsigned int i = 0; i < n_facet_dofs; ++i)
                    copy_data.cell_rhs(i) += -fe_face.shape_value(i, point) // \phi_i
                                             * g[point]                     // g*/
                                             * beta_dot_n                   // \beta . n
                                             * JxW[point];                  // dx
        }
    };

    const auto face_worker = [&](const Iterator &cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim, spacedim> &scratch_data,
                                 CopyData &copy_data)
    {
        FEInterfaceValues<dim, spacedim> &fe_iv = scratch_data.fe_interface_values;
        fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
        const auto &q_points = fe_iv.get_quadrature_points();

        copy_data.face_data.emplace_back();
        CopyDataFace &copy_data_face = copy_data.face_data.back();

        const unsigned int n_dofs = fe_iv.n_current_interface_dofs();
        copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

        copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

        const std::vector<double> &JxW = fe_iv.get_JxW_values();
        const auto &normals = fe_iv.get_normal_vectors();

        for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)

        {
            const auto wind = beta<spacedim>(q_points[qpoint]);
            const double w_dot_n = wind * normals[qpoint]; // dot product
            const double abs_w_dot_n = std::abs(w_dot_n);
            const double beta_dot_n = theta * abs_w_dot_n;

            for (unsigned int i = 0; i < n_dofs; ++i)

            {
                for (unsigned int j = 0; j < n_dofs; ++j)

                {
                    copy_data_face.cell_matrix(i, j) +=
                        (beta<spacedim>(q_points[qpoint]) * normals[qpoint] *
                             fe_iv.average_of_shape_values(j, qpoint) *
                             fe_iv.jump_in_shape_values(i, qpoint) +
                         theta * std::abs(beta_dot_n) *
                             fe_iv.jump_in_shape_values(j, qpoint) *
                             fe_iv.jump_in_shape_values(i, qpoint)) *
                        JxW[qpoint];
                }
            }
        }
    };

    const AffineConstraints<double> constraints;

    const auto copier = [&](const CopyData &c)
    {
        constraints.distribute_local_to_global(c.cell_matrix,
                                               c.cell_rhs,
                                               c.local_dof_indices,
                                               advec_react_matrix,
                                               right_hand_side);

        for (auto &cdf : c.face_data)

        {
            constraints.distribute_local_to_global(cdf.cell_matrix,
                                                   cdf.joint_dof_indices,
                                                   advec_react_matrix);
        }
    };

    ScratchData<dim, spacedim> scratch_data(mapping,
                                            *fe,
                                            quadrature,
                                            quadrature_face);
    CopyData copy_data;

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
void AdvectionReaction<dim, spacedim>::solve()

{
    if (use_direct_solver)

    {
        SparseDirectUMFPACK system_matrix_inverse;
        system_matrix_inverse.initialize(system_matrix);
        system_matrix_inverse.vmult(solution, right_hand_side);
    }
    else
    {
        SolverControl solver_control(1000, 1e-15);
        SolverRichardson<Vector<double>> solver(solver_control);
        PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix, fe->n_dofs_per_cell());
        solver.solve(system_matrix, solution, right_hand_side, preconditioner);
        std::cout << "  Solver converged in " << solver_control.last_step()
                  << " iterations." << std::endl;
    }
}

template <int dim, int spacedim>
void AdvectionReaction<dim, spacedim>::refine_grid()

{
    if (refinement == "residual")

    {
        // compute_local_projection_and_estimate();
        // const double refinement_fraction = 0.6;
        // GridRefinement::refine_and_coarsen_fixed_fraction(
        //   triangulation, error_indicator_per_cell, refinement_fraction, 0.0);
        // triangulation.execute_coarsening_and_refinement();

        triangulation.refine_global(1);
    }
    else if (refinement == "gradient")

    {
        Vector<float> gradient_indicator(triangulation.n_active_cells());

        // DerivativeApproximation::approximate_gradient(mapping,
        //                                               dof_handler,
        //                                               solution,
        //                                               gradient_indicator);

        unsigned int cell_no = 0;
        for (const auto &cell : dof_handler.active_cell_iterators())
            gradient_indicator(cell_no++) *=
                std::pow(cell->diameter(), 1 + 1.0 * dim / 2);

        // GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
        //                                                   gradient_indicator,
        //                                                   0.25,
        //                                                   0.0);

        // triangulation.execute_coarsening_and_refinement();
        // std::cout << gradient_indicator.l2_norm() << '\n';
    }
    else if (refinement == "global")

    {
        triangulation.refine_global(
            3); // just for testing on uniformly refined meshes
    }
    else
    {
        Assert(false, ExcInternalError());
    }
}

template <int dim, int spacedim>
void AdvectionReaction<dim, spacedim>::output_results(const unsigned int cycle) const

{
    const std::string filename = "solution-" + std::to_string(cycle) + ".vtk";
    std::cout << "  Writing solution to <" << filename << ">" << std::endl;
    std::ofstream output(filename);

    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             "u",
                             DataOut<dim, spacedim>::type_dof_data);
    data_out.build_patches(mapping);
    data_out.write_vtk(output);
}

template <int dim, int spacedim>
void AdvectionReaction<dim, spacedim>::compute_error()

{
    error_table.error_from_exact(
        mapping,
        dof_handler,
        solution,
        exact_solution); // be careful: a FD approximation of the gradient is used
}

template <int dim, int spacedim>
double
AdvectionReaction<dim, spacedim>::compute_energy_norm()

{
    energy_norm_square_per_cell.reinit(triangulation.n_active_cells());

    using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;

    const auto cell_worker = [&](const Iterator &cell,
                                 ScratchData<dim, spacedim> &scratch_data,
                                 CopyData &copy_data)
    {
        const unsigned int n_dofs =
            scratch_data.fe_values.get_fe().n_dofs_per_cell();
        copy_data.reinit(cell, n_dofs);
        scratch_data.fe_values.reinit(cell);

        copy_data.cell_index = cell->active_cell_index();

        const auto &q_points = scratch_data.fe_values.get_quadrature_points();
        const FEValues<dim, spacedim> &fe_v = scratch_data.fe_values;
        const std::vector<double> &JxW = fe_v.get_JxW_values();

        double error_square_norm{0.0};
        std::vector<double> sol_u(fe_v.n_quadrature_points);
        fe_v.get_function_values(solution, sol_u);

        for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)

        {
            const double diff =
                (sol_u[point] - exact_solution.value(q_points[point]));
            error_square_norm += diff * diff * JxW[point];
        }
        copy_data.value = error_square_norm;
    };

    const auto face_worker = [&](const Iterator &cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim, spacedim> &scratch_data,
                                 CopyData &copy_data)
    {
        FEInterfaceValues<dim, spacedim> &fe_iv = scratch_data.fe_interface_values;
        fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

        copy_data.face_data.emplace_back();
        CopyDataFace &copy_data_face = copy_data.face_data.back();
        copy_data_face.cell_indices[0] = cell->active_cell_index();
        copy_data_face.cell_indices[1] = ncell->active_cell_index();

        const auto &q_points = fe_iv.get_quadrature_points();
        const unsigned n_q_points = q_points.size();
        const std::vector<double> &JxW = fe_iv.get_JxW_values();
        std::vector<double> g(n_q_points);

        std::vector<double> jump(n_q_points);
        get_function_jump(fe_iv, solution, jump);

        const auto &normals = fe_iv.get_normal_vectors();

        double error_jump_square = 0.0;
        for (unsigned int point = 0; point < n_q_points; ++point)
        {
            // 1) compute wind and normal vectors
            const auto wind =
                beta<spacedim>(q_points[point]); // Tensor<1, spacedim>
            const auto normals =
                fe_iv.get_normal_vectors(); // std::vector<Tensor<1, spacedim>>

            // 2) dot product gives a scalar
            const double w_dot_n = wind * normals[point];

            // 3) take absolute value
            const double abs_w_dot_n = std::abs(w_dot_n);

            // 4) multiply by theta
            const double beta_dot_n = theta * abs_w_dot_n;

            error_jump_square +=
                beta_dot_n * jump[point] * jump[point] * JxW[point];
        }

        copy_data.value = error_jump_square;
    };

    const auto boundary_worker = [&](const Iterator &cell,
                                     const unsigned int &face_no,
                                     ScratchData<dim, spacedim> &scratch_data,
                                     CopyData &copy_data)
    {
        scratch_data.fe_interface_values.reinit(cell, face_no);
        const FEFaceValuesBase<dim, spacedim> &fe_fv =
            scratch_data.fe_interface_values.get_fe_face_values(0);
        const auto &q_points = fe_fv.get_quadrature_points();
        const unsigned n_q_points = q_points.size();
        const std::vector<double> &JxW = fe_fv.get_JxW_values();

        std::vector<double> g(n_q_points);

        std::vector<double> sol_u(n_q_points);
        fe_fv.get_function_values(solution, sol_u);

        const auto &normals = fe_fv.get_normal_vectors();

        double difference_norm_square = 0.0;
        for (unsigned int point = 0; point < q_points.size(); ++point)
        {
            // 1) compute wind and normal
            const auto wind = beta<spacedim>(q_points[point]);
            const auto normals = fe_fv.get_normal_vectors();

            // 2) dot product to scalar
            const double w_dot_n = wind * normals[point];

            // 3) absolute value and multiply by theta
            const double beta_dot_n = theta * std::abs(w_dot_n);

            // 4) compute boundary jump
            const double diff =
                boundary_conditions.value(q_points[point]) - sol_u[point];
            difference_norm_square += beta_dot_n * diff * diff * JxW[point];
        }

        copy_data.value = difference_norm_square;
    };

    const auto copier = [&](const auto &copy_data)
    {
        if (copy_data.cell_index != numbers::invalid_unsigned_int)

        {
            energy_norm_square_per_cell[copy_data.cell_index] += copy_data.value;
        }
        for (auto &cdf : copy_data.face_data)
            for (unsigned int j = 0; j < 2; ++j)
                energy_norm_square_per_cell[cdf.cell_indices[j]] += cdf.values[j];
    };

    ScratchData<dim, spacedim> scratch_data(mapping,
                                            *fe,
                                            QGauss<dim>{fe->tensor_degree() + 1},
                                            QGauss<dim - 1>{fe->tensor_degree() +
                                                            1});

    CopyData copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_own_interior_faces_once |
                              MeshWorker::assemble_boundary_faces,
                          boundary_worker,
                          face_worker);

    const double energy_error = std::sqrt(energy_norm_square_per_cell.l1_norm());
    return energy_error;
}

template <int dim, int spacedim>
void AdvectionReaction<dim, spacedim>::compute_local_projection_and_estimate()
{
    using Iterator = typename DoFHandler<dim, spacedim>::active_cell_iterator;
    error_indicator_per_cell.reinit(triangulation.n_active_cells());

    const auto cell_worker = [&](const Iterator &cell,
                                 ScratchData<dim, spacedim> &scratch_data,
                                 CopyData &copy_data)
    {
        const unsigned int n_dofs =
            scratch_data.fe_values.get_fe().n_dofs_per_cell();

        copy_data.reinit(cell, n_dofs);
        scratch_data.fe_values.reinit(cell);
        copy_data.cell_index = cell->active_cell_index();

        const auto &q_points = scratch_data.fe_values.get_quadrature_points();
        const unsigned n_q_points = q_points.size();

        const FEValues<dim, spacedim> &fe_v = scratch_data.fe_values;
        const std::vector<double> &JxW = fe_v.get_JxW_values();

        std::vector<double> sol_u_at_quadrature_points(fe_v.n_quadrature_points);
        fe_v.get_function_values(solution, sol_u_at_quadrature_points);

        for (unsigned int point = 0; point < n_q_points; ++point)

        {
            for (unsigned int i = 0; i < n_dofs; ++i)

            {
                for (unsigned int j = 0; j < n_dofs; ++j)

                {
                    copy_data.cell_mass_matrix(i, j) +=
                        fe_v.shape_value(i, point) * // phi_i(x_q)
                        fe_v.shape_value(j, point) * // phi_j(x_q)
                        JxW[point];                  // dx(x_q)
                }
                copy_data.cell_mass_rhs(i) +=
                    (rhs.value(q_points[point]) *   // f(x_q)
                         fe_v.shape_value(i, point) // phi_i(x_q)
                     - advection_coeff.value(q_points[point]) *
                           fe_v.shape_value(i, point) *         // c*phi_i(x_q)
                           sol_u_at_quadrature_points[point]) * // u_h(x_q)
                    JxW[point];                                 // dx
            }
        }

        FullMatrix<double> inverse(fe_v.n_quadrature_points,
                                   fe_v.n_quadrature_points);
        inverse.invert(copy_data.cell_mass_matrix);
        Vector<double> proj(fe_v.n_quadrature_points); // projection of (f-c*U_h) on

        inverse.vmult(proj, copy_data.cell_mass_rhs); // M^{-1}*rhs = proj

        double square_norm_over_cell = 0.0;
        for (unsigned int point = 0; point < n_q_points; ++point)

        {
            const double diff = rhs.value(q_points[point]) -
                                sol_u_at_quadrature_points[point] - proj[point];
            square_norm_over_cell += diff * diff * JxW[point];
        }
        copy_data.value_estimator = square_norm_over_cell;
    };

    const auto boundary_worker = [&](const Iterator &cell,
                                     const unsigned int &face_no,
                                     ScratchData<dim, spacedim> &scratch_data,
                                     CopyData &copy_data)
    {
        scratch_data.fe_interface_values.reinit(cell, face_no);
        const FEFaceValuesBase<dim, spacedim> &fe_fv =
            scratch_data.fe_interface_values.get_fe_face_values(0);
        const auto &q_points = fe_fv.get_quadrature_points();
        const unsigned n_q_points = q_points.size();
        const std::vector<double> &JxW = fe_fv.get_JxW_values();

        std::vector<double> g(n_q_points);
        exact_solution.value_list(q_points, g);

        std::vector<double> sol_u(n_q_points);
        fe_fv.get_function_values(solution, sol_u);

        const auto &normals = fe_fv.get_normal_vectors();

        double square_norm_over_bdary_face = 0.;
        for (unsigned int point = 0; point < q_points.size(); ++point)

        {
            const auto wind = beta<spacedim>(q_points[point]);
            const double w_dot_n = wind * normals[point]; // dot product
            const double abs_w_dot_n = std::abs(w_dot_n);
            const double beta_dot_n = theta * abs_w_dot_n;

            if (beta_dot_n < 0) //\partial_{-T} \cap \partial_{- \Omega}

            {
                const double diff =
                    std::abs(beta_dot_n) * (g[point] - sol_u[point]);
                square_norm_over_bdary_face += diff * diff * JxW[point];
            }
        }
        copy_data.value_estimator += square_norm_over_bdary_face;
    };

    const auto face_worker = [&](const Iterator &cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim, spacedim> &scratch_data,
                                 CopyData &copy_data)
    {
        FEInterfaceValues<dim, spacedim> &fe_iv = scratch_data.fe_interface_values;
        fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

        copy_data.face_data.emplace_back();
        CopyDataFace &copy_data_face = copy_data.face_data.back();
        copy_data_face.cell_indices[0] = cell->active_cell_index();
        copy_data_face.cell_indices[1] = ncell->active_cell_index();

        const auto &q_points = fe_iv.get_quadrature_points();
        const unsigned n_q_points = q_points.size();

        const std::vector<double> &JxW = fe_iv.get_JxW_values();
        std::vector<double> g(n_q_points);

        std::vector<double> jump(n_q_points);
        get_function_jump(fe_iv, solution, jump);

        const auto &normals = fe_iv.get_normal_vectors();

        double error_jump_square{0.0};
        for (unsigned int point = 0; point < n_q_points; ++point)

        {
            const auto wind = beta<spacedim>(q_points[point]);
            const double w_dot_n = wind * normals[point]; // dot product
            const double abs_w_dot_n = std::abs(w_dot_n);
            const double beta_dot_n = theta * abs_w_dot_n;

            if (beta_dot_n < 0)

            {
                error_jump_square +=
                    std::abs(beta_dot_n) * jump[point] * jump[point] * JxW[point];
            }
        }

        copy_data_face.values[0] = error_jump_square;
        copy_data_face.values[1] = copy_data_face.values[0];
    };

    ScratchData<dim, spacedim> scratch_data(mapping,
                                            *fe,
                                            QGauss<dim>{fe->tensor_degree() + 1},
                                            QGauss<dim - 1>{fe->tensor_degree() +
                                                            1});

    const auto copier = [&](const auto &copy_data)
    {
        if (copy_data.cell_index != numbers::invalid_unsigned_int)

        {
            error_indicator_per_cell[copy_data.cell_index] +=
                copy_data.value_estimator;
        }
        for (auto &cdf : copy_data.face_data)

        {
            for (unsigned int j = 0; j < 2; ++j)

            {
                error_indicator_per_cell[cdf.cell_indices[j]] += cdf.values[j];
            }
        }
    };

    CopyData copy_data;
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
void AdvectionReaction<dim, spacedim>::run()
{
    // 1. Build initial mesh
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(n_global_refinements);

    // 2. Setup FE system
    setup_system();
    std::cout << "  Number of degrees of freedom: "
              << dof_handler.n_dofs() << std::endl;

    // 3. Assemble matrices
    assemble_mass_matrix();
    assemble_system(); // fills advec_react_matrix and right_hand_side

    // 4. Initial condition (project exact solution at t=0)
    const std::string vars = (spacedim == 1 ? "x" : spacedim == 2 ? "x,y"
                                                                  : "x,y,z");
    FunctionParser<spacedim> init_f;
    init_f.initialize(vars, exact_solution_expression, constants);
    AffineConstraints<double> constraints;
    constraints.close();

    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(fe->tensor_degree() + 1),
                         init_f,
                         solution);

    solution_old = solution;

    // 5. Time loop setup
    n_time_steps = static_cast<unsigned int>(std::round(final_time / time_step));
    system_matrix_time.reinit(sparsity_pattern);
    system_matrix_time.copy_from(advec_react_matrix);
    system_matrix_time.add(1.0 / time_step, mass_matrix);

    SparseDirectUMFPACK direct;
    if (use_direct_solver)
        direct.initialize(system_matrix_time);

    tmp_vector.reinit(dof_handler.n_dofs());

    time = 0.0;

    // 6. Time stepping
    for (unsigned int step = 1; step <= n_time_steps; ++step)
    {
        time += time_step;
        std::cout << "Step " << step << "  t=" << time << std::endl;

        // RHS = (1/dt) M u_old + F
        mass_matrix.vmult(tmp_vector, solution_old);
        tmp_vector *= (1.0 / time_step);
        tmp_vector += right_hand_side;

        if (use_direct_solver)
            direct.vmult(solution, tmp_vector);
        else
        {
            SolverControl solver_control(1000, 1e-12);
            SolverCG<> cg(solver_control);
            PreconditionSSOR<> preconditioner;
            preconditioner.initialize(system_matrix_time, 1.0);
            cg.solve(system_matrix_time, solution, tmp_vector, preconditioner);
        }

        solution_old = solution;

        output_results(step);
        compute_error();
    }
}
template class AdvectionReaction<1, 3>;
// template class AdvectionReaction<2, 3>;
// template class AdvectionReaction<3, 3>;