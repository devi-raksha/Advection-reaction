// ---------------------------------------------------------------------
//
//
//
// This file is part of the blood_flow_system application, based on
// the deal.II library.
//
// The blood_flow_system application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at the top
// level of this distribution.
// ---------------------------------------------------------------------
#ifndef BLOOD_FLOW_SYSTEM_1D3D_H
#define BLOOD_FLOW_SYSTEM_1D3D_H

#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/tria.h>

// #include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/base/function.h> // for FunctionParser
#include <deal.II/base/function_parser.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_generator.h> // for hyper_cube
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/mesh_loop.h> // for MeshWorker

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.h> // for VectorTools::project

#include <fstream>
#include <iostream>

#include <deal.II/base/function_parser.h> // for FunctionParser

namespace dealii
{

    /**
     * @brief Blood flow system solver for Triangulation<1,3>
     *
     * Solves:
     *   A_t + b · \nabla (AU) = 0
     *   U_t + U \nabla_\Gamma U + (1/ρ) \nabla_\Gamma P(A) + c U = 0
     */

    template <int dim, int spacedim>
    class BloodFlowSystem : public ParameterAcceptor
    {
    public:
        BloodFlowSystem();
        void
        run();

        // Parameter initialization
        void
        initialize_params(const std::string &filename);

    private:
        const FEValuesExtractors::Scalar area_extractor;
        const FEValuesExtractors::Scalar velocity_extractor;

        class ExactSolution : public Function<spacedim>
        {
        public:
            ExactSolution() : Function<spacedim>(2) {} // two components: A and U

            virtual double
            value(const Point<spacedim> &p,
                  const unsigned int component = 0) const override
            {
                // Parameters matching the Python version
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

            virtual void
            vector_value(const Point<spacedim> &p, Vector<double> &values) const override
            {
                Assert(values.size() == 2, ExcDimensionMismatch(values.size(), 2));
                values[0] = value(p, 0);
                values[1] = value(p, 1);
            }

            virtual void
            vector_value_list(const std::vector<Point<spacedim>> &points,
                              std::vector<Vector<double>> &value_list) const override
            {
                const unsigned int n = points.size();
                Assert(value_list.size() == n, ExcDimensionMismatch(value_list.size(), n));
                for (unsigned int i = 0; i < n; ++i)
                    vector_value(points[i], value_list[i]);
            }
        };

        // Declare all necessary objects
        Triangulation<dim, spacedim> triangulation;

        DoFHandler<dim, spacedim> dof_handler;
        std::unique_ptr<FESystem<dim, spacedim>> fe;
        // Parameters exposed via add_parameter()
        unsigned int fe_degree = 1;
        std::vector<double> constants;
        std::string rhs_expression = "0.0";
        std::string initial_A_expression = "1.0";
        std::string initial_U_expression = "0.0";
        std::string pressure_bc_expression = "0.0";
        bool use_direct_solver = true;
        unsigned int n_refinement_cycles = 1;
        unsigned int n_global_refinements = 3;
        double final_time = 1.0;

        // Mesh and system setup
        void
        setup_system();
        void
        read_mesh_and_data();
        void
        create_face_connectivity_map();

        // Assembly routines
        void
        assemble_mass_matrix();
        void
        assemble_system();

        // Time stepping
        void
        solve();
        void
        output_results(const unsigned int cycle) const;

        // Utility
        void
        compute_pressure();

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> system_matrix;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> system_matrix_time;

        Vector<double> solution;
        Vector<double> solution_old;
        Vector<double> right_hand_side;
        Vector<double> tmp_vector;
        Vector<double> pressure;

        // So far we declared the usual objects. Hereafter we declare
        // `FunctionParser<dim>` objects
        FunctionParser<spacedim> parsed_exact_solution;
        FunctionParser<spacedim> initial_A;
        FunctionParser<spacedim> initial_U;
        FunctionParser<spacedim> rhs;
        FunctionParser<spacedim> pressure_bc;
        FunctionParser<spacedim> advection_coeff;

        // Time stepping
        double time_step;
        double time;
        unsigned int n_time_steps;

        // Parameters
        double rho = 1.0;
        double viscosity_c = 1.0;
        double reference_area = 1.0;
        double elastic_modulus = 1.0;
        double reference_pressure = 1.0;
        double theta = 0.5;

        std::string output_filename = "output.vtk";
    };

} // namespace dealii

#endif // BLOOD_FLOW_SYSTEM_1D3D_H