// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the bare-dealii-app application, based on the
// deal.II library.
//
// The bare-dealii-app application is free software; you can use it,
// redistribute it, and/or modify it under the terms of the Apache-2.0 License
// WITH LLVM-exception as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md
// at the top level of the bare-dealii-app distribution.
//
// ---------------------------------------------------------------------

// Test constant functions

#include <deal.II/base/parsed_function.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/vector_tools.h>

#include "blood_flow_system_updated_1d3d.cc"
#include "tests.h"

using namespace dealii;

void
test_constant()
{
  BloodFlowSystem<1, 3> problem; // 1-dim geometry embedded in \mathbb{R}^3
  GridGenerator::hyper_cube(problem.triangulation);
  problem.triangulation.refine_global(3);

  problem.setup_system();

  FunctionParser<3> my_function(2);
  my_function.initialize("x,t", "1.0; 0.0", {});

  // Project initial conditions
  AffineConstraints<double> constraints;
  constraints.close();
  VectorTools::project(problem.dof_handler,
                       constraints,
                       QGauss<1>(problem.fe_degree + 1),
                       my_function,
                       problem.solution);

  problem.solution_old = problem.solution;
  problem.assemble_system();


  FunctionParser<3> my_test_function(2);
  my_test_function.initialize("x,t", "1.0; 0.0", {});

  VectorTools::project(problem.dof_handler,
                       constraints,
                       QGauss<1>(problem.fe_degree + 1),
                       my_test_function,
                       problem.solution);

  auto rhs = problem.solution;
  problem.system_matrix.vmult(rhs, problem.solution);

  deallog << "Solution norm: " << problem.solution.linfty_norm() << std::endl;
}


int
main()
{
  initlog();
  test_constant();
}
