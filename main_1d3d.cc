/* -----------------------------------------------------------------------------
 *
 *  SPDX-License-Identifier: LGPL-2.1-or-later
 *  Copyright:
 *      2025  Your Name or Institution
 *
 *  This file is part of the blood-flow example built on the deal.II library.
 *  It provides the top-level executable that steers the templated
 *  BloodFlowSystem<1,3> class.  The structure mirrors main_embedded.cc so that
 *  build rules, CMake targets, and user habits remain consistent across
 *  multiple applications in the same repository.
 *
 *  --------------------------------------------------------------------------
 */

#include <deal.II/base/logstream.h>         // deallog control
#include <deal.II/base/parameter_handler.h> // only needed for better hints

#include "include/blood-flow-system-1d3d.h" // header exposing BloodFlowSystem

using namespace dealii;

int
main(int argc, char **argv)
{
  try
    {
      /* --------------------------- 1. Locate parameter file -----------------
       */
      std::string par_name;
      if (argc > 1)
        par_name = argv[1]; // first CLI argument
      else
        par_name = "parameters.prm"; // fallback default

      /* ---------------------- 2. Initialise deal.II logging -----------------
       */
      dealii::deallog.depth_console(
        2); // 0 = silent, 1 = headings, ≥2 = verbose

      /* ------------------------- 3. Set up the problem ----------------------
       */
      BloodFlowSystem<1, 3> problem; // 1-dim geometry embedded in ℝ³
      problem.initialize_params(par_name);
      problem.run(); // perform mesh gen, time loop, I/O

      /* ------------------------ 4. Normal program exit ----------------------
       */
      return 0;
    }

  /* --------------------- 5. Dedicated exception catchers ------------------ */
  catch (const std::exception &theta_range)
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Exception on processing parameters: \n"
                << theta_range.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (std::exception &exc)
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Exception on processing: \n"
                << exc.what() << '\n'
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (...)
    {
      std::cerr << '\n'
                << "----------------------------------------------------\n"
                << "Unknown exception!\n"
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
}
