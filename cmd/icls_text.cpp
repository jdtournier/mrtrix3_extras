/*
 * Copyright (c) 2008-2016 the MRtrix3 contributors
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/
 *
 * MRtrix is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * For more details, see www.mrtrix.org
 *
 */


#include "command.h"
#include "progressbar.h"
#include "timer.h"
#include "math/constrained_least_squares.h"


using namespace MR;
using namespace App;

void usage ()
{
  AUTHOR = "J-Donald Tournier (jdtournier@gmail.com)";
  SYNOPSIS = "perform generic inequality-constrained least-squares on input images";

  DESCRIPTION
    + "perform generic inequality-constrained least-squares on input images"
    + "i.e. solve for   MX = Y\n\n     such that   CX >= t";

  ARGUMENTS
    + Argument ("problem", "the problem matrix M").type_file_in()
    + Argument ("input", "the input vector Y.").type_file_in()
    + Argument ("output", "the output solution vector X.").type_file_out ();

  OPTIONS

    + Option ("constraint", "specify C, the constraint matrix. By default, the algorithm will solve for a non-negative solution vector, and set this matrix to the identity.")
    +   Argument ("matrix").type_file_in()

    + Option ("threshold", "specify t, the constraint thresholds. By default, the algorithm will set this zero.")
    +   Argument ("matrix").type_file_in()

    + Option ("num_equalities", "specify the number of constraints at the end of the contraint matrix/vector that should be treated as equalities (default: 0).")
    +   Argument ("num").type_integer(0)

    + Option ("niter", "specify the maximum number of iterations to perform (default: 10 x num_parameters)")
    +   Argument ("num").type_integer (0)

    + Option ("tolerance", "specify the tolerance on the change in the solution, used to establishe convergence (default: 0.0)")
    +   Argument ("value").type_float (0.0)

    + Option ("solution_norm", "specify the regularisation to apply on the solution norm - useful for poorly condition problems (default: 0.0)")
    +   Argument ("value").type_float (0.0)

    + Option ("constraint_norm", "specify the regularisation to apply on the constraint vector norm - useful for poorly condition problems (default: 0.0)")
    +   Argument ("value").type_float (0.0);
}




typedef double compute_type;




void run ()
{
  auto max_iterations      = get_option_value ("niter",           0  );
  auto tolerance           = get_option_value ("tolerance",       0.0);
  auto solution_norm_reg   = get_option_value ("solution_norm",   0.0);
  auto constraint_norm_reg = get_option_value ("constraint_norm", 0.0);
  auto num_equalities      = get_option_value ("num_equalities",  0);

  auto problem_matrix    = load_matrix<compute_type> (argument[0]);
  auto problem_vector    = load_vector<compute_type> (argument[1]);
  decltype (problem_matrix) constraint_matrix;
  decltype (problem_vector) constraint_vector;

  auto opt = get_options ("constraint");
  if (opt.size()) {
    constraint_matrix = load_matrix<compute_type> (opt[0][0]);
    if (problem_matrix.cols() != constraint_matrix.cols())
      throw Exception ("number of columns in problem matrix \"" + std::string (argument[1]) + "\" does not match number of columns in constraint matrix \"" + std::string(opt[0][0]) + "\"");
  }
  else
    constraint_matrix = decltype(constraint_matrix)::Identity (problem_matrix.cols(), problem_matrix.cols());

  Math::ICLS::Problem<compute_type> problem (problem_matrix, constraint_matrix, solution_norm_reg, constraint_norm_reg, max_iterations, tolerance);

  opt = get_options ("threshold");
  if (opt.size()) {
    constraint_vector = load_vector<compute_type> (opt[0][0]);
    if (constraint_vector.size() != problem.num_constraints())
      throw Exception ("size of threshold vector does not match number of n constraint matrix");
  }

  Eigen::VectorXd x (problem_matrix.cols());

  Timer timer;
  Math::ICLS::Solver<compute_type> solve (problem);
  auto niter = solve (x, problem_vector, constraint_vector, num_equalities);
  auto elapsed = timer.elapsed();

  if (niter >= solve.problem().max_niter) {
    WARN ("failed to converge in " + str(niter) + " iterations (runtime: " + str(elapsed) + "s)");
  } else {
    CONSOLE ("converged in " + str(niter) + " iterations (runtime: " + str(elapsed) + "s)");
  }

  CONSOLE (str(x.transpose()));
  save_vector (x, argument[2], KeyValues(), false);
}


