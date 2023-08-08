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
#include "file/matrix.h"


using namespace MR;
using namespace App;
using namespace File::Matrix;

void usage ()
{
  AUTHOR = "J-Donald Tournier (jdtournier@gmail.com)";
  SYNOPSIS = "perform generic inequality-constrained least-squares on input images";

  DESCRIPTION
    + "perform generic inequality-constrained least-squares on input images"
    + "i.e. solve for   Hx = y\n\n     such that   Ax >= t and Bx = s";

  ARGUMENTS
    + Argument ("problem", "the problem matrix H").type_file_in()
    + Argument ("input", "the input vector y.").type_file_in()
    + Argument ("output", "the output solution vector x.").type_file_out ();

  OPTIONS

    + Option ("constraint", "specify A, the inequality constraint matrix. By default, the algorithm will solve for a non-negative solution vector, and set this matrix to the identity.")
    +   Argument ("matrix").type_file_in()

    + Option ("values", "specify t, the inequality constraint vector. By default, the algorithm will set this zero.")
    +   Argument ("matrix").type_file_in()

    + Option ("equality_constraint", "specify B, the (optional) equality constraint matrix.")
    +   Argument ("matrix").type_file_in()

    + Option ("equality_values", "specify s, the (optional) equality constraint vector.")
    +   Argument ("matrix").type_file_in()

    + Option ("num_equalities", "as an alternative to supplying separate A & B, and t & s, you can specify that the last num constraints of the contraint matrix/vector should be treated as equalities (default: 0).")
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
  decltype (problem_matrix) eq_constraint_matrix;
  decltype (problem_vector) eq_constraint_vector;

  auto opt = get_options ("constraint");
  if (opt.size())
    constraint_matrix = load_matrix<compute_type> (opt[0][0]);
  else
    constraint_matrix = decltype(constraint_matrix)::Identity (problem_matrix.cols(), problem_matrix.cols());

  opt = get_options ("values");
  if (opt.size())
    constraint_vector = load_vector<compute_type> (opt[0][0]);

  opt = get_options ("equality_constraint");
  if (opt.size())
    eq_constraint_matrix = load_matrix<compute_type> (opt[0][0]);

  opt = get_options ("equality_values");
  if (opt.size())
    eq_constraint_vector = load_vector<compute_type> (opt[0][0]);


  Math::ICLS::Problem<compute_type> problem;
  if (num_equalities)
    problem = Math::ICLS::Problem<compute_type> (problem_matrix, constraint_matrix, constraint_vector, num_equalities, solution_norm_reg, constraint_norm_reg, max_iterations, tolerance);
  else
    problem = Math::ICLS::Problem<compute_type> (problem_matrix, constraint_matrix, eq_constraint_matrix, constraint_vector, eq_constraint_vector, solution_norm_reg, constraint_norm_reg, max_iterations, tolerance);

  Eigen::VectorXd x (problem_matrix.cols());

  Timer timer;
  Math::ICLS::Solver<compute_type> solve (problem);
  auto niter = solve (x, problem_vector);
  auto elapsed = timer.elapsed();

  if (niter >= solve.problem().max_niter) {
    WARN ("failed to converge in " + str(niter) + " iterations (runtime: " + str(elapsed) + "s)");
  } else {
    CONSOLE ("converged in " + str(niter) + " iterations (runtime: " + str(elapsed) + "s)");
  }

  CONSOLE (str(x.transpose()));
  save_vector (x, argument[2], KeyValues(), false);
}


