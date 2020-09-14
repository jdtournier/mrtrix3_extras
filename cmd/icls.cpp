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
#include "image.h"
#include "algo/threaded_loop.h"
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
    + Argument ("input", "the input images Y.").type_image_in ()
    + Argument ("problem", "the problem matrix M")
    + Argument ("output", "the output solution image X.").type_image_out ();

  OPTIONS

    + Option ("mask", "only perform computation within the specified binary brain mask image.")
    +   Argument ("image").type_image_in()

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
    +   Argument ("value").type_float (0.0)

    + Option ("prediction", "output predicted image")
    +   Argument ("image").type_image_out();
}




typedef float value_type;
typedef double compute_type;

class Processor {
  public:
    Processor (const Math::ICLS::Problem<compute_type>& problem, Image<value_type>& prediction, Image<bool>& mask) :
      solve (problem),
      x(problem.H.cols()),
      b(problem.H.rows()),
      prediction (prediction),
      mask (mask) { }

    void operator() (Image<value_type>& in, Image<value_type>& out)
    {
      if (mask.valid()) {
       assign_pos_of (in, 0, 3).to (mask);
       if (!mask.value())
         return;
      }

      for (auto l = Loop (3) (in); l; ++l)
        b[in.index(3)] = in.value();

      auto niter = solve (x, b);
      if (niter >= solve.problem().max_niter)
        INFO ("voxel at [ " + str(in.index(0)) + " " + str(in.index(1)) + " " + str(in.index(2)) + " ] failed to converge");

      for (auto l = Loop (3) (out); l; ++l)
        out.value() = x[out.index(3)];

      if (prediction.valid()) {
        assign_pos_of (in, 0, 3).to (prediction);
        b = solve.problem().H * x;
        for (auto l = Loop (3) (prediction); l; ++l)
          prediction.value() = b[prediction.index(3)];
      }
    }

    Math::ICLS::Solver<compute_type> solve;
    Eigen::VectorXd x, b;
    Image<value_type> prediction;
    Image<bool> mask;
};





void run ()
{
  auto max_iterations      = get_option_value ("niter",           0  );
  auto tolerance           = get_option_value ("tolerance",       0.0);
  auto solution_norm_reg   = get_option_value ("solution_norm",   0.0);
  auto constraint_norm_reg = get_option_value ("constraint_norm", 0.0);
  auto num_equalities      = get_option_value ("num_equalities",    0);

  auto problem_matrix    = load_matrix<compute_type> (argument[1]);
  decltype (problem_matrix) constraint_matrix;

  auto opt = get_options ("constraint");
  if (opt.size()) {
    constraint_matrix = load_matrix<compute_type> (opt[0][0]);
    if (problem_matrix.cols() != constraint_matrix.cols())
      throw Exception ("number of columns in problem matrix \"" + std::string (argument[1]) + "\" does not match number of columns in constraint matrix \"" + std::string(opt[0][0]) + "\"");
  }
  else
    constraint_matrix = decltype(constraint_matrix)::Identity (problem_matrix.cols(), problem_matrix.cols());


  Eigen::Matrix<compute_type, Eigen::Dynamic, 1> threshold;
  opt = get_options ("threshold");
  if (opt.size()) {
    threshold = load_vector<compute_type> (opt[0][0]);
    if (threshold.size() != constraint_matrix.rows())
      throw Exception ("size of threshold vector does not match number of n constraint matrix");
  }

  Math::ICLS::Problem<compute_type> problem (problem_matrix, constraint_matrix, threshold, num_equalities, solution_norm_reg, constraint_norm_reg, max_iterations, tolerance);

  auto in = Image<value_type>::open (argument[0]);
  if (in.size(3) != ssize_t (problem.num_measurements()))
    throw Exception ("number of volumes in input image \"" + std::string (argument[0]) + "\" does not match number of columns in problem matrix \"" + std::string (argument[1]) + "\"");

  Image<bool> mask;
  opt = get_options ("mask");
  if (opt.size()) {
    mask = Image<bool>::open (opt[0][0]);
    check_dimensions (mask, in, 0, 3);
  }

  opt = get_options ("prediction");
  Image<value_type> prediction;
  if (opt.size()) {
    Header header = in;
    header.datatype() = DataType::Float32;
    prediction = Image<value_type>::create (opt[0][0], header);
  }

  Header header (in);
  header.size (3) = problem.num_parameters();
  auto out = Image<value_type>::create (argument[2], header);

  ThreadedLoop ("performing constrained least-squares fit", in, 0, 3)
    .run (Processor (problem, prediction, mask), in, out);
}

