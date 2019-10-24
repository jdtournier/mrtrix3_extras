#include "command.h"
#include "image.h"
#include "algo/threaded_loop.h"

using namespace MR;
using namespace App;

void usage ()
{
  AUTHOR = "Joe Bloggs (joe.bloggs@acme.org)";
  SYNOPSIS = "compute matrix multiplication of each voxel vector of "
    "values with matrix";
  ARGUMENTS
  + Argument ("in", "the input image.").type_image_in ()
  + Argument ("matrix", "the mixing matrix.").type_file_in ()
  + Argument ("out", "the output image.").type_image_out ();
}


using value_type = float;
using compute_type = float;


class MathMulFunctor {
  public:
    MathMulFunctor (const Eigen::MatrixXd& matrix) :
      M (matrix),
      vec_in (M.cols()),
      vec_out (M.rows()) { }

    void operator() (Image<value_type>& in, Image<value_type>& out)
    {
      vec_in = in.row(3);
      vec_out = M * vec_in;
      out.row(3) = vec_out;
    }

  protected:
    const Eigen::MatrixXd& M;
    Eigen::VectorXd vec_in, vec_out;
};


void run ()
{
  auto in = Image<value_type>::open (argument[0]);

  if (in.ndim() != 4)
    throw Exception ("expected 4D input image");

  auto matrix = load_matrix (argument[1]);

  if (matrix.cols() != in.size(3))
    throw Exception ("number of volumes does not match number of columns of matrix");

  Header header (in);
  header.datatype() = DataType::Float32;
  header.size(3) = matrix.rows();

  auto out = Image<value_type>::create (argument[2], header);

  ThreadedLoop ("performing matrix multiplication", in, 0, 3)
    .run (MathMulFunctor (matrix), in , out);
}
