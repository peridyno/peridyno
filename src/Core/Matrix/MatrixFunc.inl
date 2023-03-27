#include "Vector.h"
#include "Matrix.h"

#ifdef CUDA_BACKEND
	#include "SparseMatrix/svd3_cuda.h"
#endif // CUDA_BACKEND

namespace dyno
{
	template<typename Real>
	DYN_FUNC void jacobiRotate(SquareMatrix<Real, 3> &A, SquareMatrix<Real, 3> &R, int p, int q)
	{
		// rotates A through phi in pq-plane to set A(p,q) = 0
		// rotation stored in R whose columns are eigenvectors of A
		if (A(p, q) == 0.0f)
			return;

		Real d = (A(p, p) - A(q, q)) / (2.0f*A(p, q));
		Real t = 1.0f / (fabs(d) + sqrt(d*d + 1.0f));
		if (d < 0.0f) t = -t;
		Real c = 1.0f / sqrt(t*t + 1);
		Real s = t*c;
		A(p, p) += t*A(p, q);
		A(q, q) -= t*A(p, q);
		A(p, q) = A(q, p) = 0.0f;
		// transform A
		int k;
		for (k = 0; k < 3; k++) {
			if (k != p && k != q) {
				Real Akp = c*A(k, p) + s*A(k, q);
				Real Akq = -s*A(k, p) + c*A(k, q);
				A(k, p) = A(p, k) = Akp;
				A(k, q) = A(q, k) = Akq;
			}
		}
		// store rotation in R
		for (k = 0; k < 3; k++) {
			Real Rkp = c*R(k, p) + s*R(k, q);
			Real Rkq = -s*R(k, p) + c*R(k, q);
			R(k, p) = Rkp;
			R(k, q) = Rkq;
		}
	}

	template<typename Real>
	DYN_FUNC void EigenDecomposition(const SquareMatrix<Real, 3> &A, SquareMatrix<Real, 3> &eigenVecs, Vector<Real, 3> &eigenVals)
	{
		const int numJacobiIterations = 10;
		const Real epsilon = 1e-15f;

		SquareMatrix<Real, 3> D = A;

		// only for symmetric Matrix!
		eigenVecs = SquareMatrix<Real, 3>::identityMatrix();	// unit matrix
		int iter = 0;
		while (iter < numJacobiIterations) {	// 3 off diagonal elements
												// find off diagonal element with maximum modulus
			int p, q;
			Real a, max;
			max = fabs(D(0, 1));
			p = 0; q = 1;
			a = fabs(D(0, 2));
			if (a > max) { p = 0; q = 2; max = a; }
			a = fabs(D(1, 2));
			if (a > max) { p = 1; q = 2; max = a; }
			// all small enough -> done
			if (max < epsilon) break;
			// rotate matrix with respect to that element
			jacobiRotate<Real>(D, eigenVecs, p, q);
			iter++;
		}
		eigenVals[0] = D(0, 0);
		eigenVals[1] = D(1, 1);
		eigenVals[2] = D(2, 2);
	}

	template<typename Real>
	DYN_FUNC void QRDecomposition(SquareMatrix<Real, 3> &A, SquareMatrix<Real, 3> &R, int p, int q)
	{

	}

#ifdef CUDA_BACKEND
	template<typename Real>
	DYN_FUNC void polarDecomposition(const SquareMatrix<Real, 3> &A, SquareMatrix<Real, 3> &R, SquareMatrix<Real, 3> &U, SquareMatrix<Real, 3> &D, SquareMatrix<Real, 3> &V)
	{
		// A = SR, where S is symmetric and R is orthonormal
		// -> S = (A A^T)^(1/2)

		D = SquareMatrix<Real, 3>(0);
		svd(A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2),
			U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2),
			D(0, 0), D(1, 1), D(2, 2),
			V(0, 0), V(0, 1), V(0, 2), V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));

		SquareMatrix<Real, 3> H(0);
		H(0, 0) = 1;
		H(1, 1) = 1;
		H(2, 2) = (V*U.transpose()).determinant();

		R = U*H*V.transpose();

		// A = U D U^T R
/*		Vector<Real, 3> eigenVals;
		SquareMatrix<Real, 3> ATA = A.transpose()*A;
		EigenDecomposition<Real>(ATA, V, eigenVals);

		Real d0 = sqrt(eigenVals[0]);
		Real d1 = sqrt(eigenVals[1]);
		Real d2 = sqrt(eigenVals[2]);
		D = SquareMatrix<Real, 3>(0);
		D(0, 0) = d0;
		D(1, 1) = d1;
		D(2, 2) = d2;

		const Real eps = 1e-6;

		Real l0 = eigenVals[0]; if (l0 <= eps) l0 = 0.0f; else l0 = 1.0f / d0;
		Real l1 = eigenVals[1]; if (l1 <= eps) l1 = 0.0f; else l1 = 1.0f / d1;
		Real l2 = eigenVals[2]; if (l2 <= eps) l2 = 0.0f; else l2 = 1.0f / d2;

		SquareMatrix<Real, 3> invD(0);
		invD(0, 0) = l0;
		invD(1, 1) = l1;
		invD(2, 2) = l2;
		SquareMatrix<Real, 3> S1 = V*invD*V.transpose();
		R = A*S1;*/


/*		SquareMatrix<Real, 3> AAT = A*A.transpose();

		R = SquareMatrix<Real, 3>::identityMatrix();
		Vector<Real, 3> eigenVals;
		EigenDecomposition<Real>(AAT, U, eigenVals);

		Real d0 = sqrt(eigenVals[0]);
		Real d1 = sqrt(eigenVals[1]);
		Real d2 = sqrt(eigenVals[2]);
		D = SquareMatrix<Real, 3>(0);
		D(0, 0) = d0;
		D(1, 1) = d1;
		D(2, 2) = d2;

		const Real eps = 1e-6;

		Real l0 = eigenVals[0]; if (l0 <= eps) l0 = 0.0f; else l0 = 1.0f / d0;
		Real l1 = eigenVals[1]; if (l1 <= eps) l1 = 0.0f; else l1 = 1.0f / d1;
		Real l2 = eigenVals[2]; if (l2 <= eps) l2 = 0.0f; else l2 = 1.0f / d2;

		SquareMatrix<Real, 3> invD(0);
		invD(0, 0) = l0;
		invD(1, 1) = l1;
		invD(2, 2) = l2;

		SquareMatrix<Real, 3> S1 = U*invD*U.transpose();
		R = S1*A;

		Vector<Real, 3> c0, c1, c2;
		c0 = R.col(0);
		c1 = R.col(1);
		c2 = R.col(2);

		int maxCol = 0;
		Real maxMag = c0.normSquared();
		if (c1.normSquared() > maxMag)
		{
			maxCol = 1;
			maxMag = c1.normSquared();
		}
		if (c2.normSquared() > maxMag)
		{
			maxCol = 2;
			maxMag = c2.normSquared();
		}

		if (R.col(maxCol).normSquared() < eps)
		{
			R = SquareMatrix<Real, 3>::identityMatrix();
		}
		else
		{
			Vector<Real, 3> col_0 = R.col(maxCol);
			col_0 = col_0.normalize();
			if (R.col((maxCol + 1) % 3).normSquared() > R.col((maxCol + 2) % 3).normSquared())
			{
			R.setCol(maxCol, col_0);
				Vector<Real, 3> col_1 = R.col((maxCol + 1) % 3);
				col_1 = col_1.normalize();
				R.setCol((maxCol + 1) % 3, col_1);
				Vector<Real, 3> col_2 = col_0.cross(col_1);
				R.setCol((maxCol + 2) % 3, col_2);
			}
			else
			{
				if (R.col((maxCol + 2) % 3).normSquared() > eps)
				{
					Vector<Real, 3> col_2 = R.col((maxCol + 2) % 3);
					col_2 = col_2.normalize();
					R.setCol((maxCol + 2) % 3, col_2);
					Vector<Real, 3> col_1 = col_2.cross(col_0);
					R.setCol((maxCol + 1) % 3, col_1);
				}
				else
				{
					SquareMatrix<Real, 3> unity = SquareMatrix<Real, 3>::identityMatrix();
					Vector<Real, 3> col_1;
					for (int i = 0; i < 3; i++)
					{
						col_1 = unity.col(i);
						if (col_1.cross(col_0).norm() > eps)
							break;
					}
					col_1 = col_0.cross(col_1).normalize();
					col_1 = col_0.cross(col_1).normalize();
					R.setCol((maxCol + 1) % 3, col_1);
					R.setCol((maxCol + 2) % 3, col_0.cross(col_1).normalize());
				}
			}
		}
		*/

	}
#endif

	template<typename Real>
	DYN_FUNC void polarDecomposition(const SquareMatrix<Real, 3> &A, SquareMatrix<Real, 3> &R, SquareMatrix<Real, 3> &U, SquareMatrix<Real, 3> &D)
	{
		// A = SR, where S is symmetric and R is orthonormal
		// -> S = (A A^T)^(1/2)

		// A = U D U^T R

// 		float a11, a12, a13, a21, a22, a23, a31, a32, a33;
// 		a11 = ;	a12 = ;	a13 = ;
// 		a11 = A(0, 0);	a12 = A(0, 1);	a13 = A(0, 2);
// 		a11 = A(0, 0);	a12 = A(0, 1);	a13 = A(0, 2);
// 
// 		float u11, u12, u13, u21, u22, u23, u31, u32, u33;
// 		float s11, s12, s13, s21, s22, s23, s31, s32, s33;
// 		float v11, v12, v13, v21, v22, v23, v31, v32, v33;
		
// 		SquareMatrix<Real, 3> V;
// 		D = SquareMatrix<Real, 3>(0);
// 		svd(A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2),
// 			U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2),
// 			D(0, 0), D(1, 1), D(2, 2),
// 			V(0, 0), V(0, 1), V(0, 2), V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));
// 
// 		SquareMatrix<Real, 3> H(0);
// 		H(0, 0) = 1;
// 		H(1, 1) = 1;
// 		H(2, 2) = (V*U.transpose()).determinant();

//		R = U*H*V.transpose();

		SquareMatrix<Real, 3> AAT;
		AAT(0, 0) = A(0, 0)*A(0, 0) + A(0, 1)*A(0, 1) + A(0, 2)*A(0, 2);
		AAT(1, 1) = A(1, 0)*A(1, 0) + A(1, 1)*A(1, 1) + A(1, 2)*A(1, 2);
		AAT(2, 2) = A(2, 0)*A(2, 0) + A(2, 1)*A(2, 1) + A(2, 2)*A(2, 2);

		AAT(0, 1) = A(0, 0)*A(1, 0) + A(0, 1)*A(1, 1) + A(0, 2)*A(1, 2);
		AAT(0, 2) = A(0, 0)*A(2, 0) + A(0, 1)*A(2, 1) + A(0, 2)*A(2, 2);
		AAT(1, 2) = A(1, 0)*A(2, 0) + A(1, 1)*A(2, 1) + A(1, 2)*A(2, 2);

		AAT(1, 0) = AAT(0, 1);
		AAT(2, 0) = AAT(0, 2);
		AAT(2, 1) = AAT(1, 2);

		R = SquareMatrix<Real, 3>::identityMatrix();
		Vector<Real, 3> eigenVals;
		EigenDecomposition<Real>(AAT, U, eigenVals);

		Real d0 = sqrt(eigenVals[0]);
		Real d1 = sqrt(eigenVals[1]);
		Real d2 = sqrt(eigenVals[2]);
		D = SquareMatrix<Real, 3>(0);
		D(0, 0) = d0;
		D(1, 1) = d1;
		D(2, 2) = d2;

		const Real eps = 1e-15f;

		Real l0 = eigenVals[0]; if (l0 <= eps) l0 = 0.0f; else l0 = 1.0f / d0;
		Real l1 = eigenVals[1]; if (l1 <= eps) l1 = 0.0f; else l1 = 1.0f / d1;
		Real l2 = eigenVals[2]; if (l2 <= eps) l2 = 0.0f; else l2 = 1.0f / d2;

		SquareMatrix<Real, 3> S1;
		S1(0, 0) = l0*U(0, 0)*U(0, 0) + l1*U(0, 1)*U(0, 1) + l2*U(0, 2)*U(0, 2);
		S1(1, 1) = l0*U(1, 0)*U(1, 0) + l1*U(1, 1)*U(1, 1) + l2*U(1, 2)*U(1, 2);
		S1(2, 2) = l0*U(2, 0)*U(2, 0) + l1*U(2, 1)*U(2, 1) + l2*U(2, 2)*U(2, 2);

		S1(0, 1) = l0*U(0, 0)*U(1, 0) + l1*U(0, 1)*U(1, 1) + l2*U(0, 2)*U(1, 2);
		S1(0, 2) = l0*U(0, 0)*U(2, 0) + l1*U(0, 1)*U(2, 1) + l2*U(0, 2)*U(2, 2);
		S1(1, 2) = l0*U(1, 0)*U(2, 0) + l1*U(1, 1)*U(2, 1) + l2*U(1, 2)*U(2, 2);

		S1(1, 0) = S1(0, 1);
		S1(2, 0) = S1(0, 2);
		S1(2, 1) = S1(1, 2);

		R = S1*A;

		// stabilize
		Vector<Real, 3> c0, c1, c2;
		// 		c0 = R.col(0);
		// 		c1 = R.col(1);
		// 		c2 = R.col(2);
		c0[0] = R(0, 0);	c1[0] = R(0, 1);	c2[0] = R(0, 2);
		c0[1] = R(1, 0);	c1[1] = R(1, 1);	c2[1] = R(1, 2);
		c0[2] = R(2, 0);	c1[2] = R(2, 1);	c2[2] = R(2, 2);

		if (c0.normSquared() < eps)
			c0 = c1.cross(c2);
		else if (c1.normSquared() < eps)
			c1 = c2.cross(c0);
		else
			c2 = c0.cross(c1);
		// 		R.col(0) = c0;
		// 		R.col(1) = c1;
		// 		R.col(2) = c2;
		R(0, 0) = c0[0];	R(0, 1) = c1[0];	R(0, 2) = c2[0];
		R(1, 0) = c0[1];	R(1, 1) = c1[1];	R(1, 2) = c2[1];
		R(2, 0) = c0[2];	R(2, 1) = c1[2];	R(2, 2) = c2[2];
	}

	template<typename Real>
	DYN_FUNC void polarDecomposition(const SquareMatrix<Real, 3> &M, SquareMatrix<Real, 3> &R, Real tolerance)
	{
		SquareMatrix<Real, 3> Mt = M.transpose();
		Real Mone = M.oneNorm();
		Real Minf = M.infNorm();
		Real Eone;
		SquareMatrix<Real, 3> MadjTt, Et;
		do
		{
			MadjTt.setRow(0, Mt.row(1).cross(Mt.row(2))); //glm::cross(Mt[1], Mt[2]);
			MadjTt.setRow(1, Mt.row(2).cross(Mt.row(0))); //glm::cross(Mt[2], Mt[0]);
			MadjTt.setRow(2, Mt.row(0).cross(Mt.row(1))); //glm::cross(Mt[0], Mt[1]);

			Real det = Mt(0, 0) * MadjTt(0, 0) + Mt(1, 0) * MadjTt(1, 0) + Mt(2, 0) * MadjTt(2, 0);

			if (fabs(det) < 1.0e-12)
			{
				Vector<Real, 3> len;
				unsigned int index = 0xffffffff;
				for (unsigned int i = 0; i < 3; i++)
				{
					len[i] = MadjTt.col(i).norm();
					if (len[i] > 1.0e-12)
					{
						// index of valid cross product
						// => is also the index of the vector in Mt that must be exchanged
						index = i;
						break;
					}
				}
				if (index == 0xffffffff)
				{
					R = SquareMatrix<Real, 3>::identityMatrix();
					return;
				}
				else
				{
					Mt.setRow(index, Mt.row((index + 1) % 3).cross(Mt.row((index + 2) % 3))); //Mt[index] = glm::cross(Mt[(index + 1) % 3], Mt[(index + 2) % 3]);
					MadjTt.setRow((index + 1) % 3, Mt.row((index + 2) % 3).cross(Mt.row((index) % 3))); //MadjTt[(index + 1) % 3] = glm::cross(Mt[(index + 2) % 3], Mt[(index) % 3]);;
					MadjTt.setRow((index + 2) % 3, Mt.row((index) % 3).cross(Mt.row((index + 1) % 3))); //MadjTt[(index + 2) % 3] = glm::cross(Mt[(index) % 3], Mt[(index + 1) % 3]);
					SquareMatrix<Real, 3> M2 = Mt.transpose();
					Mone = M2.oneNorm();
					Minf = M2.infNorm();
					det = Mt(0, 0) * MadjTt(0, 0) + Mt(1, 0) * MadjTt(1, 0) + Mt(2, 0) * MadjTt(2, 0);
				}
			}

			const Real MadjTone = MadjTt.oneNorm();
			const Real MadjTinf = MadjTt.infNorm();

			const Real gamma = sqrt(sqrt((MadjTone*MadjTinf) / (Mone*Minf)) / fabs(det));

			const Real g1 = gamma*0.5f;
			const Real g2 = 0.5f / (gamma*det);

			for (unsigned char i = 0; i < 3; i++)
			{
				for (unsigned char j = 0; j < 3; j++)
				{
					Et(i, j) = Mt(i, j);
					Mt(i, j) = g1*Mt(i, j) + g2*MadjTt(i, j);
					Et(i, j) -= Mt(i, j);
				}
			}

			Eone = Et.oneNorm();

			Mone = Mt.oneNorm();
			Minf = Mt.infNorm();
		} while (Eone > Mone * tolerance);

		// Q = Mt^T 
		R = Mt.transpose();
	}
}