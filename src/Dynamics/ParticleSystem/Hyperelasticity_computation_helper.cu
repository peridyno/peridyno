#include "HyperelasticityModule_test.h"
#include "Utility.h"
#include "Algorithm/MatrixFunc.h"

#include "Kernel.h"
#include <math.h>



namespace dyno
{
	
	//-test: to find all the deformation gradient matrices
	// these deformation gradients are mat3x3
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	GPU_FUNC void getDeformationGradient(
		int curParticleID,
		DeviceArray<Coord>& position,
		NeighborList<NPair>& restShapes,
		Real horizon,
		Matrix* pResultMatrix)
	{

		Matrix& resultMatrix = *pResultMatrix;

		resultMatrix = Matrix(0.0f);

		CorrectedKernel<Real> g_weightKernel;

		NPair np_i = restShapes.getElement(curParticleID, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(curParticleID);

		Real total_weight = Real(0);
		Matrix deform_i = Matrix(0.0f);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(curParticleID, ne);
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Real r = (rest_j - rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = g_weightKernel.Weight(r, horizon);

				Coord p = (position[j] - position[curParticleID]) / horizon;
				Coord q = (rest_j - rest_i) / horizon * weight;

				deform_i(0, 0) += p[0] * q[0]; deform_i(0, 1) += p[0] * q[1]; deform_i(0, 2) += p[0] * q[2];
				deform_i(1, 0) += p[1] * q[0]; deform_i(1, 1) += p[1] * q[1]; deform_i(1, 2) += p[1] * q[2];
				deform_i(2, 0) += p[2] * q[0]; deform_i(2, 1) += p[2] * q[1]; deform_i(2, 2) += p[2] * q[2];
				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			deform_i *= (1.0f / total_weight);
		}
		else
		{
			total_weight = 1.0f;
		}

		resultMatrix = deform_i;
		*pResultMatrix = resultMatrix;

	}




	// -test: singular value decomposition
	// matrix A = U * S * V^T
	// matrix A are 3x3
	template <typename Matrix>
	GPU_FUNC void getSVDmatrix(
		const Matrix A,
		Matrix* pMatU,
		Matrix* pMatS,
		Matrix* pMatV) 
	{
		typedef typename Matrix::VarType Real;
		typedef typename Vector<Real, 3> Coord;

		Matrix& U = *pMatU;
		Matrix& S = *pMatS;
		Matrix& V = *pMatV;

		int rowA = A.rows();
		int colA = A.cols();

		S = Matrix(0.0);
		U = A;
		V = Matrix::identityMatrix();

		// set the tolerance
		Real TOL = 1e-5;
		// current tolerance
		Real converge = TOL + 1.0;


		// Jacobi Rotation Loop
		// reference to http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html
		while (converge > TOL) {
			converge = 0.0;
			for (int j = 1; j <= colA-1; ++j) {
				for (int i = 0; i <= j - 1; ++i) {
					// compute [alpha gamma; gamma beta]=(i,j) submatrix of U^T * U
					Real coeAlpha = Real(0);
					for (int k = 0; k <= colA - 1; ++k) {
						coeAlpha += U(k, i)*U(k, i);
					}
					Real coeBeta = Real(0);
					for (int k = 0; k <= colA - 1; ++k) {
						coeBeta += U(k, j)*U(k, j);
					}
					Real coeGamma = Real(0);
					for (int k = 0; k <= colA - 1; ++k) {
						coeGamma += U(k, i)*U(k, j);
					}

					// find current tolerance
					if (coeGamma==0.0 || coeAlpha==0.0 || coeAlpha==0.0) { continue; }
					converge = max(converge, abs(coeGamma)/sqrt(coeAlpha*coeBeta));

					// compute Jacobi Rotation
					// take care Gamma may be zero
					Real coeZeta, coeTan=0.0;
					if (converge > TOL) {
						coeZeta = (coeBeta - coeAlpha) / (2 * coeGamma);
						int signOfZeta = (Real(0.0) < coeZeta) - (coeZeta < Real(0.0));
						if (signOfZeta == 0) { signOfZeta = 1; }
						assert(signOfZeta==1 || signOfZeta==-1);
						coeTan = signOfZeta / ( abs(coeZeta) + sqrt(1.0 + coeZeta * coeZeta));
					}
					else {
						coeTan = 0.0;
					}
					
					Real coeCos = Real(1.0) / (sqrt(1.0 + coeTan * coeTan));
					Real coeSin = coeCos * coeTan;

					// update columns i and j of U
					for (int k = 0; k <= colA - 1; ++k) {
						Real tmp = U(k, i);
						U(k, i) = coeCos * tmp - coeSin * U(k, j);
						U(k, j) = coeSin * tmp + coeCos * U(k, j);
					}
					//update columns of V
					for (int k = 0; k <= colA - 1; ++k) {
						Real tmp = V(k, i);
						V(k, i) = coeCos * tmp - coeSin * V(k, j);
						V(k, j) = coeSin * tmp + coeCos * V(k, j);
					}

				}
			}
		}

		// find singular values and normalize U
		Coord singValues = Coord(1.0);
		for (int j = 0; j <= colA - 1; ++j) {
			singValues[j] = U.col(j).norm();
			if (abs(singValues[j]) > 0.0) {
				for (int i = 0; i <= rowA - 1; ++i) {
					U(i, j) = U(i, j) / singValues[j];
				}
			}
			else singValues[j] = 0.0;
		}

		// get matrix S
		for (int i = 0; i <= rowA - 1; ++i) {
			S(i, i) = singValues[i];
		}

		*pMatU = U;
		*pMatS = S;
		*pMatV = V;
	}

	template <typename Matrix>
	GPU_FUNC void getGInverseMatrix_SVD(
		Matrix U,
		Matrix S,
		Matrix V,
		Matrix* pResultMat)
	{
		typedef typename Matrix::VarType Real;
		typedef typename Vector<Real, 3> Coord;

		Matrix& resultMat = *pResultMat;

		int colS = S.cols();

		// inverse matrix S
		for (int j = 0; j <= colS - 1; ++j) {
			if (S(j, j) > EPSILON) { S(j, j) = 1.0 / S(j, j); }
			else { S(j, j) = 0.0; }
		}

		//transpose mat U
		U = U.transpose();

		// A = U*S*V^T;  so that A^-1 = V * S^-1 * U^T
		resultMat = V * S * U;

		*pResultMat = resultMat;
	}



	template <typename Matrix>
	GPU_FUNC void GInverseMat(
		Matrix A,
		Matrix* pResultMat) 
	{
		Matrix& resultMat = *pResultMat;

		Matrix matU = Matrix(0.0), matV = Matrix(0.0), matS = Matrix(0.0);
		getSVDmatrix(A, &matU, &matS, &matV);
		getGInverseMatrix_SVD(matU, matS, matV, &resultMat);

		*pResultMat = resultMat;
	}


	//-test: to find generalized inverse of all deformation gradient matrices
	// these deformation gradients are mat3x3, may be singular
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	GPU_FUNC void get_GInverseOfF_PiolaKirchhoff(
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real distance,
		Real mu,
		Real lambda,
		DeviceArray<Matrix> resultGInverseMatrices,
		DeviceArray<Matrix> firstPiolaKirchhoffMatrices)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Real total_weight = Real(0);
		Matrix deform_i = Matrix(0.0f);
		getDeformationGradient(
			pId,
			position,
			restShapes,
			horizon,
			&deform_i);
		
		// deformation gradients mat SVD
		Matrix matU = Matrix(0.0), matV = Matrix(0.0), matS = Matrix(0.0);
		getSVDmatrix(deform_i, &matU, &matS, &matV);

		// get g-inverse of deformaition gradients F
		Matrix matInverseF = Matrix(0.0);
		getGInverseMatrix_SVD(matU, matS, matV, &matInverseF);
		for (int i = 0; i <= matInverseF.rows() - 1; ++i) {
			if (matInverseF(i, i) == 0.0) { matInverseF(i, i) = 1.0; }
		}
		resultGInverseMatrices[pId] = matInverseF;

		// find strain tensor E = 1/2(F^T * F - I)
		Matrix strainMat = 0.5*(deform_i.transpose() * deform_i - Matrix::identityMatrix());
		// find first Piola-Kirchhoff matix; StVK material
		firstPiolaKirchhoffMatrices[pId] = deform_i * (2 * mu*strainMat + lambda * strainMat.trace() * Matrix::identityMatrix());
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	GPU_FUNC void getJacobiMethod_D_R_b_constants(
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		DeviceArray<Coord> velocity,

		Real horizon,
		Real mass,
		Real volume,
		Real dt,

		DeviceArray<Matrix> deformGradGInverseMats,
		DeviceArray<Matrix> PiolaKirchhoffMats,

		DeviceArray< Matrix > arrayR,
		DeviceArray<int> arrayRIndex,
		DeviceArray<Matrix> arrayDiagInverse,
		DeviceArray<Coord> array_b)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		int curParticleID = pId;


		CorrectedKernel<Real> g_weightKernel;

		NPair np_i = restShapes.getElement(curParticleID, 0);
		Coord rest_i = np_i.pos;
		// size_i include this particle itself
		int size_i = restShapes.getNeighborSize(curParticleID);

		Real total_weight = Real(0);

		// compute mat R
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(curParticleID, ne);
			Coord rest_j = np_j.pos;

			int j = np_j.index;

			Real r = (rest_j - rest_i).norm();
			Real weight;
			if (r > EPSILON)
			{
				// compute weights: w_ij w_ji
				weight = g_weightKernel.Weight(r, horizon);

				total_weight += weight;
			}
			else {
				weight = 0.0;
			}

			arrayR[ arrayRIndex[curParticleID]+ ne] = (-1.0) * dt * dt* volume* volume* (
				weight * PiolaKirchhoffMats[curParticleID] * deformGradGInverseMats[curParticleID]
				+ weight * PiolaKirchhoffMats[j] * deformGradGInverseMats[j]);
		}

		if (total_weight > EPSILON)
		{
			for (int ne = 1; ne < size_i; ne++) {
				arrayR[arrayRIndex[curParticleID] + ne] *= 1.0f / (total_weight);
			}
		}
		else
		{
			total_weight = 1.0f;
		}

		// compute mat D
		Matrix matDiag = mass * Matrix::identityMatrix();
		for (int ne = 1; ne < size_i; ne++) {
			matDiag += (-1.0)* arrayR[arrayRIndex[curParticleID] + ne];
		}
		Matrix matDiagInverse = Matrix(0.0);
		GInverseMat(matDiag, &matDiagInverse);
		arrayDiagInverse[curParticleID] = matDiagInverse;

		array_b[curParticleID] = mass * (position[curParticleID] + dt * velocity[curParticleID]);
	}

	
	// one iteration of Jacobi method 
	template <typename Coord, typename Matrix, typename NPair>
	GPU_FUNC void JacobiStep(
		DeviceArray<Matrix> arrayR,
		DeviceArray<int> arrayRIndex,
		DeviceArray<Matrix> arrayDiagInverse,
		DeviceArray<Coord> array_b,
		
		NeighborList<NPair> restShapes,

		DeviceArray<Coord> y_pre,
		DeviceArray<Coord> y_next) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_pre.size()) return;

		NPair np_i = restShapes.getElement(pId, 0);

		// size_i include this particle itself
		int size_i = restShapes.getNeighborSize(pId);

		Coord sigma = Coord(0.0);
		for (int ne = 1; ne < size_i; ++ne) {
			int index_j = restShapes.getElement(pId, ne).index;
			Matrix remainder = arrayR[arrayRIndex[pId] + ne];
			sigma += remainder * y_pre[index_j];
		}
		y_next[pId] = arrayDiagInverse[pId] * (array_b[pId] - sigma);
	}

	/*************************** functions below are used for debug *******************************************/

	// cuda test function
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	GPU_FUNC void get_DeformationMat_F(
		DeviceArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real distance,
		Real mu,
		Real lambda,
		DeviceArray<Matrix> resultDeformationMatrices)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;


		Real total_weight = Real(0);
		Matrix deform_i = Matrix(0.0f);
		getDeformationGradient(
			pId,
			position,
			restShapes,
			horizon,
			&deform_i);

		resultDeformationMatrices[pId] = deform_i;
		
	}

	template <typename NPair>
	__global__ void findNieghborNums(
		NeighborList<NPair> restShapes,
		DeviceArray<int> neighborNums)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= neighborNums.size()) return;

		// size_i include this particle itself
		int size_i = restShapes.getNeighborSize(pId);
		neighborNums[pId] = size_i;
	}

	template <typename Real, typename Coord>
	GPU_FUNC void computeDelta_vec_const(
		DeviceArray<Coord> vec1,
		Coord vec2,
		DeviceArray<Real> delta_norm)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vec1.size()) return;

		delta_norm[pId] = (vec1[pId] - vec2).norm();
	}

	template <typename Real, typename Coord>
	GPU_FUNC void computeDelta_vec(
		DeviceArray<Coord> vec1,
		DeviceArray<Coord> vec2,
		DeviceArray<Real> delta_norm)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vec1.size()) return;

		delta_norm[pId] = (vec1[pId] - vec2[pId]).norm();
	}

	template <typename Real, typename Matrix>
	GPU_FUNC void computeDelta_mat_const(
		DeviceArray<Matrix> mat1,
		Matrix mat2,
		DeviceArray<Real> delta_norm)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mat1.size()) return;

		delta_norm[pId] = (mat1[pId] - mat2).frobeniusNorm();
	}

	template <typename Real, typename Matrix>
	GPU_FUNC void computeDelta_mat(
		DeviceArray<Matrix> mat1,
		DeviceArray<Matrix> mat2,
		DeviceArray<Real> delta_norm)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mat1.size()) return;

		delta_norm[pId] = (mat1[pId] - mat2[pId]).frobeniusNorm();
	}

	template <typename Coord>
	bool isExitNaN_vec3f(Coord vec) {
		float tmp = vec[0] + vec[1] + vec[2];
		if (std::isnan(tmp))return true;
		else return false;
	}

	template <typename Matrix>
	bool isExitNaN_mat3f(Matrix mat) {
		float tmp = mat(0, 0) + mat(0, 1) + mat(0, 2) + mat(1, 0) 
					+ mat(1, 1) + mat(1, 2) + mat(2, 0) + mat(2, 1) + mat(2, 2);
		if (std::isnan(tmp))return true;
		else return false;
	}

}
