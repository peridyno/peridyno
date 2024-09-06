#include "CalculateNormalSDF.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(CalculateNormalSDF, TDataType)


	template<typename Real, typename Coord, typename Tetrahedron>
	__global__ void K_UpdateNormalSDF(
		DArray<Coord> posArr,
		DArray<Tetrahedron> tets,
		DArray<Real> distance,
		DArray<Coord> normalTet)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tets.size()) return;

		Coord posCenter = (posArr[tets[pId][0]] + posArr[tets[pId][1]] + posArr[tets[pId][2]] + posArr[tets[pId][3]]) / 4.0f;
		Coord dir(0);
		for (int i = 0; i < 4; i++)
		{
			dir += distance[tets[pId][i]] * (posArr[tets[pId][i]] - posCenter) / (posArr[tets[pId][i]] - posCenter).norm();
		}
		if (dir.norm() > EPSILON)
			dir /= dir.norm();
		else
			dir = Coord(0, 1, 0);
		normalTet[pId] = dir;
	}

	template<typename TDataType>
	void CalculateNormalSDF<TDataType>::compute()
	{
		cuExecute(this->inTets()->size(),
			K_UpdateNormalSDF,
			this->inPosition()->getData(),
			this->inTets()->getData(),
			this->inDisranceSDF()->getData(),
			this->inNormalSDF()->getData()
			);
	}

	/*template<typename TDataType>
	void CalculateNormalSDF<TDataType>::resetStates()
	{

	}*/

	DEFINE_CLASS(CalculateNormalSDF);
}