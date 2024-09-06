#include "SharedFunc.h"

namespace dyno
{
	template <typename Coord, typename Bond>
	__global__ void K_UpdateRestShape(
		DArrayList<Bond> bonds,
		DArrayList<int> nbr,
		DArray<Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Bond np;

		List<Bond>& bonds_i = bonds[pId];
		List<int>& list_id_i = nbr[pId];
		int nbSize = list_id_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_id_i[ne];
			np.idx = j;
			np.xi = pos[j] - pos[pId];
//			np.weight = 1;

			bonds_i.insert(np);
// 			if (pId == j)
// 			{
// 				Bond np_0 = rest_shape_i[0];
// 				rest_shape_i[0] = np;
// 				rest_shape_i[ne] = np_0;
// 			}
		}
	}


	template<typename Coord, typename Bond>
	void constructRestShape(DArrayList<Bond>& shape, DArrayList<int>& nbr, DArray<Coord>& pos)
	{
		cuExecute(nbr.size(),
			K_UpdateRestShape,
			shape,
			nbr,
			pos);
	}

	__global__ void K_AddOne(
		DArray<uint> num,
		DArrayList<int> nbr)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;

		num[tId] = nbr[tId].size();
	}

// 	template <typename Coord, typename Bond>
// 	__global__ void K_UpdateRestShapeSelf(
// 		DArrayList<Bond> shape,
// 		DArray<Coord> pos)
// 	{
// 		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
// 		if (tId >= pos.size()) return;
// 
// 		Bond np;
// 		np.index = tId;
// 		np.pos = pos[tId];
// 		np.weight = 1;
// 
// 		shape[tId].insert(np);
// 	}

	template<typename Coord, typename Bond>
	void constructRestShapeWithSelf(DArrayList<Bond>& shape, DArrayList<int>& nbr, DArray<Coord>& pos)
	{
		DArray<uint> num(nbr.size());

		cuExecute(nbr.size(),
			K_AddOne,
			num,
			nbr);

		shape.resize(num);

// 		cuExecute(nbr.size(),
// 			K_UpdateRestShapeSelf,
// 			shape,
// 			pos);

		cuExecute(nbr.size(),
			K_UpdateRestShape,
			shape,
			nbr,
			pos);

		num.clear();
	}

	template void constructRestShape<Vec3f, TBond<DataType3f>>(DArrayList<TBond<DataType3f>>& shape, DArrayList<int>& nbr, DArray<Vec3f>& pos);
	template void constructRestShapeWithSelf<Vec3f, TBond<DataType3f>>(DArrayList<TBond<DataType3f>>& shape, DArrayList<int>& nbr, DArray<Vec3f>& pos);
}