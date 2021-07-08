#include "SharedFunc.h"

namespace dyno
{
	template <typename Coord, typename NPair>
	__global__ void K_UpdateRestShape(
		DArrayList<NPair> shape,
		DArrayList<int> nbr,
		DArray<Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		NPair np;

		List<NPair>& rest_shape_i = shape[pId];
		List<int>& list_id_i = nbr[pId];
		int nbSize = list_id_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_id_i[ne];
			np.index = j;
			np.pos = pos[j];
			np.weight = 1;

			rest_shape_i.insert(np);
			if (pId == j)
			{
				NPair np_0 = rest_shape_i[0];
				rest_shape_i[0] = np;
				rest_shape_i[ne] = np_0;
			}
		}
	}


	template<typename Coord, typename NPair>
	void constructRestShape(DArrayList<NPair>& shape, DArrayList<int>& nbr, DArray<Coord>& pos)
	{
		cuExecute(nbr.size(),
			K_UpdateRestShape,
			shape,
			nbr,
			pos);
	}

	template void constructRestShape<Vec3f, TPair<DataType3f>>(DArrayList<TPair<DataType3f>>& shape, DArrayList<int>& nbr, DArray<Vec3f>& pos);
}