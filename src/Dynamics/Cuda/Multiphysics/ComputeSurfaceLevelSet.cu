#include "ComputeSurfaceLevelset.h"

namespace dyno
{
	template<typename TDataType>
	ComputeSurfaceLevelset<TDataType>::ComputeSurfaceLevelset()
		:ConstraintModule()
	{
	
	}


	template<typename TDataType>
	ComputeSurfaceLevelset<TDataType>::~ComputeSurfaceLevelset()
	{

	}

	//template <typename Real, typename Coord>
	//__device__ Real ComputeSurface_density(Coord cell, Coord points, Real mass, SpikyKernel<Real> )
	//{
	//	return 0.0f;
	//}


	template < typename Coord>
	__device__ Coord CSur_getCellPosition(int3 index,  Coord origin, Coord h)
	{
		return Coord(
			index.x * h[0] + origin[0],
			index.y * h[1] + origin[1],
			index.z * h[2] + origin[2]
		);
	}

	template < typename Coord>
	__device__ int3 CSur_getCellIndex(Coord pos, Coord origin, Coord h)
	{
		int3 index;
		/*
		*	2.5 h : 2.0 h + 0.5 h;
		*			2.0 h : the expanded cells; 
		*			0.5 h : the offset of cell position.
		*/
		//index.x = (int)((pos - origin + 0.5 * h[0])[0] / h[0]);
		//index.y = (int)((pos - origin + 0.5 * h[0])[1] / h[1]);
		//index.z = (int)((pos - origin + 0.5 * h[0])[2] / h[2]);
		index.x = (int)((pos - origin )[0] / h[0]);
		index.y = (int)((pos - origin )[1] / h[1]);
		index.z = (int)((pos - origin )[2] / h[2]);
		return index;
	}


	template < typename Real>
	__device__ Real CSur_PhiByDensity(Real V,  Real weight)
	{
		return -V * weight;
	}



	template <typename Real, typename Coord>
	__global__ void ComputeSurface_PointToLevelset(
		DArray3D<Real> distance,
		DArray<Coord> point,
		Coord cell_dx,
		Coord origin,
		SpikyKernel<Real> kern,
		int3 neiborGridNum,
		Real smoothingLength,
		Real mass
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= point.size()) return;

		Coord& pos_i = point[pId];

		int3 index = CSur_getCellIndex(pos_i, origin, cell_dx);
		//Coord cellpos = CSur_getCellPosition(index, origin, h);
		Coord cellpos(0.0f);
		Real phi = 0.0f;
		/*
		*	
		*/
	

		for(int i = -neiborGridNum.x; i <= neiborGridNum.x; i ++)
			for (int j = -neiborGridNum.y; j <= neiborGridNum.y; j++)
				for (int k = -neiborGridNum.z; k <= neiborGridNum.z; k++)
				{
					int3 index_offset;
					index_offset.x = index.x + i;
					index_offset.y = index.y + j;
					index_offset.z = index.z + k;
					
					cellpos = CSur_getCellPosition(index_offset, origin, cell_dx);
					Real r = (cellpos - pos_i).norm();
					Real w = kern.Weight(r, smoothingLength);

					atomicAdd(&distance(index_offset.x, index_offset.y, index_offset.z),
						CSur_PhiByDensity(1.0f, w));


				}

	
		//Real value = distance(index.x, index.y, index.z);
		//if (value < -10000.0f)
		////if (pId < 10)
		//	printf("  %f \r\n", value);

	}


	template <typename Real, typename Coord>
	__global__ void ComputeSurface_Smoothing(
		DArray3D<Real> distance,
		DArray<Coord> point,
		Coord cell_dx,
		Coord origin,
		Real smoothingLength,
		SpikyKernel<Real> kern
	)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		Real grid_ijk = distance(i, j, k) ;
		Real grid_neighbor = 0.0f;


	}


	template <typename Real>
	__global__ void ComputeSurface_LevelsetInitial(
		DArray3D<Real> distance,
		Real value
	) {
		int dId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (dId >= distance.nx() * distance.ny() * distance.nz()) return;


	/*	if (distance[dId]> 10* EPSILON)
			printf("distance %f \r\n",distance[dId]);*/

		distance[dId] = value;
		for (int i = -1; i < 2; i++)
			for (int j = -1; j < 2; j++)
				for (int k = -1; k < 2; k++)
				{
					//int3 index_offset;
					//index_offset.x = index.x + i;
					//index_offset.y = index.y + j;
					//index_offset.z = index.z + k;

					//cellpos = CSur_getCellPosition(index, origin, cell_dx);
					//Real r = (cellpos - pos_i).norm();
					//Real w = kern.Weight(r, smoothingLength);
				}
	}

	template <typename Real>
	__global__ void ComputeSurface_LevelsetTest(
		DArray3D<Real> distance,
		Real value
	) {
		int dId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (dId >= distance.nx() * distance.ny() * distance.nz()) return;


		//if (distance[dId] < -100* EPSILON)
		//	printf("%f \r\n",distance[dId]);
	}

	template<typename TDataType>
	void ComputeSurfaceLevelset<TDataType>::constrain()
	{
		std::cout << "ComputeSurfaceLevelset: " << this->inPoints()->size() <<std::endl;

		int num = this->inPoints()->size();
		auto& sdf = this->inLevelSet()->getDataPtr()->getSDF();

		auto& distances = this->inLevelSet()->getDataPtr()->getSDF().getMDistance();
		Coord cell_dx = this->inLevelSet()->getDataPtr()->getSDF().getH();
		Coord origin = this->inLevelSet()->getDataPtr()->getSDF().lowerBound();

		//cuExecute(distances.nx() * distances.ny() * distances.nz(),
		//	ComputeSurface_LevelsetInitial,
		//	distances,
		//	0.0f);

		distances.reset();

		Real smoothingLength = (Real)(2.5f * cell_dx.norm());

		int3 neiborGridNumber = make_int3(
			(int)(smoothingLength / cell_dx[0]),
			(int)(smoothingLength / cell_dx[1]),
			(int)(smoothingLength / cell_dx[2])
		);
		std::cout << neiborGridNumber.x << ", " << neiborGridNumber.y << ", " << neiborGridNumber.z << ", "  << std::endl;

		cuExecute(num, ComputeSurface_PointToLevelset,
			distances,
			this->inPoints()->getData(),
			cell_dx,
			origin,
			m_kernel,
			neiborGridNumber,
			smoothingLength,
			1.0f);

		//cuExecute3D(make_uint3(distances.nx(), distances.ny(), distances.nz()),
		//	ComputeSurface_Smoothing,
		//	distances,
		//	this->inPoints()->getData(),
		//	h,
		//	origin,
		//	(Real)(1.5f * h.norm()),
		//	m_kernel
		//)


		//cuExecute(distances.nx() * distances.ny() * distances.nz(),
		//	ComputeSurface_LevelsetTest,
		//	distances,
		//	0.0f);
		
	}

	DEFINE_CLASS(ComputeSurfaceLevelset);
}