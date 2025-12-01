#include "CWAaddBoatSpeed.h"
#include "SceneGraph.h"
#include "HeightField/Module/NumericalScheme.h"
#include "HeightField/AdaptiveCapillaryWaveHelper.h"

namespace dyno
{
	IMPLEMENT_TCLASS(CWAaddBoatSpeed, TDataType)

		template<typename TDataType>
	CWAaddBoatSpeed<TDataType>::CWAaddBoatSpeed()
		: AdaptiveCapillaryWave<TDataType>()
	{
		mAGrid0 = std::make_shared<AdaptiveGridSet2D<TDataType>>();
	}

	template<typename TDataType>
	CWAaddBoatSpeed<TDataType>::~CWAaddBoatSpeed()
	{
	}

	template<typename TDataType>
	void CWAaddBoatSpeed<TDataType>::resetStates()
	{
		AdaptiveCapillaryWave<TDataType>::resetStates();
		maintainAGrid();
	}
	
	template <typename Real, typename Coord2D, typename Coord4D>
	__global__ void CWASpeed_InterpolateFromPast(
		DArray<Coord4D> grid,
		DArray<Coord2D> pos,
		DArray<int> index,
		DArray<Coord4D> grid_old,
		DArray<AdaptiveGridNode2D> leaves_old,
		DArrayList<int> neighbors_old,
		Level levelmax_old,
		Real dx_old)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= grid.size()) return;

		if (index[tId] == EMPTY)
		{
			printf("InterpolateFromPast???? \n");

			grid[tId] = Coord4D(0.0f, 0.0f, 0.0f, 0.0f);
			return;
		}
		int oldind = index[tId];
		Coord2D pos_offset = pos[tId] - leaves_old[oldind].m_position;
		Real up_dx = dx_old * (1 << (levelmax_old - (leaves_old[oldind].m_level)));

		Coord4D gradient_x = Coord4D(0.0f, 0.0f, 0.0f, 0.0f);
		Coord4D gradient_y = Coord4D(0.0f, 0.0f, 0.0f, 0.0f);
		if (pos_offset[0] < 0)
		{
			if (neighbors_old[4 * oldind].size() == 1)
			{
				int nind = neighbors_old[4 * oldind][0];
				if (leaves_old[nind].m_level == leaves_old[oldind].m_level)
					gradient_x = (grid_old[oldind] - grid_old[nind]) / up_dx;
				else
					gradient_x = (grid_old[oldind] - grid_old[nind]) * 2 / 3 / up_dx;
			}
			else if (neighbors_old[4 * oldind].size() == 2)
			{
				int nind1 = neighbors_old[4 * oldind][0];
				int nind2 = neighbors_old[4 * oldind][1];
				gradient_x = (grid_old[oldind] - (grid_old[nind1] + grid_old[nind2]) / 2) * 4 / 3 / up_dx;
			}
		}
		else
		{
			if (neighbors_old[4 * oldind + 1].size() == 1)
			{
				int nind = neighbors_old[4 * oldind + 1][0];
				if (leaves_old[nind].m_level == leaves_old[oldind].m_level)
					gradient_x = (grid_old[nind] - grid_old[oldind]) / up_dx;
				else
					gradient_x = (grid_old[nind] - grid_old[oldind]) * 2 / 3 / up_dx;
			}
			else if (neighbors_old[4 * oldind + 1].size() == 2)
			{
				int nind1 = neighbors_old[4 * oldind + 1][0];
				int nind2 = neighbors_old[4 * oldind + 1][1];
				gradient_x = ((grid_old[nind1] + grid_old[nind2]) / 2 - grid_old[oldind]) * 4 / 3 / up_dx;
			}
		}

		if (pos_offset[1] < 0)
		{
			if (neighbors_old[4 * oldind + 2].size() == 1)
			{
				int nind = neighbors_old[4 * oldind + 2][0];
				if (leaves_old[nind].m_level == leaves_old[oldind].m_level)
					gradient_y = (grid_old[oldind] - grid_old[nind]) / up_dx;
				else
					gradient_y = (grid_old[oldind] - grid_old[nind]) * 2 / 3 / up_dx;
			}
			else if (neighbors_old[4 * oldind + 2].size() == 2)
			{
				int nind1 = neighbors_old[4 * oldind + 2][0];
				int nind2 = neighbors_old[4 * oldind + 2][1];
				gradient_y = (grid_old[oldind] - (grid_old[nind1] + grid_old[nind2]) / 2) * 4 / 3 / up_dx;
			}
		}
		else
		{
			if (neighbors_old[4 * oldind + 3].size() == 1)
			{
				int nind = neighbors_old[4 * oldind + 3][0];
				if (leaves_old[nind].m_level == leaves_old[oldind].m_level)
					gradient_y = (grid_old[nind] - grid_old[oldind]) / up_dx;
				else
					gradient_y = (grid_old[nind] - grid_old[oldind]) * 2 / 3 / up_dx;
			}
			else if (neighbors_old[4 * oldind + 3].size() == 2)
			{
				int nind1 = neighbors_old[4 * oldind + 3][0];
				int nind2 = neighbors_old[4 * oldind + 3][1];
				gradient_y = ((grid_old[nind1] + grid_old[nind2]) / 2 - grid_old[oldind]) * 4 / 3 / up_dx;
			}
		}
	
		grid[tId] = grid_old[oldind] + gradient_x * pos_offset[0] + gradient_y * pos_offset[1];
	}

	template <typename Real, typename Coord2D, typename Coord4D>
	__global__ void CWASpeed_ApplySpeedToWater(
		DArray<Coord4D> grid,
		DArray<Coord2D> grid_pos,
		TOrientedBox2D<Real> obox,
		Real linearv)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= grid.size()) return;

		TPoint2D<Real> tp(grid_pos[tId]);

		if (tp.inside(obox))
		{
			Coord2D linear_velocity = linearv * obox.v;
			grid[tId][1] += 0.5 * linear_velocity[0];
			grid[tId][2] += 0.5 * linear_velocity[1];

		}
	}

	template<typename TDataType>
	void CWAaddBoatSpeed<TDataType>::updateStates()
	{
		auto grid = this->inAGrid2D()->constDataPtr();

		DArray<Coord2D> pos;
		DArray<Real> scale;
		grid->extractLeafs(pos, scale);
		scale.clear();

		DArray<int> index;
		mAGrid0->accessRandom(index, pos);

		DArray<AdaptiveGridNode2D> leaves_old;
		DArrayList<int> neighbors_old;
		mAGrid0->extractLeafs(leaves_old, neighbors_old);

		auto& data = this->stateHeigh()->getData();
		mDeviceGridNext.resize(pos.size());
		mDeviceGridNext.reset();
		cuExecute(pos.size(),
			CWASpeed_InterpolateFromPast,
			mDeviceGridNext,
			pos,
			index,
			data,
			leaves_old,
			neighbors_old,
			mAGrid0->adaptiveGridLevelMax2D(),
			mAGrid0->adaptiveGridDx2D());
		leaves_old.clear();
		neighbors_old.clear();
		index.clear();

		auto shapes = this->getShapes();
		for (int i = 0; i < shapes.size(); i++)
		{
			Quat<Real> q = shapes[i]->computeQuaternion();
			q.normalize();

			Coord3D length = shapes[i]->varLength()->getData();
			Coord3D center = shapes[i]->varLocation()->getData();

			center = q.rotate(Coord3D(center[0] + shapes[i]->varRotationRadius()->getData(), 0.0f, center[2] - length[2] * 0.1f));
			Coord3D u_axis = q.rotate(Coord3D(1, 0, 0));
			Coord3D w_axis = q.rotate(Coord3D(0, 0, 1));

			TOrientedBox2D<Real> retangle;
			retangle.center = Coord2D(center[0], center[2]);
			retangle.u = Coord2D(u_axis[0], u_axis[2]);
			retangle.v = Coord2D(w_axis[0], w_axis[2]);
			retangle.extent = Coord2D(length[0] * 0.9f, length[2] * 0.8f);

			Real linear_v = (shapes[i]->varRotationRadius()->getData()) * 3.14 / (shapes[i]->varFrequency()->getData());
			cuExecute(pos.size(),
				CWASpeed_ApplySpeedToWater,
				mDeviceGridNext,
				pos,
				retangle,
				linear_v);
		}

		pos.clear();
		maintainAGrid();

		AdaptiveCapillaryWave<TDataType>::updateStates();
	}

	template<typename TDataType>
	void CWAaddBoatSpeed<TDataType>::maintainAGrid()
	{
		auto grid = this->inAGrid2D()->constDataPtr();
		mAGrid0->setLevelNum(grid->adaptiveGridLevelNum2D());
		mAGrid0->setLevelMax(grid->adaptiveGridLevelMax2D());
		mAGrid0->setDx(grid->adaptiveGridDx2D());
		mAGrid0->setOrigin(grid->adaptiveGridOrigin2D());
		mAGrid0->setQuadType(grid->adaptiveGridType2D());
		mAGrid0->setAGrids(grid->adaptiveGridNode2D());
		mAGrid0->update();
	}

	DEFINE_CLASS(CWAaddBoatSpeed);
}