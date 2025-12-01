#include "AdaptiveSWE.h"
#include "SceneGraph.h"
#include "HeightField/Module/NumericalScheme.h"
#include "AdaptiveCapillaryWaveHelper.h"

namespace dyno
{
	IMPLEMENT_TCLASS(AdaptiveSWE, TDataType)

		template<typename TDataType>
	AdaptiveSWE<TDataType>::AdaptiveSWE()
		: AdaptiveVolumeFromBBox2D<TDataType>()
	{
	}

	template<typename TDataType>
	AdaptiveSWE<TDataType>::~AdaptiveSWE()
	{
	}

	template <typename Coord4D>
	__global__ void SWE_InitHeights(
		DArray<Coord4D> heights,
		DArray<Vec2f> pos)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < heights.size())
		{
			Vec2f pi = pos[i];
			float h = 8.0f;

			//if ((pi.x - 21.0) * (pi.x - 21.0) + pi.y * pi.y < 0.5)
			//{
			//	h = 1.5;
			//}
			
			float u = 0.0f;
			float v = 0.0f;
			heights[i] = Coord4D(h, h * u, h * v, 0.0f);
		}
	}

	template<typename TDataType>
	void AdaptiveSWE<TDataType>::resetStates()
	{
		AdaptiveVolumeFromBBox2D<TDataType>::resetStates();

		auto grid = this->stateAGridSet()->constDataPtr();

		DArray<Vec2f> pos;
		DArrayList<int> neighbors;

		grid->extractLeafs(pos, neighbors);

		this->stateGrid()->resize(pos.size());
		mDeviceGridNext.resize(pos.size());

		cuExecute(pos.size(),
			SWE_InitHeights,
			this->stateGrid()->constData(),
			pos);

		cuExecute(pos.size(),
			SWE_InitHeights,
			mDeviceGridNext,
			pos);

		pos.clear();
		neighbors.clear();
	}
	
	template <typename Coord4D, typename Real>
	__global__ void SWE_InterpolateFromPast(
		DArray<Coord4D> grid,
		DArray<Vec2f> pos,
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
		Vec2f pos_offset = pos[tId] - leaves_old[oldind].m_position;
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

	template <typename Coord4D, typename Real>
	__device__ bool SWE_ApplyOneBoat(
		Coord4D& water,
		Vec2f pos,
		Vec2f center,
		Vec2f boxsize,
		Real angle,
		Real radius,
		Real linear)
	{
		Vec2f center_new;
		center_new[0] = center[0] * (std::cos(angle)) - center[1] * (std::sin(angle)) + radius * (std::cos(angle));
		center_new[1] = center[0] * (std::sin(angle)) + center[1] * (std::cos(angle)) + radius * (std::sin(angle));
		Vec2f normal1_new = Vec2f(1.0, 0.0);
		normal1_new[0] = std::cos(angle);
		normal1_new[1] = std::sin(angle);
		Vec2f normal2_new = Vec2f(0.0, 1.0);
		normal2_new[0] = -std::sin(angle);
		normal2_new[1] = std::cos(angle);
		Vec2f posvec = pos - center_new;
		if ((abs(posvec.dot(normal1_new)) < 0.9 * (boxsize[0] / 2)) && ((posvec.dot(normal2_new) < 0.6 * (boxsize[1] / 2)) && (posvec.dot(normal2_new) > -(boxsize[1] / 2))))
		{
			//Real coef = 5.0 - 5 * abs(posvec.dot(normal1_new)) / (0.8 * (boxsize[0] / 2));
			Vec2f linear_velocity = linear * normal2_new;
			water[1] += 0.5 * linear_velocity[0];
			water[2] += 0.5 * linear_velocity[1];
			//printf("????AdaptiveSWE %f %f %f %f, %f %f \n", water.x, water.y, water.z, water.w, 0.4 * linear_velocity[0], 0.4 * linear_velocity[1]);

			return true;
		}
		return false;
	}

	template <typename Coord4D, typename Real>
	__global__ void SWE_ApplySpeedToWater(
		DArray<Coord4D> grid,
		DArray<Vec2f> grid_pos,
		Vec2f center,
		Vec2f boxsize,
		Real angle1,
		Real angle2,
		Real angle3,
		Real angle4,
		Real angle5,
		Real angle6,
		Real linearv1,
		Real linearv2)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= grid.size()) return;

		if (SWE_ApplyOneBoat(grid[tId], grid_pos[tId], center, boxsize, angle1, 42.0f, linearv1)) return;
		if (SWE_ApplyOneBoat(grid[tId], grid_pos[tId], center, boxsize, angle2, 42.0f, linearv1)) return;
		if (SWE_ApplyOneBoat(grid[tId], grid_pos[tId], center, boxsize, angle3, 42.0f, linearv1)) return;
		if (SWE_ApplyOneBoat(grid[tId], grid_pos[tId], center, boxsize, angle4, 21.0f, linearv2)) return;
		if (SWE_ApplyOneBoat(grid[tId], grid_pos[tId], center, boxsize, angle5, 21.0f, linearv2)) return;
		if (SWE_ApplyOneBoat(grid[tId], grid_pos[tId], center, boxsize, angle6, 21.0f, linearv2)) return;
	}

	template<typename TDataType>
	void AdaptiveSWE<TDataType>::updateStates()
	{
		auto scn = this->getSceneGraph();
		auto GRAVITY = scn->getGravity().norm();
		Real dt = this->stateTimeStep()->getValue();

		AdaptiveGridSet2D<TDataType>* AGrid0 = new AdaptiveGridSet2D<TDataType>;
		auto grid = this->stateAGridSet()->constDataPtr();
		AGrid0->setLevelNum(grid->getLevelNum());
		AGrid0->setLevelMax(grid->getLevelMax());
		AGrid0->setDx(grid->getDx());
		AGrid0->setOrigin(grid->getOrigin());
		AGrid0->setQuadType(grid->getQuadType());
		AGrid0->setAGrids(grid->getAGrids());
		AGrid0->update();
		DArray<AdaptiveGridNode2D> leaves_old;
		DArrayList<int> neighbors_old;
		AGrid0->extractLeafs(leaves_old, neighbors_old);

		AdaptiveVolumeFromBBox2D<TDataType>::updateStates();
		printf("AdaptiveSWE: %d %d \n", (AGrid0->getAGrids()).size(), (grid->getAGrids()).size());

		DArray<Vec2f> pos;
		DArray<AdaptiveGridNode2D> leaves;
		DArrayList<int> neighbors;
		grid->extractLeafs(pos, leaves, neighbors);

		DArray<int> index;
		AGrid0->accessRandom(index, pos);

		auto& data = this->stateGrid()->getData();
		mDeviceGridNext.resize(leaves.size());
		cuExecute(leaves.size(),
			SWE_InterpolateFromPast,
			mDeviceGridNext,
			pos,
			index,
			data,
			leaves_old,
			neighbors_old,
			AGrid0->getLevelMax(),
			AGrid0->getDx());
		leaves_old.clear();
		neighbors_old.clear();
		index.clear();
		AGrid0->clear();
		delete AGrid0;

		uint m_frame = this->stateFrameNumber()->getValue();
		auto m_freq1 = this->varFrequency1()->getData();
		auto m_freq2 = this->varFrequency2()->getData();
		Real m_angle1 = m_frame * 3.14 / m_freq1;
		Real m_angle2 = 2 * 3.14 / 3 + m_frame * 3.14 / m_freq1;
		Real m_angle3 = 4 * 3.14 / 3 + m_frame * 3.14 / m_freq1;
		Real m_angle4 = m_frame * 3.14 / m_freq2;
		Real m_angle5 = 2 * 3.14 / 3 + m_frame * 3.14 / m_freq2;
		Real m_angle6 = 4 * 3.14 / 3 + m_frame * 3.14 / m_freq2;
		Real linear_v1 = 42.0 * 3.14 / m_freq1;
		Real linear_v2 = 21.0 * 3.14 / m_freq2;
		cuExecute(leaves.size(),
			SWE_ApplySpeedToWater,
			mDeviceGridNext,
			pos,
			BBoxCenter,
			BBoxSize,
			m_angle1,
			m_angle2,
			m_angle3,
			m_angle4,
			m_angle5,
			m_angle6,
			linear_v1,
			linear_v2);
		pos.clear();

//		mDeviceGridNext.assign(data);

		data.resize(leaves.size());
		AdaptiveCapillaryWaveHelper<TDataType>::ACWHelper_OneWaveStepVersion1(
			data,
			mDeviceGridNext,
			leaves,
			neighbors,
			grid->getLevelMax(),
			GRAVITY,
			dt);

		leaves.clear();
		neighbors.clear();
	}

	DEFINE_CLASS(AdaptiveSWE);
}