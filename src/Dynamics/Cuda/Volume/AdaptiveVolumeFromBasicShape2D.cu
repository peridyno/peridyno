#include "AdaptiveVolumeFromBasicShape2D.h"
#include "STL/Stack.h"
#include <thrust/sort.h>

#include "BasicShapes/CircleModel2D.h"
#include "BasicShapes/RectangleModel2D.h"

namespace dyno
{
	IMPLEMENT_TCLASS(AdaptiveVolumeFromBasicShape2D, TDataType)

	template<typename TDataType>
	AdaptiveVolumeFromBasicShape2D<TDataType>::AdaptiveVolumeFromBasicShape2D()
		: AdaptiveVolume2D<TDataType>()
	{
		mMSTGen = std::make_shared<MSTsGenerator2D<TDataType>>();
		this->stateAGridSet()->promoteOuput()->connect(mMSTGen->inAGridSet());
		this->stateIncreaseMorton()->connect(mMSTGen->inpMorton());
		this->stateFrameNumber()->connect(mMSTGen->inFrameNumber());
		this->varLevelNum()->connect(mMSTGen->varLevelNum());
		mMSTGen->varQuadType()->setCurrentKey(AdaptiveGridGenerator2D<DataType2f>::EDGE_BALANCED);

		mMSTGenLocal = std::make_shared<MSTsGeneratorLocalUpdate2D<TDataType>>();
		this->stateAGridSet()->promoteOuput()->connect(mMSTGenLocal->inAGridSet());
		this->stateIncreaseMorton()->connect(mMSTGenLocal->inpMorton());
		this->stateDecreaseMorton()->connect(mMSTGenLocal->inDecreaseMorton());
		this->stateFrameNumber()->connect(mMSTGenLocal->inFrameNumber());
		this->varLevelNum()->connect(mMSTGenLocal->varLevelNum());
		mMSTGenLocal->varQuadType()->setCurrentKey(AdaptiveGridGenerator2D<DataType2f>::EDGE_BALANCED);
	}

	template<typename TDataType>
	AdaptiveVolumeFromBasicShape2D<TDataType>::~AdaptiveVolumeFromBasicShape2D()
	{
	}

	template<typename TDataType>
	void AdaptiveVolumeFromBasicShape2D<TDataType>::initParameter()
	{
		m_origin = this->varLowerBound()->getValue();
		Coord2D m_max = this->varUpperBound()->getValue(); ;

		TAlignedBox2D<Real> aabb_box(m_origin, m_max);
		auto shapes = this->getShapes();
		for (int i = 0; i < shapes.size(); i++)
		{
			BasicShapeType2D type = shapes[i]->getShapeType();
			if (type == BasicShapeType2D::CIRCLE)
			{
				auto circleModel = dynamic_cast<CircleModel2D<TDataType>*>(shapes[i]);

				auto box = circleModel->outCircle()->getData();
				auto aabb = box.aabb();
				printf("circle aabb %f %f , %f; \n", box.center[0], box.center[1], box.radius);
				aabb_box = aabb_box.merge(aabb);
			}
			else if (type == BasicShapeType2D::RECTANGLE)
			{
				auto rectangleModel = dynamic_cast<RectangleModel2D<TDataType>*>(shapes[i]);

				auto box = rectangleModel->outRectangle()->getData();
				auto aabb = box.aabb();
				aabb_box = aabb_box.merge(aabb);
			}
		}
		m_origin = aabb_box.v0;
		m_max = aabb_box.v1;

		Real m_dx = this->varDx()->getValue();
		int rs = std::floor(std::max(m_max[0] - m_origin[0], m_max[1] - m_origin[1]) / m_dx);
		m_levelmax = std::ceil(std::log2(float(rs)));
		m_levelmax = std::max(m_levelmax, this->varMaxLevel()->getValue());

		int rs_max = (1 << m_levelmax);
		m_origin = m_origin - (m_dx * (rs_max - rs) / 2);
		std::printf("The origin, dx, levelmax are: %f  %f,  %f  %f,   %f, %d %d \n", m_origin[0], m_origin[1], m_max[0], m_max[1], m_dx, rs, m_levelmax);

		this->varMaxLevel()->setValue(m_levelmax);
		this->stateAGridSet()->allocate();
		auto m_AGrid = this->stateAGridSet()->getDataPtr();
		m_AGrid->setOrigin(m_origin);
		m_AGrid->setDx(m_dx);
		m_AGrid->setLevelMax(m_levelmax);
	}

	template <typename Real, typename Coord2D>
	__global__ void AVBS2D_CircleNarrowBandCount(
		DArray<uint> count,
		Coord2D center,
		Real radius,
		Coord2D origin,
		Real dx,
		Real band)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		OcIndex nx, ny;
		RecoverFromMortonCode2D(OcKey(tId), nx, ny);
		Coord2D pos(origin[0] + (nx + 0.5) * dx, origin[1] + (ny + 0.5) * dx);

		Real dist = (pos - center).norm();
		if (abs(dist - radius) < band)
			count[tId] = 1;
	}

	__global__ void AVBS2D_NarrowBandCompute(
		DArray<OcKey> nodes,
		DArray<uint> count)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (tId == (count.size() - 1) && count[tId] < nodes.size())
			nodes[count[tId]] = OcKey(tId);
		else if ((tId < (count.size() - 1)) && (count[tId] < count[tId + 1]))
			nodes[count[tId]] = OcKey(tId);
	}

	template <typename Real, typename Coord2D, typename Coord3D>
	__global__ void AVBS2D_ParticlesCount(
		DArray<uint> count,
		DArray<Coord3D> particles,
		Coord2D origin,
		Real dx,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= particles.size()) return;

		Coord3D pos = particles[tId];
		int i = (int)floor((pos[0] - origin[0]) / dx);
		int j = (int)floor((pos[1] - origin[1]) / dx);

		if ((i < 0 || i >= resolution) || (j < 0 || j >= resolution)) return;

		OcKey index = CalculateMortonCode2D((OcIndex)i, (OcIndex)j);
		count[index] = 1;
	}

	template<typename TDataType>
	void AdaptiveVolumeFromBasicShape2D<TDataType>::circleSeeds(DArray<uint>& marker,Coord2D center, Real radius)
	{
		Real m_dx = this->varDx()->getValue();
		Real band = this->varNarrowWidth()->getValue();
		int resolution = (1 << m_levelmax);

		cuExecute(marker.size(),
			AVBS2D_CircleNarrowBandCount,
			marker,
			center,
			radius,
			m_origin,
			m_dx,
			band);

		if (!this->inParticles()->isEmpty())
		{
			auto& particles = this->inParticles()->getData();
			cuExecute(particles.size(),
				AVBS2D_ParticlesCount,
				marker,
				particles,
				m_origin,
				m_dx,
				resolution);
		}
	}

	template <typename Real, typename Coord2D>
	__global__ void AVBS2D_RectangleCount(
		DArray<uint> count,
		TOrientedBox2D<Real> obox,
		Coord2D origin,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		OcIndex nx, ny;
		RecoverFromMortonCode2D(OcKey(tId), nx, ny);
		Coord2D pos(origin[0] + (nx + 0.5) * dx, origin[1] + (ny + 0.5) * dx);
		TPoint2D<Real> tp(pos);

		if(tp.inside(obox)) count[tId] = 1;
	}

	template<typename TDataType>
	void AdaptiveVolumeFromBasicShape2D<TDataType>::rectangleSeeds(DArray<uint>& marker, TOrientedBox2D<Real>& obox)
	{
		Real m_dx = this->varDx()->getValue();

		if (this->varIsHollow()->getData() == false)
		{
			cuExecute(marker.size(),
				AVBS2D_RectangleCount,
				marker,
				obox,
				m_origin,
				m_dx);
		}
		else
		{
			std::printf("Waiting for supplementation \n");
		}
	}

	template<typename TDataType>
	void AdaptiveVolumeFromBasicShape2D<TDataType>::computeSeeds(DArray<uint>& marker)
	{
		auto shapes = this->getShapes();
		for (int i = 0; i < shapes.size(); i++)
		{
			BasicShapeType2D type = shapes[i]->getShapeType();
			if (type == BasicShapeType2D::CIRCLE)
			{
				auto circleModel = dynamic_cast<CircleModel2D<TDataType>*>(shapes[i]);
				Coord2D center = circleModel->varCenter2D()->getValue();
				Real radius = circleModel->varRadius()->getValue();

				circleSeeds(marker, center, radius);
			}
			else if (type == BasicShapeType2D::RECTANGLE)
			{
				auto rectangleModel = dynamic_cast<RectangleModel2D<TDataType>*>(shapes[i]);
				auto box = rectangleModel->outRectangle()->getData();

				rectangleSeeds(marker, box);
			}
		};
	}
	
	template<typename TDataType>
	void AdaptiveVolumeFromBasicShape2D<TDataType>::computeSeedsFromScratch()
	{
		int resolution = (1 << m_levelmax);
		DArray<uint> data_count(resolution * resolution);
		data_count.reset();

		computeSeeds(data_count);

		Reduction<uint> reduce;
		int grid_num = reduce.accumulate(data_count.begin(), data_count.size());
		Scan<uint> scan;
		scan.exclusive(data_count.begin(), data_count.size());

		if (this->stateIncreaseMorton()->isEmpty())
			this->stateIncreaseMorton()->allocate();
		auto& nodes = this->stateIncreaseMorton()->getData();
		nodes.resize(grid_num);
		cuExecute(data_count.size(),
			AVBS2D_NarrowBandCompute,
			nodes,
			data_count);

		data_count.clear();
	}

	__global__ void AVBS2D_CountGridAdaptiveGrid(
		DArray<uint> increase,
		DArray<uint> decrease,
		DArray<AdaptiveGridNode2D> leaves,
		Level lmax,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= leaves.size()) return;

		if (leaves[tId].m_level == lmax)
		{
			if (increase[leaves[tId].m_morton] == 1) increase[leaves[tId].m_morton] = 0;
			else decrease[tId] = 1;
		}
	}

	__global__ void AVBS2D_CountDecrease(
		DArray<uint> mark,
		DArray<uint> decrease,
		DArray<AdaptiveGridNode2D> leaves,
		Level lmax)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= decrease.size()) return;

		if (leaves[tId].m_level == lmax)
		{
			int index = tId - (leaves[tId].m_morton & 3);
			if (decrease[index] == 1 && decrease[index + 1] == 1 && decrease[index + 2] == 1 && decrease[index + 3] == 1)
				mark[index] = 1;
		}
	}

	__global__ void AVBS2D_ComputeDecrease(
		DArray<OcKey> denode,
		DArray<uint> decount,
		int denum,
		DArray<AdaptiveGridNode2D> leaves)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= decount.size()) return;

		if (tId == (decount.size() - 1) && decount[tId] < denum)
		{
			denode[decount[tId]] = leaves[tId].m_morton;
			return;
		}
		else if (tId < (decount.size() - 1) && decount[tId] != decount[tId + 1])
		{
			denode[decount[tId]] = leaves[tId].m_morton;
			return;
		}
	}

	template<typename TDataType>
	void AdaptiveVolumeFromBasicShape2D<TDataType>::computeSeedsDynamicUpdate()
	{
		int resolution = (1 << m_levelmax);
		DArray<uint> data_count(resolution * resolution);
		data_count.reset();

		computeSeeds(data_count);

		auto& m_agrid = this->stateAGridSet()->constDataPtr();
		Level m_levelmax = m_agrid->adaptiveGridLevelMax2D();
		DArray<AdaptiveGridNode2D> leaves;
		m_agrid->extractLeafs(leaves);

		DArray<uint> decrease_temp(leaves.size());
		decrease_temp.reset();
		cuExecute(leaves.size(),
			AVBS2D_CountGridAdaptiveGrid,
			data_count,
			decrease_temp,
			leaves,
			m_levelmax,
			resolution);

		DArray<uint> mark_decrease(leaves.size());
		mark_decrease.reset();
		cuExecute(leaves.size(),
			AVBS2D_CountDecrease,
			mark_decrease,
			decrease_temp,
			leaves,
			m_levelmax);
		decrease_temp.clear();

		Reduction<uint> reduce;
		Scan<uint> scan;
		int increase_num = reduce.accumulate(data_count.begin(), data_count.size());
		scan.exclusive(data_count.begin(), data_count.size());
		int decrease_num = reduce.accumulate(mark_decrease.begin(), mark_decrease.size());
		scan.exclusive(mark_decrease.begin(), mark_decrease.size());

		if (this->stateIncreaseMorton()->isEmpty()) this->stateIncreaseMorton()->allocate();
		auto& innodes = this->stateIncreaseMorton()->getData();
		innodes.resize(increase_num);
		if (this->stateDecreaseMorton()->isEmpty()) this->stateDecreaseMorton()->allocate();
		auto& denodes = this->stateDecreaseMorton()->getData();
		denodes.resize(decrease_num);

		cuExecute(data_count.size(),
			AVBS2D_NarrowBandCompute,
			innodes,
			data_count);

		cuExecute(leaves.size(),
			AVBS2D_ComputeDecrease,
			denodes,
			mark_decrease,
			decrease_num,
			leaves);

		leaves.clear();
		mark_decrease.clear();
		data_count.clear();
	}

	template<typename TDataType>
	void AdaptiveVolumeFromBasicShape2D<TDataType>::resetStates()
	{
		initParameter();
		computeSeedsFromScratch();
		mMSTGen->update();

		AdaptiveVolume2D<TDataType>::resetStates();
	}

	template<typename TDataType>
	void AdaptiveVolumeFromBasicShape2D<TDataType>::updateStates()
	{
		if (this->varDynamicMode()->getData() == true)
		{
			computeSeedsDynamicUpdate();
			mMSTGenLocal->update();
		}
		else
		{
			computeSeedsFromScratch();
			mMSTGen->update();
		}

		AdaptiveVolume2D<TDataType>::updateStates();
	}

	template<typename TDataType>
	bool AdaptiveVolumeFromBasicShape2D<TDataType>::validateInputs()
	{
		auto shapes = this->getShapes();
		if (shapes.size() == 0) return false;
	}

	DEFINE_CLASS(AdaptiveVolumeFromBasicShape2D);
}