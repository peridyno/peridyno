#include <cuda_runtime.h>
#include "AdaptiveCapillaryWave.h"
#include "SceneGraph.h"
#include "Module/NumericalScheme.h"
#include "AdaptiveCapillaryWaveHelper.h"

namespace dyno
{
	IMPLEMENT_TCLASS(AdaptiveCapillaryWave, TDataType)

	template<typename TDataType>
	AdaptiveCapillaryWave<TDataType>::AdaptiveCapillaryWave()
		: Node()
	{
	}

	template<typename TDataType>
	AdaptiveCapillaryWave<TDataType>::~AdaptiveCapillaryWave()
	{
		mDeviceGridNext.clear();
	}

	template <typename Real, typename Coord4D>
	__global__ void ACW_InitHeights(
		DArray<Coord4D> heights,
		Real level)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < heights.size())
		{
			Real h = level;

			Real u = 0.0f;
			Real v = 0.0f;
			heights[i] = Coord4D(h, h * u, h * v, 0.0f);
		}
	}

	template<typename TDataType>
	void AdaptiveCapillaryWave<TDataType>::resetStates()
	{
		auto grid = this->inAGrid2D()->constDataPtr();

		DArray<Coord2D> pos;
		DArrayList<int> neighbors;
		grid->extractLeafs(pos, neighbors);

		this->stateHeigh()->resize(pos.size());
		mDeviceGridNext.resize(pos.size());

		cuExecute(pos.size(),
			ACW_InitHeights,
			this->stateHeigh()->getData(),
			this->varWaterLevel()->getData());

		cuExecute(pos.size(),
			ACW_InitHeights,
			mDeviceGridNext,
			this->varWaterLevel()->getData());

		pos.clear();
		neighbors.clear();

	}

	template<typename TDataType>
	void AdaptiveCapillaryWave<TDataType>::updateStates()
	{
		auto scn = this->getSceneGraph();
		auto GRAVITY = scn->getGravity().norm();
		Real dt = this->stateTimeStep()->getValue();

		auto grid = this->inAGrid2D()->constDataPtr();
		DArray<AdaptiveGridNode2D> leaves;
		DArrayList<int> neighbors;
		grid->extractLeafs(leaves, neighbors);

		auto& data = this->stateHeigh()->getData();
		data.resize(leaves.size());
		data.reset();
		AdaptiveCapillaryWaveHelper<TDataType>::ACWHelper_OneWaveStepVersion1(
			data,
			mDeviceGridNext,
			leaves,
			neighbors,
			grid->adaptiveGridLevelMax2D(),
			GRAVITY,
			dt);

		leaves.clear();
		neighbors.clear();
	}

	DEFINE_CLASS(AdaptiveCapillaryWave);
}