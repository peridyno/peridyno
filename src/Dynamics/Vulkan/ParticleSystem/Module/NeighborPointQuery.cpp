#include "NeighborPointQuery.h"

#include "VkDeviceArray3D.h"

namespace dyno
{
	IMPLEMENT_CLASS(NeighborPointQuery)

	struct HashGrid
	{
		uint nx;
		uint ny;
		uint nz;
		uint total;

		float h;
	};

	NeighborPointQuery::NeighborPointQuery()
		: ComputeModule()
	{
		this->inOther()->tagOptional(true);

		this->addKernel(
			"ResetUInt",
			std::make_shared<VkProgram>(
				BUFFER(uint),				//array
				CONSTANT(uint))				//num
		);
		kernel("ResetUInt")->load(getAssetPath() + "shaders/glsl/math/ResetUInt.comp.spv");

		this->addKernel(
			"CountParticles",
			std::make_shared<VkProgram>(
				BUFFER(uint),				//count
				BUFFER(Vec3f),				//position
				UNIFORM(HashGrid),			//hash grid structure
				UNIFORM(uint))				//num
		);
		kernel("CountParticles")->load(getAssetPath() + "shaders/glsl/particlesystem/CountParticles.comp.spv");

		this->addKernel(
			"SetupParticles",
			std::make_shared<VkProgram>(
				BUFFER(uint),				//ids
				BUFFER(uint),				//radix
				BUFFER(uint),				//count
				BUFFER(Vec3f),				//position
				UNIFORM(HashGrid),			//hash grid structure
				UNIFORM(ArrayListInfo))				//num
		);
		kernel("SetupParticles")->load(getAssetPath() + "shaders/glsl/particlesystem/SetupParticles.comp.spv");

		this->addKernel(
			"CountNeighbors",
			std::make_shared<VkProgram>(
				BUFFER(uint),				//count
				BUFFER(uint),				//elements
				BUFFER(uint),				//radix
				BUFFER(Vec3f),				//position
				UNIFORM(HashGrid),			//hash grid structure
				UNIFORM(ArrayListInfo))				//num
		);
		kernel("CountNeighbors")->load(getAssetPath() + "shaders/glsl/particlesystem/CountNeighbors.comp.spv");
	}

	NeighborPointQuery::~NeighborPointQuery()
	{
	}

	void NeighborPointQuery::compute()
	{
		requestDynamicNeighborIds();
	}

	void NeighborPointQuery::requestDynamicNeighborIds()
	{
		// Prepare inputs
		auto& points	= this->inPosition()->getData();
		auto& other		= this->inOther()->isEmpty() ? this->inPosition()->getData() : this->inOther()->getData();
		auto h			= this->inRadius()->getValue();

		// Prepare outputs
		if (this->outNeighborIds()->isEmpty())
			this->outNeighborIds()->allocate();

		auto& nbrIds = this->outNeighborIds()->getData();

		uint num = points.size();

		VkConstant<uint> constNum(num);

		Vec3f lo = Vec3f(-0.5f, 0.0f, -0.5f);
		Vec3f hi = Vec3f(0.5f, 1.0f, 0.5f);

		uint nx = (hi.x - lo.x) / h;
		uint ny = (hi.y - lo.y) / h;
		uint nz = (hi.z - lo.z) / h;

		DArray<uint> counter(nx * ny * nz);

		kernel("ResetUInt")->flush(
			vkDispatchSize(num, 64),
			counter.handle(),
			&VkConstant<uint>(nx * ny * nz));

		HashGrid grid;
		grid.nx = nx;
		grid.ny = ny;
		grid.nz = nz;
		grid.total = nx * ny * nz;
		grid.h = h;

		VkUniform<HashGrid> uniGrid;
		uniGrid.setValue(grid);

		VkUniform<uint> uniNum;
		uniNum.setValue(num);

		kernel("CountParticles")->flush(
			vkDispatchSize(num, 64),
			counter.handle(),
			points.handle(),
			&uniGrid,
			&uniNum);

		DArrayList<uint> cellIds;

		cellIds.resize(*counter.handle());

// 		kernel("ResetUInt")->flush(
// 			vkDispatchSize(num, 64),
// 			counter.handle(),
// 			&VkConstant<uint>(nx * ny * nz));
// 
// 		kernel("SetupParticles")->flush(
// 			vkDispatchSize(num, 64),
// 			cellIds.mElements.handle(),
// 			cellIds.mIndex.handle(),
// 			counter.handle(),
// 			points.handle(),
// 			&uniGrid,
// 			&cellIds.mInfo);

		CArray<uint> hElements;
		hElements.assign(cellIds.mIndex);

		for (int i = 0; i < hElements.size(); i++)
		{
			if(hElements[i] > 0)
				printf("%d %u \n", i, hElements[i]);
		}

		uint dd = hElements[hElements.size() - 1];

		DArray<uint> nbrCount(num);

		kernel("ResetUInt")->flush(
			vkDispatchSize(num, 64),
			nbrCount.handle(),
			&constNum);

		kernel("CountNeighbors")->flush(
			vkDispatchSize(num, 64),
			nbrCount.handle(),
			cellIds.mElements.handle(),
			cellIds.mIndex.handle(),
			points.handle(),
			&uniGrid,
			&cellIds.mInfo);

		nbrIds.resize(*nbrCount.handle());

// 
// 		printf("%u \n", eleNum);

// 		GridHash<TDataType> hashGrid;
// 		hashGrid.setSpace(h, loBound - Coord(h), hiBound + Coord(h));
// 		hashGrid.clear();
// 		hashGrid.construct(points);
// 
// 		DArray<uint> counter(other.size());
// 		cuExecute(other.size(),
// 			K_CalNeighborSize,
// 			counter,
// 			other,
// 			points, 
// 			hashGrid, 
// 			h);
// 
// 		nbrIds.resize(counter);
// 
// 		cuExecute(other.size(),
// 			K_GetNeighborElements,
// 			nbrIds, 
// 			other,
// 			points, 
// 			hashGrid,
// 			h);
// 
// 		counter.clear();
// 		hashGrid.release();
		counter.clear();
		cellIds.clear();
	}
}