#include "NeighborPointQuery.h"

#include "VkDeviceArray3D.h"

namespace dyno
{
	IMPLEMENT_CLASS(NeighborPointQuery)

	struct HashGrid
	{
		Vec3f lo;
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
				UNIFORM(ArrayListInfo),		//
				UNIFORM(uint))				//particle num
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
				UNIFORM(ArrayListInfo),		//hash grid array list
				UNIFORM(uint))				//particle num
		);
		kernel("CountNeighbors")->load(getAssetPath() + "shaders/glsl/particlesystem/CountNeighbors.comp.spv");

		this->addKernel(
			"SetupNeighbors",
			std::make_shared<VkProgram>(
				BUFFER(uint),				//elements of neighbor lists
				BUFFER(uint),				//radix of neighbor lists
				BUFFER(uint),				//count
				BUFFER(uint),				//elements
				BUFFER(uint),				//radix
				BUFFER(Vec3f),				//position
				UNIFORM(HashGrid),			//hash grid structure
				UNIFORM(ArrayListInfo),		//neighbor list
				UNIFORM(ArrayListInfo),		//hash grid array list
				UNIFORM(uint))				//particle num
		);
		kernel("SetupNeighbors")->load(getAssetPath() + "shaders/glsl/particlesystem/SetupNeighbors.comp.spv");
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

		uint pNum = points.size();

		VkConstant<uint> constNum(pNum);

// 		Vec3f lo = Vec3f(-0.5f, 0.0f, -0.5f);
// 		Vec3f hi = Vec3f(0.5f, 1.0f, 0.5f);
		Vec3f lo = Vec3f(-1.0f, 3.0f, -1.0f);
		Vec3f hi = Vec3f(1.0f, 5.0f, 1.0f);

		uint nx = (hi.x - lo.x) / h;
		uint ny = (hi.y - lo.y) / h;
		uint nz = (hi.z - lo.z) / h;

		uint gNum = nx * ny * nz;

		DArray<uint> counter(gNum);

		kernel("ResetUInt")->flush(
			vkDispatchSize(gNum, 64),
			counter.handle(),
			&VkConstant<uint>(gNum));

		HashGrid grid;
		grid.lo = lo;
		grid.nx = nx;
		grid.ny = ny;
		grid.nz = nz;
		grid.total = nx * ny * nz;
		grid.h = h;

		VkUniform<HashGrid> uniGrid;
		uniGrid.setValue(grid);

		VkUniform<uint> pUniform;
		pUniform.setValue(pNum);

		kernel("CountParticles")->flush(
			vkDispatchSize(pNum, 64),
			counter.handle(),
			points.handle(),
			&uniGrid,
			&pUniform);

		DArrayList<uint> cellIds;

		cellIds.resize(*counter.handle());

		kernel("ResetUInt")->flush(
			vkDispatchSize(gNum, 64),
			counter.handle(),
			&VkConstant<uint>(gNum));

		kernel("SetupParticles")->flush(
			vkDispatchSize(pNum, 64),
			cellIds.mElements.handle(),
			cellIds.mIndex.handle(),
			counter.handle(),
			points.handle(),
			&uniGrid,
			&cellIds.mInfo,
			&pUniform);

// 		CArray<uint> hElements;
// 		hElements.assign(cellIds.mIndex);
// 
// 		for (int i = 0; i < hElements.size() - 1; i++)
// 		{
// 			if(hElements[i] != hElements[i + 1])
// 				printf("%d %u %u \n", i, hElements[i], hElements[i + 1]);
// 		}

// 		CArray<uint> hElements;
// 		hElements.assign(cellIds.mElements);
// 
// 		for (int i = 0; i < hElements.size(); i++)
// 		{
// 			printf("%d %u \n", i, hElements[i]);
// 		}

		DArray<uint> nbrCount(pNum);

		kernel("ResetUInt")->flush(
			vkDispatchSize(pNum, 64),
			nbrCount.handle(),
			&constNum);

		kernel("CountNeighbors")->flush(
			vkDispatchSize(pNum, 64),
			nbrCount.handle(),
			cellIds.mElements.handle(),
			cellIds.mIndex.handle(),
			points.handle(),
			&uniGrid,
			&cellIds.mInfo,
			&pUniform);

		nbrIds.resize(*nbrCount.handle());

		kernel("ResetUInt")->flush(
			vkDispatchSize(pNum, 64),
			nbrCount.handle(),
			&constNum);

		kernel("SetupNeighbors")->flush(
			vkDispatchSize(pNum, 64),
			nbrIds.mElements.handle(),
			nbrIds.mIndex.handle(),
			nbrCount.handle(),
			cellIds.mElements.handle(),
			cellIds.mIndex.handle(),
			points.handle(),
			&uniGrid,
			&nbrIds.mInfo,
			&cellIds.mInfo,
			&pUniform);

// 		CArrayList<uint> hNeighbors;
// 		hNeighbors.assign(nbrIds);

// 		for (int i = 0; i < hNeighbors.size(); i++)
// 		{
// 			auto& list = hNeighbors[i];
// 			std::cout << i << " ";
// 			for (int j = 0; j < list.size(); j++)
// 			{
// 				std::cout << list[j] << " ";
// 			}
// 			std::cout << std::endl;
// 		}

		nbrCount.clear();
		counter.clear();
		cellIds.clear();
	}
}