#include "TetrahedronSet.h"
#include "VkTransfer.h"

namespace dyno
{
	template<typename TDataType>
	TetrahedronSet<TDataType>::TetrahedronSet(): TriangleSet<TDataType>(), mSort(getSpvFile("shaders/glsl/topology/TetKeySort.comp.spv"))
	{
        this->addKernel("SetupTetKey",
                        std::make_shared<VkProgram>(BUFFER(TKey), BUFFER(Info), BUFFER(Tetrahedron), CONSTANT(uint)));
        this->kernel("SetupTetKey")->load(getSpvFile("shaders/glsl/topology/SetupTetKey.comp.spv"));
        this->addKernel("CountTKey",
                        std::make_shared<VkProgram>(BUFFER(int), BUFFER(TKey), CONSTANT(uint)));
        this->kernel("CountTKey")->load(getSpvFile("shaders/glsl/topology/CountTKey.comp.spv"));
        this->addKernel("SetupTriangle",
                        std::make_shared<VkProgram>(BUFFER(Triangle), BUFFER(Tri2Tet), BUFFER(Tri2Tet), BUFFER(TKey), BUFFER(int), BUFFER(Info), CONSTANT(uint)));
        this->kernel("SetupTriangle")->load(getSpvFile("shaders/glsl/topology/SetupTriangle.comp.spv"));
	}

	template<typename TDataType>
	TetrahedronSet<TDataType>::~TetrahedronSet()
	{

	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(std::vector<Tetrahedron>& indices)
	{
		mTethedrons.assign(indices);
		this->updateTriangles();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(DArray<Tetrahedron>& tetrahedrons) {
		mTethedrons.assign(tetrahedrons);
		this->updateTriangles();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::copyFrom(TetrahedronSet& es)
	{
		this->mTethedrons.assign(es.mTethedrons);
		this->mTriangleIndex.assign(es.mTriangleIndex);
		this->mIndex.assign(es.mIndex);
		this->mEdgeIndex.assign(es.mEdgeIndex);
		this->mCoords.assign(es.mCoords);
	}

	template<typename TDataType>
	DArrayList<int>& TetrahedronSet<TDataType>::getVer2Tet()
	{
        DArray<uint> counter(this->mCoords.size());
        VkConstant<uint> num {mTethedrons.size()};
        VkConstant<uint> shape_size {sizeof(Tetrahedron)};

        counter.reset();
        this->kernel("CountShape")->flush(vkDispatchSize(mTethedrons.size(), 64), counter.handle(), mTethedrons.handle(), &num, &shape_size);
        mVer2Tet.resize(counter);

        VkConstant<VkDeviceAddress> vec2QuadAddr {mVer2Tet.lists().handle()->bufferAddress()};
        counter.reset();
        this->kernel("SetupShapeId")->flush(vkDispatchSize(mTethedrons.size(), 64), mTethedrons.handle(), &num, &shape_size, &vec2QuadAddr);
        counter.clear();

		return mVer2Tet;
	}


	template<typename TDataType>
	void TetrahedronSet<TDataType>::updateTriangles() {
		uint tetSize = mTethedrons.size();

		DArray<TKey> keys;
		DArray<Info> tetIds;

		keys.resize(4 * tetSize);
		tetIds.resize(4 * tetSize);
		VkConstant<uint> vk_num {tetSize};


		this->kernel("SetupTetKey")->flush(vkDispatchSize(tetSize, 64), keys.handle(), tetIds.handle(), mTethedrons.handle(), &vk_num);
		mSort.sortByKey(keys, tetIds, SortParam::eUp);

		DArray<int> counter;
		counter.resize(4 * tetSize);
		vk_num.setValue(keys.size());
        this->kernel("CountTKey")->flush(vkDispatchSize(keys.size(), 64), counter.handle(), keys.handle(), &vk_num);

		int triNum = this->reduce().reduce(*counter.handle());;
		this->scan().scan(*counter.handle(), *counter.handle(), VkScan<int>::Type::Exclusive);

		mTri2Tet.resize(triNum);
		mTri2Tetorder.resize(triNum);

		auto& pTri = this->getTriangles();
		pTri.resize(triNum);

		this->kernel("SetupTriangle")->flush(vkDispatchSize(keys.size(), 64), pTri.handle(), mTri2Tet.handle(), mTri2Tetorder.handle(), keys.handle(), counter.handle(), tetIds.handle(), &vk_num);
		
		counter.clear();
		tetIds.clear();
		keys.clear();

		this->updateEdges();
	}

	/*
	template<typename TDataType>
	void TetrahedronSet<TDataType>::updateTopology()
	{
		std::vector<dyno::TopologyModule::Tetrahedron> tets(mTethedrons.size());
		std::vector<uint32_t> tris;

		std::vector<dyno::TopologyModule::Triangle> triangles;

		//TODO: atomic operations are not supported yet, replace the following implementation with a parallel algorithm later.
		vkTransfer(tets, *mTethedrons.handle());
		//tets.assign(mTetrahedronIndex);

		for (size_t i = 0; i < tets.size(); i++)
		{
			uint32_t v0 = tets[i][0];
			uint32_t v1 = tets[i][1];
			uint32_t v2 = tets[i][2];
			uint32_t v3 = tets[i][3];

			tris.push_back(v0);
			tris.push_back(v1);
			tris.push_back(v2);

			tris.push_back(v0);
			tris.push_back(v3);
			tris.push_back(v1);

			tris.push_back(v0);
			tris.push_back(v2);
			tris.push_back(v3);

			tris.push_back(v1);
			tris.push_back(v3);
			tris.push_back(v2);

			triangles.push_back(dyno::TopologyModule::Triangle(v0, v1, v2));
			triangles.push_back(dyno::TopologyModule::Triangle(v0, v1, v3));
			triangles.push_back(dyno::TopologyModule::Triangle(v0, v2, v3));
			triangles.push_back(dyno::TopologyModule::Triangle(v1, v2, v3));
		}

		this->mIndex.resize(tris.size());
		this->mTriangleIndex.resize(triangles.size());

		//vkTransfer(mIndex, tris);
		this->mIndex.assign(tris);
		//vkTransfer(mTriangleIndex, triangles);
		this->mTriangleIndex.assign(triangles);

		tets.clear();
		tris.clear();
		triangles.clear();
	}
	*/

	DEFINE_CLASS(TetrahedronSet)
}