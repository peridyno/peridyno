#include "PolygonSet.h"

namespace dyno
{
	template<typename TDataType>
	PolygonSet<TDataType>::PolygonSet()
		: EdgeSet<TDataType>()
	{
		this->addKernel("PolygonCount",
			std::make_shared<VkProgram>(BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonCount")->load(getSpvFile("shaders/glsl/topology/PolygonCount.comp.spv"));

		this->addKernel("PolygonCountEdge",
			std::make_shared<VkProgram>(BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonCountEdge")->load(getSpvFile("shaders/glsl/topology/PolygonCountEdge.comp.spv"));

		this->addKernel("PolygonSetupEdgeIndice",
			std::make_shared<VkProgram>(BUFFER(Edge), BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonSetupEdgeIndice")->load(getSpvFile("shaders/glsl/topology/PolygonSetupEdgeIndice.comp.spv"));

		this->addKernel("PolygonCountTriangle",
			std::make_shared<VkProgram>(BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonCountTriangle")->load(getSpvFile("shaders/glsl/topology/PolygonCountTriangle.comp.spv"));

		this->addKernel("PolygonSetupTriangleIndice",
			std::make_shared<VkProgram>(BUFFER(Triangle), BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonSetupTriangleIndice")->load(getSpvFile("shaders/glsl/topology/PolygonSetupTriangleIndice.comp.spv"));

		this->addKernel("PolygonExtractQuad",
			std::make_shared<VkProgram>(BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonExtractQuad")->load(getSpvFile("shaders/glsl/topology/PolygonExtractQuad.comp.spv"));

		this->addKernel("PolygonExtractQuadIndice",
			std::make_shared<VkProgram>(BUFFER(Quad), BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonExtractQuadIndice")->load(getSpvFile("shaders/glsl/topology/PolygonExtractQuadIndice.comp.spv"));

		this->addKernel("PolygonExtractTriangle",
			std::make_shared<VkProgram>(BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonExtractTriangle")->load(getSpvFile("shaders/glsl/topology/PolygonExtractTriangle.comp.spv"));

		this->addKernel("PolygonExtractTriangleIndice",
			std::make_shared<VkProgram>(BUFFER(Triangle), BUFFER(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		this->kernel("PolygonExtractTriangleIndice")->load(getSpvFile("shaders/glsl/topology/PolygonExtractTriangleIndice.comp.spv"));
	}

	template<typename TDataType>
	PolygonSet<TDataType>::~PolygonSet()
	{
		mPolygonIndex.clear();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::setPolygons(const CArrayList<uint>& indices)
	{
		mPolygonIndex.assign(indices);
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::setPolygons(const DArrayList<uint>& indices)
	{
		mPolygonIndex.assign(indices);
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::copyFrom(PolygonSet<TDataType>& polygons)
	{
		PointSet<TDataType>::copyFrom(polygons);

		mPolygonIndex.assign(polygons.mPolygonIndex);
	}

	template<typename TDataType>
	bool PolygonSet<TDataType>::isEmpty()
	{
		bool empty = true;
		empty |= mPolygonIndex.size() == 0;

		return empty;
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::updateTopology()
	{
		return;
		uint vNum = this->mCoords.size();

		//Update the vertex to polygon mapping
		DArray<uint> counter(vNum);
		counter.reset();

		VkConstant<uint> vk_num{ vNum };
		VkConstant<VkDeviceAddress> vk_triangle_index { mPolygonIndex.lists().handle()->bufferAddress() };
		this->kernel("PolygonCount")
			->flush(vkDispatchSize(vk_num, 64), counter.handle(), &vk_num, &vk_triangle_index);

/*
		mVer2Poly.resize(counter);

		cuExecute(vNum,
			PolygonSet_SetupVertex2Polygon,
			mVer2Poly,
			mPolygonIndex);
*/
		EdgeSet<TDataType>::updateTopology();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::extractEdgeSet(EdgeSet<TDataType>& es)
	{
		es.clear();

		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);

		VkConstant<uint> vk_num{ polyNum };
		VkConstant<VkDeviceAddress> vk_triangle_index { mPolygonIndex.lists().handle()->bufferAddress() };
		this->kernel("PolygonCountEdge")
			->flush(vkDispatchSize(vk_num, 64), radix.handle(), &vk_num, &vk_triangle_index);
		int eNum = mReduce.reduce(*radix.handle());
		assert(eNum >= 0);
		mScan.scan(*radix.handle(), *radix.handle(), VkScan<uint>::Exclusive);

		DArray<Edge> edges(eNum);
		vk_num.setValue(polyNum);
		this->kernel("PolygonSetupEdgeIndice")
			->flush(vkDispatchSize(vk_num, 64), edges.handle(), radix.handle(), &vk_num, &vk_triangle_index);

		es.setPoints(this->mCoords);
		es.setEdges(edges);
		es.update();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::extractTriangleSet(TriangleSet<TDataType>& ts)
	{
		ts.clear();

		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);

		VkConstant<uint> vk_num{ polyNum };
		VkConstant<VkDeviceAddress> vk_triangle_index{ mPolygonIndex.lists().handle()->bufferAddress() };
		this->kernel("PolygonExtractTriangle")
			->flush(vkDispatchSize(vk_num, 64), radix.handle(), &vk_num, &vk_triangle_index);
		int eNum = mReduce.reduce(*radix.handle());
		assert(eNum >= 0);
		mScan.scan(*radix.handle(), *radix.handle(), VkScan<uint>::Exclusive);

		int tNum = 0;
		DArray<Triangle> triangleIndices(tNum);

		vk_num.setValue(polyNum);
		this->kernel("PolygonExtractTriangleIndice")
			->flush(vkDispatchSize(vk_num, 64), triangleIndices.handle(), radix.handle(), &vk_num, &vk_triangle_index);

		ts.setPoints(this->mCoords);
		ts.setTriangles(triangleIndices);
		ts.update();

		radix.clear();
		triangleIndices.clear();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::extractQuadSet(QuadSet<TDataType>& qs)
	{
		qs.clear();

		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);

		VkConstant<uint> vk_num{ polyNum };
		VkConstant<VkDeviceAddress> vk_triangle_index{ mPolygonIndex.lists().handle()->bufferAddress() };
		this->kernel("PolygonExtractQuad")
			->flush(vkDispatchSize(vk_num, 64), radix.handle(), &vk_num, &vk_triangle_index);
		int eNum = mReduce.reduce(*radix.handle());
		assert(eNum >= 0);
		mScan.scan(*radix.handle(), *radix.handle(), VkScan<uint>::Exclusive);

		int tNum = 0;
		DArray<Quad> quadIndices(tNum);

		vk_num.setValue(polyNum);
		this->kernel("PolygonExtractQuadIndice")
			->flush(vkDispatchSize(vk_num, 64), quadIndices.handle(), radix.handle(), &vk_num, &vk_triangle_index);

		qs.setPoints(this->mCoords);
		qs.setQuads(quadIndices);
		qs.update();

		radix.clear();
		quadIndices.clear();
	}

	template<typename TDataType>
	void PolygonSet<TDataType>::turnIntoTriangleSet(TriangleSet<TDataType>& ts)
	{
		ts.clear();

		uint polyNum = mPolygonIndex.size();

		DArray<uint> radix(polyNum);
		radix.reset();

		VkConstant<uint> vk_num{ polyNum };
		VkConstant<VkDeviceAddress> vk_triangle_index{ mPolygonIndex.lists().handle()->bufferAddress() };
		this->kernel("PolygonCountTriangle")
			->flush(vkDispatchSize(vk_num, 64), radix.handle(), &vk_num, &vk_triangle_index);

		int tNum = mReduce.reduce(*radix.handle());
		assert(tNum >= 0);
		mScan.scan(*radix.handle(), *radix.handle(), VkScan<uint>::Exclusive);

		DArray<Triangle> triangleIndex(tNum);

		vk_num.setValue(polyNum);
		this->kernel("PolygonSetupTriangleIndice")
			->flush(vkDispatchSize(vk_num, 64), triangleIndex.handle(), radix.handle(), &vk_num, &vk_triangle_index);

		ts.setPoints(this->mCoords);
		ts.setTriangles(triangleIndex);
		ts.update();

		radix.clear();
		triangleIndex.clear();
	}

	DEFINE_CLASS(PolygonSet);
}