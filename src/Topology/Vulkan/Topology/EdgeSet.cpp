#include "EdgeSet.h"

namespace dyno
{

	template<typename TDataType>
	EdgeSet<TDataType>::EdgeSet()
		: PointSet<TDataType>()
	{
        this->addKernel("CountShape", std::make_shared<VkProgram>(BUFFER(uint), BUFFER(PointType), CONSTANT(uint), CONSTANT(uint)));
        this->addKernel("SetupShapeId",
                        std::make_shared<VkProgram>(BUFFER(int), CONSTANT(uint), CONSTANT(uint), CONSTANT(VkDeviceAddress)));
		
        this->kernel("CountShape")->load(getSpvFile("shaders/glsl/topology/CountShape.comp.spv"));
        this->kernel("SetupShapeId")->load(getSpvFile("shaders/glsl/topology/SetupShapeId.comp.spv"));
	}

	template<typename TDataType>
	EdgeSet<TDataType>::~EdgeSet()
	{
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::setEdges(const DArray<Edge>& edges)
	{
		mEdgeIndex.assign(edges);

		this->tagAsChanged();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::setEdges(const std::vector<Edge>& edges)
	{
		mEdgeIndex.assign(edges);

		this->tagAsChanged();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::copyFrom(EdgeSet& es)
	{
		mEdgeIndex.assign(es.mEdgeIndex);
		this->mCoords.assign(es.mCoords);

		this->tagAsChanged();
	}

	template<typename TDataType>
	void EdgeSet<TDataType>::updateTopology()
	{
		this->updateEdges();

		PointSet<TDataType>::updateTopology();
	}

	DEFINE_CLASS(EdgeSet)
}