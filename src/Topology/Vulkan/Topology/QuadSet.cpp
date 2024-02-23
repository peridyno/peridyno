#include "QuadSet.h"
#include <fstream>
#include <iostream>
#include <sstream>
namespace dyno
{

    template <typename TDataType>
    QuadSet<TDataType>::QuadSet()
        : EdgeSet<TDataType>(), mSort(getSpvFile("shaders/glsl/topology/QuadKeySort.comp.spv")) {
        this->addKernel("SetupQuadKey",
                        std::make_shared<VkProgram>(BUFFER(EKey), BUFFER(int), BUFFER(Quad), CONSTANT(uint)));
        this->addKernel("CountEKey", std::make_shared<VkProgram>(BUFFER(int), BUFFER(EKey), CONSTANT(uint)));
        this->addKernel("SetupEdge", std::make_shared<VkProgram>(BUFFER(Edge), BUFFER(Edg2Quad), BUFFER(EKey),
                                                                 BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->addKernel("SetupVertexNormal", std::make_shared<VkProgram>(BUFFER(Coord), BUFFER(Coord), BUFFER(Quad),
                                                                         BUFFER(List<int>), CONSTANT(uint), CONSTANT(VkDeviceAddress)));

        this->kernel("SetupQuadKey")->load(getSpvFile("shaders/glsl/topology/SetupQuadKey.comp.spv"));
        this->kernel("CountEKey")->load(getSpvFile("shaders/glsl/topology/CountEKey.comp.spv"));
        this->kernel("SetupEdge")->load(getSpvFile("shaders/glsl/topology/SetupEdge.comp.spv"));
        this->kernel("SetupVertexNormal")->load(getSpvFile("shaders/glsl/topology/SetupVertexNormal.comp.spv"));
    }

    template <typename TDataType>
    QuadSet<TDataType>::~QuadSet() {
        mQuads.clear();
        mVer2Quad.clear();
        mEdg2Quad.clear();
    }

    template <typename TDataType>
    DArrayList<int>& QuadSet<TDataType>::getVertex2Quads() {
        DArray<uint> counter(this->mCoords.size());
        VkConstant<uint> num {mQuads.size()};
        VkConstant<uint> shape_size {sizeof(Quad)};

        counter.reset();
        this->kernel("CountShape")->flush(vkDispatchSize(mQuads.size(), 64), counter.handle(), mQuads.handle(), &num, &shape_size);
        mVer2Quad.resize(counter);

        VkConstant<VkDeviceAddress> vec2QuadAddr {mVer2Quad.lists().handle()->bufferAddress()};
        counter.reset();
        this->kernel("SetupShapeId")->flush(vkDispatchSize(mQuads.size(), 64), mQuads.handle(), &num, &shape_size, &vec2QuadAddr);
        counter.clear();
        return mVer2Quad;
    }

    template <typename TDataType>
    void QuadSet<TDataType>::updateEdges() {
        VkConstant<uint> n;
        uint quadSize = mQuads.size();
        DArray<EKey> keys;
        DArray<int> quadIds;

        n.setValue(quadSize);
        keys.resize(4 * quadSize);
        quadIds.resize(4 * quadSize);
        this->kernel("SetupQuadKey")
            ->flush(vkDispatchSize(quadSize, 64), keys.handle(), quadIds.handle(), mQuads.handle(), &n);
        mSort.sortByKey(keys, quadIds, SortParam::eUp);
        DArray<int> counter;
        counter.resize(4 * quadSize);

        n.setValue(keys.size());
        this->kernel("CountEKey")->flush(vkDispatchSize(keys.size(), 64), counter.handle(), keys.handle(), &n);

        int edgeNum = this->reduce().reduce(*counter.handle());
        this->scan().scan(*counter.handle(), *counter.handle(), VkScan<int>::Type::Exclusive);

        mEdg2Quad.resize(edgeNum);

        auto& pEdges = this->getEdges();
        pEdges.resize(edgeNum);

        n.setValue(keys.size());
        this->kernel("SetupEdge")
            ->flush(vkDispatchSize(keys.size(), 64), pEdges.handle(), mEdg2Quad.handle(), keys.handle(),
                    counter.handle(), quadIds.handle(), &n);
        counter.clear();
        quadIds.clear();
        keys.clear();
    }

    template <typename TDataType>
    void QuadSet<TDataType>::setQuads(std::vector<Quad>& quads) {
        mQuads.resize(quads.size());
        mQuads.assign(quads);

        // this->updateTriangles();
    }

    template<typename TDataType>
    void QuadSet<TDataType>::setQuads(DArray<Quad>& quads)
    {
        mQuads.assign(quads);
    }

    template <typename TDataType>
    void QuadSet<TDataType>::copyFrom(QuadSet& quadSet) {
        mVer2Quad.assign(quadSet.mVer2Quad);

        mQuads.resize(quadSet.mQuads.size());
        mQuads.assign(quadSet.mQuads);

        mEdg2Quad.resize(quadSet.mEdg2Quad.size());
        mEdg2Quad.assign(quadSet.mEdg2Quad);

        EdgeSet<TDataType>::copyFrom(quadSet);
    }

    template <typename TDataType>
    void QuadSet<TDataType>::updateTopology() {
        this->updateQuads();

        EdgeSet<TDataType>::updateTopology();
    }

    template <typename TDataType>
    void QuadSet<TDataType>::updateVertexNormal() {
        if (this->outVertexNormal()->isEmpty()) this->outVertexNormal()->allocate();

        auto& vn = this->outVertexNormal()->getData();
        uint vertSize = this->mCoords.size();

        if (vn.size() != vertSize) {
            vn.resize(vertSize);
        }

        auto& vert2Quad = getVertex2Quads();

        VkConstant<uint> n;
        VkConstant<VkDeviceAddress> list {vert2Quad.lists().handle()->bufferAddress()};
        n.setValue(vertSize);
        this->kernel("SetupVertexNormal")
            ->flush(vkDispatchSize(vertSize, 64), vn.handle(), this->mCoords.handle(), mQuads.handle(), &n, &list);
    }

    DEFINE_CLASS(QuadSet);
} // namespace dyno
