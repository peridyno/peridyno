#include "TriangleSet.h"
#include "VkTransfer.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

namespace dyno
{
    template <typename TDataType>
    TriangleSet<TDataType>::TriangleSet()
        : EdgeSet<TDataType>(),
          mSort(getSpvFile("shaders/glsl/topology/QuadKeySort.comp.spv")) {
        this->addKernel("SetupTriangleIndices",
                        std::make_shared<VkProgram>(BUFFER(uint32_t), // int: height
                                                    BUFFER(Triangle), // in:  CapillaryTexture
                                                    CONSTANT(uint)    // in:  horizon & realSize
                                                    ));
        this->kernel("SetupTriangleIndices")
            ->load(getSpvFile("shaders/glsl/topology/SetupTriangleIndices.comp.spv"));

        this->addKernel("SetupTriKey",
                        std::make_shared<VkProgram>(BUFFER(EKey), BUFFER(int), BUFFER(Triangle), CONSTANT(uint)));
        this->addKernel("CountEKey", std::make_shared<VkProgram>(BUFFER(int), BUFFER(EKey), CONSTANT(uint)));
        this->addKernel("SetupEdge2Tri", std::make_shared<VkProgram>(BUFFER(Edge), BUFFER(Edg2Tri), BUFFER(EKey),
                                                                     BUFFER(int), BUFFER(int), CONSTANT(uint)));

        this->addKernel("UpdateTriangleIndex", std::make_shared<VkProgram>(BUFFER(Triangle), CONSTANT(uint), CONSTANT(uint), CONSTANT(uint)));
        this->kernel("SetupTriKey")->load(getSpvFile("shaders/glsl/topology/SetupTriKey.comp.spv"));
        this->kernel("CountEKey")->load(getSpvFile("shaders/glsl/topology/CountEKey.comp.spv"));
        this->kernel("SetupEdge2Tri")->load(getSpvFile("shaders/glsl/topology/SetupEdge2Tri.comp.spv"));
        this->kernel("UpdateTriangleIndex")->load(getSpvFile("shaders/glsl/topology/UpdateTriangleIndex.comp.spv"));
    }

    template <typename TDataType>
    TriangleSet<TDataType>::~TriangleSet() {
    }

    template <typename TDataType>
    void TriangleSet<TDataType>::setTriangles(const std::vector<Triangle>& indices) {
        mTriangleIndex.assign(indices);
    }

    template <typename TDataType>
    void TriangleSet<TDataType>::setTriangles(const DArray<Triangle>& indices) {
        mTriangleIndex.assign(indices);
    }

    template <typename TDataType>
    void TriangleSet<TDataType>::updateTopology() {
        this->updateTriangles();

        EdgeSet<TDataType>::updateTopology();
    }

    template <typename TDataType>
    void TriangleSet<TDataType>::updateTriangles() {
        uint num = mTriangleIndex.size();

        mIndex.resize(3 * num);

        auto vk_num = VkConstant<uint>(num);
        this->kernel("SetupTriangleIndices")
            ->flush(vkDispatchSize(num, 64), mIndex.handle(), mTriangleIndex.handle(), &vk_num);
    }

    template <typename TDataType>
    void TriangleSet<TDataType>::updateEdges() {
        VkConstant<uint> n;
        uint triSize = mTriangleIndex.size();
        DArray<EKey> keys;
        DArray<int> triIds;

        n.setValue(triSize);
        keys.resize(3 * triSize);
        triIds.resize(3 * triSize);
        this->kernel("SetupTriKey")
            ->flush(vkDispatchSize(triSize, 64), keys.handle(), triIds.handle(), mTriangleIndex.handle(), &n);
        mSort.sortByKey(keys, triIds, SortParam::eUp);
        DArray<int> counter;
        counter.resize(3 * triSize);

        n.setValue(keys.size());
        this->kernel("CountEKey")->flush(vkDispatchSize(keys.size(), 64), counter.handle(), keys.handle(), &n);

        int edgeNum = this->reduce().reduce(*counter.handle());
        this->scan().scan(*counter.handle(), *counter.handle(), VkScan<int>::Type::Exclusive);

        mEdg2Tri.resize(edgeNum);

        auto& pEdges = this->getEdges();
        pEdges.resize(edgeNum);

        n.setValue(keys.size());
        this->kernel("SetupEdge2Tri")
            ->flush(vkDispatchSize(keys.size(), 64), pEdges.handle(), mEdg2Tri.handle(), keys.handle(),
                    counter.handle(), triIds.handle(), &n);
        counter.clear();
        triIds.clear();
        keys.clear();
    }

    template<typename TDataType>
    bool TriangleSet<TDataType>::loadObjFile(std::string filename)
    {
        std::vector<Coord> vertList;
        std::vector<Triangle> faceList;

        tinyobj::attrib_t myattrib;
        std::vector <tinyobj::shape_t> myshape;
        std::vector <tinyobj::material_t> mymat;
        std::string mywarn;
        std::string myerr;

        char* fname = (char*)filename.c_str();

        bool succeed = tinyobj::LoadObj(&myattrib, &myshape, &mymat, &mywarn, &myerr, fname, nullptr, true, true);
        if (!succeed)
            return false;

        for (int i = 0; i < myattrib.GetVertices().size() / 3; i++)
        {
            vertList.push_back(Coord(myattrib.GetVertices()[3 * i], myattrib.GetVertices()[3 * i + 1], myattrib.GetVertices()[3 * i + 2]));
        }

        for (int i = 0; i < myshape.size(); i++)
        {
            for (int s = 0; s < myshape[i].mesh.indices.size() / 3; s++)
            {
                faceList.push_back(Triangle(myshape[i].mesh.indices[3 * s].vertex_index, myshape[i].mesh.indices[3 * s + 1].vertex_index, myshape[i].mesh.indices[3 * s + 2].vertex_index));
            }
        }
        this->setPoints(vertList);
        this->setTriangles(faceList);
        this->update();

        vertList.clear();
        faceList.clear();
        myshape.clear();
        mymat.clear();

        return true;
    }

    template<typename TDataType>
    std::shared_ptr<TriangleSet<TDataType>> TriangleSet<TDataType>::merge(TriangleSet<TDataType>& ts)
    {
        auto ret = std::make_shared<TriangleSet<TDataType>>();

        auto& vertices = ret->getPoints();
        auto& indices = ret->getTriangles();

        uint vNum0 = this->mCoords.size();
        uint vNum1 = ts.getPoints().size();

        uint tNum0 = mTriangleIndex.size();
        uint tNum1 = ts.getTriangles().size();

        vertices.resize(vNum0 + vNum1);
        indices.resize(tNum0 + tNum1);

        vertices.assign(this->mCoords, vNum0, 0, 0);
        vertices.assign(ts.getPoints(), vNum1, vNum0, 0);

        indices.assign(mTriangleIndex, tNum0, 0, 0);
        indices.assign(ts.getTriangles(), tNum1, tNum0, 0);

        VkConstant<uint> vk_tnum1 {tNum1};
        VkConstant<uint> vk_vnum0 {vNum0};
        VkConstant<uint> vk_tnum0 {tNum0};
        this->kernel("UpdateTriangleIndex")
            ->flush(vkDispatchSize(tNum1, 64), indices.handle(), &vk_tnum1, &vk_vnum0, &vk_tnum0);
        return ret;
    }

    DEFINE_CLASS(TriangleSet)
} // namespace dyno