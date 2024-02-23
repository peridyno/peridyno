#include "QuadSetToTriangleSet.h"

#include "Topology/QuadSet.h"

namespace dyno
{
    IMPLEMENT_TCLASS(QuadSetToTriangleSet, TDataType)

    template<typename TDataType>
    QuadSetToTriangleSet<TDataType>::QuadSetToTriangleSet() : TopologyMapping() {
        this->addKernel("Quad2Tri",
                        std::make_shared<VkProgram>(BUFFER(TopologyModule::Quad),     // in: quad
                                                    BUFFER(TopologyModule::Triangle), // out: tri
                                                    CONSTANT(uint32_t))               // in: num
        );
        this->kernel("Quad2Tri")->load(getSpvFile("shaders/glsl/topology/Quad2Triangle.comp.spv"));
    }

    template<typename TDataType>
    QuadSetToTriangleSet<TDataType>::~QuadSetToTriangleSet() {
    }

    template<typename TDataType>
    bool QuadSetToTriangleSet<TDataType>::apply() {
        if (this->outTriangleSet()->isEmpty()) {
            this->outTriangleSet()->allocate();
        }

        auto qs = this->inQuadSet()->constDataPtr();
        auto ts = this->outTriangleSet()->getDataPtr();

        auto& verts = qs->getPoints();
        auto& quads = qs->getQuads();

        auto& tris = ts->getTriangles();
        tris.resize(2 * quads.size());

        ts->setPoints(verts);

        static_assert(sizeof(TopologyModule::Quad) == 4 * sizeof(int32_t));
        static_assert(sizeof(TopologyModule::Triangle) == 4 * sizeof(int32_t));

        VkConstant<uint32_t> vk_num {quads.size()};
        this->kernel("Quad2Tri")->flush(vkDispatchSize(quads.size(), 64), quads.handle(), tris.handle(), &vk_num);

        this->outTriangleSet()->update();

        return true;
    }

    DEFINE_CLASS(QuadSetToTriangleSet)
} // namespace dyno