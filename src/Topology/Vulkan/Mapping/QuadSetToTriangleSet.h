#pragma once
#include "Module/TopologyMapping.h"

#include "Topology/QuadSet.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
    template<typename TDataType>
    class QuadSetToTriangleSet : public TopologyMapping {
        DECLARE_TCLASS(QuadSetToTriangleSet, TDataType)
    public:
        QuadSetToTriangleSet();
        ~QuadSetToTriangleSet();

    public:
        DEF_INSTANCE_IN(QuadSet<TDataType>, QuadSet, "");
        DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

    protected:
        bool apply() override;
    };

    using QuadSetToTriangleSet3f = QuadSetToTriangleSet<DataType3f>;
} // namespace dyno