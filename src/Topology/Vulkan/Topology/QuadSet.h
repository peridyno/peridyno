#pragma once
#include "EdgeSet.h"
#include "TopologyConstants.h"
#include "Catalyzer/VkSort.h"

namespace dyno
{
    class QKey {
    public:
        DYN_FUNC QKey() {
            id[0] = EMPTY;
            id[1] = EMPTY;
            id[2] = EMPTY;
            id[3] = EMPTY;
        }

        DYN_FUNC QKey(PointType v0, PointType v1, PointType v2, PointType v3) {
            id[0] = v0;
            id[1] = v1;
            id[2] = v2;
            id[3] = v3;
        }

        DYN_FUNC inline PointType operator[](unsigned int i) {
            return id[i];
        }
        DYN_FUNC inline PointType operator[](unsigned int i) const {
            return id[i];
        }

        DYN_FUNC inline bool operator>=(const QKey& other) const {
            if (id[0] >= other.id[0]) return true;
            if (id[0] == other.id[0] && id[1] >= other.id[1]) return true;
            if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] >= other.id[2]) return true;
            if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] >= other.id[3])
                return true;

            return false;
        }

        DYN_FUNC inline bool operator>(const QKey& other) const {
            if (id[0] > other.id[0]) return true;
            if (id[0] == other.id[0] && id[1] > other.id[1]) return true;
            if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] > other.id[2]) return true;
            if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] > other.id[3])
                return true;

            return false;
        }

        DYN_FUNC inline bool operator<=(const QKey& other) const {
            if (id[0] <= other.id[0]) return true;
            if (id[0] == other.id[0] && id[1] <= other.id[1]) return true;
            if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] <= other.id[2]) return true;
            if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] <= other.id[3])
                return true;
            return false;
        }

        DYN_FUNC inline bool operator<(const QKey& other) const {
            if (id[0] < other.id[0]) return true;
            if (id[0] == other.id[0] && id[1] < other.id[1]) return true;
            if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] < other.id[2]) return true;
            if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] < other.id[3])
                return true;
            return false;
        }

        DYN_FUNC inline bool operator==(const QKey& other) const {
            return id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] == other.id[3];
        }

        DYN_FUNC inline bool operator!=(const QKey& other) const {
            return id[0] != other.id[0] || id[1] != other.id[1] || id[2] != other.id[2] || id[3] != other.id[3];
        }

    private:
        DYN_FUNC inline void swap(PointType& v0, PointType& v1) {
            PointType vt = v0;
            v0 = v0 < v1 ? v0 : v1;
            v1 = vt < v1 ? v1 : vt;
        }

        PointType id[4];
    };

    template <typename TDataType>
    class QuadSet : public EdgeSet<TDataType> {
    public:
        using Coord = Vec3f;
        using Edge = TopologyModule::Edge;
        using Quad = TopologyModule::Quad;
        using Edg2Quad = TopologyModule::Edg2Quad;

        QuadSet();
        ~QuadSet() override;

        DArray<Quad>& getQuads() {
            return mQuads;
        }
        void setQuads(std::vector<Quad>& quads);
        void setQuads(DArray<Quad>& quads);

        DArrayList<int>& getVertex2Quads();

        void copyFrom(QuadSet& quadSet);

    public:
        DEF_ARRAY_OUT(Coord, VertexNormal, DeviceType::GPU, "");

    protected:
        void updateTopology() override;

        void updateEdges() override;

        void updateVertexNormal();

        virtual void updateQuads() {};

    private:
        DArray<Quad> mQuads;
        DArrayList<int> mVer2Quad;
        DArray<Edg2Quad> mEdg2Quad;

        VkSortByKey<EKey, int> mSort;
    };

    using QuadSet3f = QuadSet<DataType3f>;
} // namespace dyno
