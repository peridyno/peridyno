#pragma once
#include "TriangleSet.h"
#include "Catalyzer/VkSort.h"

namespace dyno
{

    template <typename TDataType>
    class TetrahedronSet : public TriangleSet<TDataType> {
    public:
        struct Info {
            int tetId;
            int orderId;
            dyno::TopologyModule::Triangle tIndex;
        };
        using Real = typename TDataType::Real;
        using Coord = typename TDataType::Coord;
        using Triangle = TopologyModule::Triangle;
        using Tetrahedron = TopologyModule::Tetrahedron;
        using Tri2Tet = TopologyModule::Tri2Tet;

        TetrahedronSet();
        ~TetrahedronSet() override;

        void setTetrahedrons(std::vector<Tetrahedron>& indices);
        void setTetrahedrons(DArray<Tetrahedron>& tetrahedrons);

        DArrayList<int>& getVer2Tet();

        void copyFrom(TetrahedronSet& es);

        DArray<Tetrahedron>& getTetrahedrons() {
            return mTethedrons;
        }
        DArray<Tri2Tet>& getTri2Tet() {
            return mTri2Tet;
        }
        DArray<Tri2Tet>& getTri2Tetorder() {
            return mTri2Tetorder;
        }

    protected:
        void updateTriangles() override;

    public:
        DArray<Tetrahedron> mTethedrons;
        DArray<Tri2Tet> mTri2Tet;
        DArray<Tri2Tet> mTri2Tetorder;

        DArrayList<int> mVer2Tet;

        VkSortByKey<TKey, Info> mSort;
    };

    using TetrahedronSet3f = TetrahedronSet<DataType3f>;
} // namespace dyno
