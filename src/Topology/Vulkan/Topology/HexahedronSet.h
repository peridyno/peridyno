#pragma once
#include "QuadSet.h"

namespace dyno
{

    template<typename TDataType>
    class HexahedronSet : public QuadSet<TDataType> {
    public:
        using Real = float;
        using Quad = TopologyModule::Quad;
        using Hexahedron = TopologyModule::Hexahedron;
        using Quad2Hex = TopologyModule::Quad2Hex;

        HexahedronSet();
        ~HexahedronSet();

        // void loadTetFile(std::string filename);

        void setHexahedrons(std::vector<Hexahedron>& hexahedrons);
        void setHexahedrons(DArray<Hexahedron>& hexahedrons);

        DArray<Hexahedron>& getHexahedrons() {
            return m_hexahedrons;
        }
        DArray<::dyno::TopologyModule::Tri2Tet>& getQua2Hex() {
            return quad2Hex;
        }

        DArrayList<int>& getVer2Hex();

        void getVolume(DArray<Real>& volume);

        void copyFrom(HexahedronSet hexSet);

    protected:
        void updateQuads() override;

    private:
        DArray<::dyno::TopologyModule::Hexahedron> m_hexahedrons;
        DArray<::dyno::TopologyModule::Quad2Hex> quad2Hex;
        DArrayList<int> m_ver2Hex;
        VkSortByKey<QKey, int> mSort;
    };

    using HexahedronSet3f = HexahedronSet<DataType3f>;
} // namespace dyno
