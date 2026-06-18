#pragma once
#include "QuadSet.h"
namespace dyno
{
	template<typename TDataType>
	class HexahedronSet : public QuadSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename Topology::Quad Quad;
		typedef typename Topology::Hexahedron Hexahedron;

		HexahedronSet();
		~HexahedronSet() override;

		//void loadTetFile(std::string filename);

		void setHexahedrons(std::vector<Hexahedron>& hexahedrons);
		void setHexahedrons(DArray<Hexahedron>& hexahedrons);

		// Return hexahedron indices
		DArray<Hexahedron>& hexahedronIndices() { return mHexahedrons; }

		// Return the mapping from quad to hexahedron
		DArray<::dyno::Topology::Quad2Hex>& quad2Hexahedron() { return mQuad2Hex; }

		DArrayList<int>& vertex2Hexahedron() { return mVer2Hex; }

		void calculateVolume(DArray<Real>& volume);

		void copyFrom(HexahedronSet<TDataType> hexSet);

	protected:
		void updateQuads() override;

		void updateTopology() override;

	private:
		// Automatically called when calling update()
		void updateVertex2Hexahedron();

		DArrayList<int> mVer2Hex;

		DArray<::dyno::Topology::Hexahedron> mHexahedrons;

		// Automatically updated when calling update()
		DArray<::dyno::Topology::Quad2Hex> mQuad2Hex;
	};
}

