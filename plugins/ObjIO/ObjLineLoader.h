#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Field.h"
#include "Field/FilePath.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	/*!
	*	\class	ObjLine
	*	\brief	A node containing a EdgeSet object
	*
	*	This class is typically used as a representation for a static boundary mesh or base class to other classes
	*
	*/
	template<typename TDataType>
	class ObjLine : public Node
	{
		DECLARE_TCLASS(ObjLine, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::Quat<Real> TQuat;

		ObjLine();
		std::string getNodeType() override { return "IO"; }

	public:
		DEF_VAR(Vec3f, Location, 0, "Node location");
		DEF_VAR(Vec3f, Rotation, 0, "Node rotation");
		DEF_VAR(Vec3f, Scale, Vec3f(1.0f), "Node scale");

		DEF_VAR(FilePath, FileName, "", "");

		DEF_VAR(Real, Radius, 1, "Point Radius");

		DEF_VAR(bool, Sequence, false, "Import Sequence");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, EdgeSet, "Topology");


	protected:
		void resetStates() override;
		void loadObj(EdgeSet<TDataType>& EdgeSet,std::string filename);

	private:
		std::string trim(const std::string& str) {
			size_t start = str.find_first_not_of(" \t\r\n");
			size_t end = str.find_last_not_of(" \t\r\n");
			if (start == std::string::npos) return "";
			return str.substr(start, end - start + 1);
		}

		void parseOBJ(const std::string& filename,
			std::vector<Vec3f>& vertices,
			std::vector<TopologyModule::Edge>& edges);

		DArray<Coord> initPos;

		Coord center;
		Coord centerInit;

	};
};