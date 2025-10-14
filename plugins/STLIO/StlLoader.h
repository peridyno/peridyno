#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Field.h"
#include "Field/FilePath.h"

#include "Node/ParametricModel.h"

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	/*!
	*	\class	StlLoader
	*	\brief	A node containing a TriangleSet object
	*
	*	This class is typically used as a representation for a static boundary mesh or base class to other classes
	*
	*/
	template<typename TDataType>
	class StlLoader : virtual public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(StlLoader, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::Quat<Real> TQuat;

		StlLoader();

		std::string getNodeType() override { return "IO"; }

	public:
		DEF_VAR(FilePath, FileName, "", "");


		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Topology, "Topology");

	public:

	protected:
		void resetStates() override;
		void loadSTL(TriangleSet<TDataType>& Triangleset,std::string filename);

	private:

		DArray<Coord> initPos;

	};
}