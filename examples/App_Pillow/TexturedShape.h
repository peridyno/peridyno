#pragma once
#include "Framework/Node.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	/*!
	*	\class	TextureShape
	*	\brief	Representing the boundary surface.
	*/
	template<typename TDataType>
	class TexturedShape : public Node
	{
		DECLARE_CLASS_1(TexturedShape, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> TexCoord;
		typedef typename TopologyModule::Triangle TexIndex;

		TexturedShape(std::string name = "default");
		virtual ~TexturedShape();

		void loadFile(std::string filename);
		void scale(Real s);

	public:
		DEF_EMPTY_CURRENT_ARRAY(TextureCoord, TexCoord, DeviceType::GPU, "");
		DEF_EMPTY_CURRENT_ARRAY(TextureIndex, TexIndex, DeviceType::GPU, "");

	protected:
	};


#ifdef PRECISION_FLOAT
	template class TexturedShape<DataType3f>;
#else
	template class TexturedShape<DataType3d>;
#endif
}