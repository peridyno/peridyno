#include "TetModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	TetModel<TDataType>::TetModel()
		: BasicShape<TDataType>()
	{
		this->stateTetSet()->setDataPtr(std::make_shared<TetrahedronSet<TDataType>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&TetModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varV0()->attach(callback);
		this->varV1()->attach(callback);
		this->varV2()->attach(callback);
		this->varV3()->attach(callback);

		auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		tsRender->setVisible(true);
		this->stateTetSet()->connect(tsRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(tsRender);

		this->stateTetSet()->promoteOuput();
	}

	template<typename TDataType>
	NBoundingBox TetModel<TDataType>::boundingBox()
	{
		NBoundingBox bound;

		auto tet = this->outTet()->getData();
		auto aabb = tet.aabb();

		Coord v0 = aabb.v0;
		Coord v1 = aabb.v1;

		bound.lower = Vec3f(v0.x, v0.y, v0.z);
		bound.upper = Vec3f(v1.x, v1.y, v1.z);

		return bound;
	}

	template<typename TDataType>
	void TetModel<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void TetModel<TDataType>::varChanged() 
	{
		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		auto v0 = this->varV0()->getValue();
		auto v1 = this->varV1()->getValue();
		auto v2 = this->varV2()->getValue();
		auto v3 = this->varV3()->getValue();

		Quat<Real> q = this->computeQuaternion();

		q.normalize();

		Transform<Real, 3> t(center, q.toMatrix3x3(), scale);

		TTet3D<Real> tet;
		tet.v[0] = t * v0;
		tet.v[1] = t * v1;
		tet.v[2] = t * v2;
		tet.v[3] = t * v3;
		this->outTet()->setValue(tet);

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Tetrahedron> tets;

		vertices.push_back(tet.v[0]);
		vertices.push_back(tet.v[1]);
		vertices.push_back(tet.v[2]);
		vertices.push_back(tet.v[3]);

		tets.push_back(TopologyModule::Tetrahedron(0, 1, 2, 3));

		auto ts = this->stateTetSet()->getDataPtr();
		ts->setPoints(vertices);
		ts->setTetrahedrons(tets);
		ts->update();

		vertices.clear();
		tets.clear();
	}

	DEFINE_CLASS(TetModel);
}