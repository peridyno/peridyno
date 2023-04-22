#include "SplineConstraint.h"

#include "Topology/TriangleSet.h"
#include <iostream>
#include <sys/stat.h>
#include "tinyobjloader/tiny_obj_loader.h"
#include "GLSurfaceVisualModule.h"
#include "math.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SplineConstraint, TDataType)

	template<typename TDataType>
	SplineConstraint<TDataType>::SplineConstraint()
		: Node()
	{
		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		
		this->stateTopology()->setDataPtr(triSet);

		auto module = std::make_shared<GLSurfaceVisualModule>();

		module->setColor(Vec3f(0.8, 0.52, 0.25));
		module->setVisible(true);
		this->stateTopology()->connect(module->inTriangleSet());
		this->graphicsPipeline()->pushModule(module);
	}

	template<typename TDataType>
	void SplineConstraint<TDataType>::resetStates()
	{

		Node::resetStates();

		tempLength = 0;
		updateTransform();
		CurrentVelocity = 0;
	}

	template <typename Coord, typename Matrix>
	__global__ void K_InitKernelFunctionMesh(
		DArray<Coord> posArr,
		DArray<Coord> posInit,
		Coord center,
		Coord centerInit,
		Matrix rotation
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;
		Coord pos;
		pos = posInit[pId] - centerInit;
		pos = rotation * pos;
		posArr[pId] = pos + center;

	}

	
	template<typename TDataType>
	void SplineConstraint<TDataType>::updateStates()
	{		

		updateTransform();


	}



	template<typename TDataType>
	void SplineConstraint<TDataType>::updateTransform() 
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTopology()->getDataPtr());
		//auto Velocity = this->varVelocity()->getData();
		auto VertexIn = this->inSpline()->getData().getPoints();
		auto VertexIn2 = this->inTriangleSet()->getData().getPoints();
		auto lengthV2 = this->inTriangleSet()->getData().getPointSize();
		auto TriIn = this->inTriangleSet()->getData().getTriangles();

		if (this->varAccelerate()->getData())
		{
			int sa = this->stateFrameNumber()->getData();
			std::cout << "CurrentFrame" << sa << std::endl;
			float st = this->stateElapsedTime()->getData();
			std::cout << "CurrentTime" << st << std::endl;

			UpdateCurrentVelocity();
		}
		else
		{
			CurrentVelocity = this->varVelocity()->getData();
		}

		CArray<Coord> c_point1;
		c_point1.assign(VertexIn);

		CArray<Coord> c_point2;
		c_point2.assign(VertexIn2);

		int lengthV = VertexIn.size();
		totalIndex = lengthV;

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		Real dt = this->stateTimeStep()->getData();
		tempLength = tempLength + CurrentVelocity * dt;

		double tlength = 0;
		size_t P1 = 0;
		size_t P2 = 1;
		size_t P3 = 2;
		double dis = 0;
		double disP = 0;
		bool update = false;
		Coord offestLocation = Coord(0, 0, 0);

		Coord Location1;
		Vec3f Location2;
		Vec3f LocationTemp1 = { 0,1,0 };

		Quat<Real> q;
		Quat<Real> q2;
		Quat<Real> qrotator;

		for (size_t i = 0; i < lengthV - 2; i++)
		{
			Vec3f ab = Vec3f(c_point1[i + 1][0] - c_point1[i][0], c_point1[i + 1][1] - c_point1[i][1], c_point1[i + 1][2] - c_point1[i][2]);
			if (tlength < tempLength)
			{
				tlength += ab.norm();
				update = false;
			}
			else
			{
				dis = ab.norm() - (tlength - tempLength);


				P1 = i;
				P2 = i + 1;
				P3 = i + 2;
				disP = dis/ab.norm();//dis / ab.norm()

				auto l = ab.normalize();

				offestLocation = Coord(c_point1[i][0]+dis*l[0], c_point1[i][1] + dis * l[1], c_point1[i][2] + dis * l[2]) ;//+ Coord(dis * l[0], dis * l[1], dis * l[2])
				
				//matrix

				Location2 = Vec3f(c_point1[P1][0], c_point1[P1][1], c_point1[P1][2]);

				LocationTemp1 = Vec3f(c_point1[P2][0], c_point1[P2][1], c_point1[P2][2]);

				Vec3f vb = Vec3f(0, 1, 0);
				Vec3f va = LocationTemp1 - Location2;
				Vec3f vc = Vec3f(c_point1[P3][0], c_point1[P3][1], c_point1[P3][2]) - Vec3f(c_point1[P2][0], c_point1[P2][1], c_point1[P2][2]);
				
				getQuatFromVector(va, vb, q);
				getQuatFromVector(vc, vb, q2);
				

				SLerp(q, q2, disP,qrotator);
				auto RV = [&](const Coord& v)->Coord
				{
					return qrotator.rotate(v);//
				};

				for (size_t k = 0; k < lengthV2; k++)
				{

					Location1 = { c_point2[k][0], c_point2[k][1], c_point2[k][2] };

					Location1 = RV(Location1)+offestLocation;

					vertices.push_back(Location1);

				}
				update = true;
				break;
			}

		}
		if (update)
		{
			triSet->setPoints(vertices);
			triSet->setTriangles(TriIn);
			triSet->update();
		}

		vertices.clear();
	
	}

	template<typename TDataType>
	void SplineConstraint<TDataType>::SLerp(Quat<Real> a, Quat<Real> b, double t, Quat<Real>& out)
	{
		double cosHalfTheta = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
		if (cosHalfTheta < 0.0F) {

			cosHalfTheta = -cosHalfTheta;
			b.x = -b.x;
			b.y = -b.y;
			b.z = -b.z;
			b.w = -b.w;
		}

		double halfTheta = (double)acos((double)cosHalfTheta);
		double sinHalfTheta = (double)sqrt((double)(1.0F - cosHalfTheta * cosHalfTheta));
		double ratioA;
		double ratioB;
		if ((double)abs(sinHalfTheta) > 0.001F) {

			double oneOverSinHalfTheta = 1.0F / sinHalfTheta;
			ratioA = (double)sin((double)((1.0F - t) * halfTheta)) * oneOverSinHalfTheta;
			ratioB = (double)sin((double)(t * halfTheta)) * oneOverSinHalfTheta;
		}
		else {

			ratioA = 1.0F - t;
			ratioB = t;
		}


		out.x = (float)(ratioA * a.x + ratioB * b.x);
		out.y = (float)(ratioA * a.y + ratioB * b.y);
		out.z = (float)(ratioA * a.z + ratioB * b.z);
		out.w = (float)(ratioA * a.w + ratioB * b.w);

		out.normalize();


	
	}

	template<typename TDataType>
	void SplineConstraint<TDataType>::getQuatFromVector(Vec3f va, Vec3f vb, Quat<Real>& q)
	{

		va.normalize();
		vb.normalize();

		Vec3f v = va.cross(vb);
		Vec3f vs = va.cross(vb);
		v.normalize();

		float ca = vb.dot(va);

		float scale = 1 - ca;

		Vec3f vt = Vec3f(v[0] * scale, v[1] * scale, v[2] * scale);

		SquareMatrix<Real, 3>rotationMatrix;
		rotationMatrix(0, 0) = vt[0] * v[0] + ca;
		rotationMatrix(1, 1) = vt[1] * v[1] + ca;
		rotationMatrix(2, 2) = vt[2] * v[2] + ca;
		vt[0] *= v[1];
		vt[2] *= v[0];
		vt[1] *= v[2];

		rotationMatrix(0, 1) = vt[0] - vs[2];
		rotationMatrix(0, 2) = vt[2] + vs[1];
		rotationMatrix(1, 0) = vt[0] + vs[2];
		rotationMatrix(1, 2) = vt[1] - vs[0];
		rotationMatrix(2, 0) = vt[2] - vs[1];
		rotationMatrix(2, 1) = vt[1] + vs[0];

		q = Quat<Real>(rotationMatrix);


	}

	template<typename TDataType>
	void SplineConstraint<TDataType>::UpdateCurrentVelocity() 
	{
		Real dt = this->stateTimeStep()->getData();
		float AcceleratedSpeed = this->varAcceleratedSpeed()->getData();
		float MaxSpeed = this->varVelocity()->getData();

		if (CurrentVelocity + AcceleratedSpeed *dt < MaxSpeed ) 
		{
			CurrentVelocity = CurrentVelocity + AcceleratedSpeed * dt;
		}
		else 
		{
			CurrentVelocity = MaxSpeed;
		}
	
	}



	DEFINE_CLASS(SplineConstraint);
}