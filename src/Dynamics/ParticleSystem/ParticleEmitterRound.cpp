#include "ParticleEmitterRound.h"
#include <time.h>

#include <stdlib.h>

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleEmitterRound, TDataType)

	template<typename TDataType>
	ParticleEmitterRound<TDataType>::ParticleEmitterRound(std::string name)
		: ParticleEmitter<TDataType>(name)
	{

		srand(time(0));
	}

	
	
	template<typename TDataType>
	ParticleEmitterRound<TDataType>::~ParticleEmitterRound()
	{
		mPosition.clear();
	}
	

	template<typename TDataType>
	void ParticleEmitterRound<TDataType>::generateParticles()
	{
		auto sampling_distance = this->varSamplingDistance()->getData();
		if (sampling_distance < EPSILON)
			sampling_distance = 0.005;
		auto center = this->varLocation()->getData();

		std::vector<Coord> pos_list;
		std::vector<Coord> vel_list;


		auto rot_vec = this->varRotation()->getData();

		Quat<Real> quat = Quat<float>::identity();
		float x_rad = rot_vec[0] / 180.0f * M_PI;
		float y_rad = rot_vec[1] / 180.0f * M_PI;
		float z_rad = rot_vec[2] / 180.0f * M_PI;

		quat = quat * Quat<Real>(x_rad, Coord(1, 0, 0));
		quat = quat * Quat<Real>(y_rad, Coord(0, 1, 0));
		quat = quat * Quat<Real>(z_rad, Coord(0, 0, 1));

		auto rot_mat = quat.toMatrix3x3();

		Coord v0 = this->varVelocityMagnitude()->getData()*rot_mat*Vec3f(0, -1, 0);

		auto r = this->varRadius()->getData();
		Real lo = -r;
		Real hi = r;

		for (Real x = lo; x <= hi; x += sampling_distance)
		{
			for (Real y = lo; y <= hi; y += sampling_distance)
			{
				Coord p = Coord(x, 0, y);
				if ((p - Coord(0)).norm() < r && rand() % 40 == 0)
				{
					//Coord q = cos(angle) * p + (1 - cos(angle)) * (p.dot(axis)) * axis + sin(angle) * axis.cross(p);
					Coord q = rot_mat * p;
					pos_list.push_back(q + center);
					vel_list.push_back(v0);
				}
			}
		}

		if (pos_list.size() > 0) {
			mPosition.resize(pos_list.size());
			mVelocity.resize(pos_list.size());

			mPosition.assign(pos_list);
			mVelocity.assign(vel_list);
		}
	
		
		pos_list.clear();
		vel_list.clear();
	}

	DEFINE_CLASS(ParticleEmitterRound);
}