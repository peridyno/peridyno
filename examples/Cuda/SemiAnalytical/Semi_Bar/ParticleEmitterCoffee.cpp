#include "ParticleEmitterCoffee.h"
#include <time.h>

#include <stdlib.h>

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleEmitterCoffee, TDataType)

		template<typename TDataType>
	ParticleEmitterCoffee<TDataType>::ParticleEmitterCoffee()
		: ParticleEmitter<TDataType>()
	{

		srand(time(0));
	}



	template<typename TDataType>
	ParticleEmitterCoffee<TDataType>::~ParticleEmitterCoffee()
	{
		mPosition.clear();
	}


	template<typename TDataType>
	void ParticleEmitterCoffee<TDataType>::generateParticles()
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

		Coord v0 = rot_mat*this->varInitialVelocity()->getData();

		auto r = this->varRadius()->getData();
		r /= 6.0;
		Real lo = -r;
		Real hi = r;
		Real d = 1.0*sampling_distance;
			for (Real z = lo; z <= hi; z += d)
			{
				for (Real y = lo; y <= hi; y += d)
				{
					Coord p = Coord(0, z, y);
					if ((p - Coord(0)).norm() < r && rand() % 3 == 0)
					{
						Coord q = p;
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

	DEFINE_CLASS(ParticleEmitterCoffee);
}