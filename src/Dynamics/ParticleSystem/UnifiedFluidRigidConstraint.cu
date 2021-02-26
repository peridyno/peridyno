#include <cuda_runtime.h>
#include "UnifiedFluidRigidConstraint.h"
#include "Framework/Node.h"
#include "Utility.h"
#include "SummationDensity.h"
#include "DensitySummationMesh.h"
#include "Attribute.h"
#include "Kernel.h"
#include "Topology/Primitive3D.h"



namespace dyno
{
	
	template <typename Coord>
	__global__ void USFI_update_rigid_rigid(
		GArray<Coord> tmp_velocity,
		GArray<Coord> tmp_angular_velocity,
		GArray<Coord> AA,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= AA.size() / 2) return;

		tmp_velocity[pId] += AA[2 * pId] * dt;
		tmp_angular_velocity[pId] += AA[2 * pId + 1] * dt;
	}

	template <typename Coord, typename Matrix>
	__global__ void USFI_update_fluid_rigid(
		GArray<Real> force,
		GArray<Coord> AA,
		GArray<Coord> rigid_pos,
		GArray<Real> rigid_mass,
		GArray<Matrix> rigid_interior,
		GArray<NeighborConstraints> nbc,
		Real sampling_distance,
		Real restDensity,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;
		
		int i = nbc[pId].idx1;
		int j = nbc[pId].idx2;

		Real mass_rigid = rigid_mass[j];
		//printf("mass of rigid = %.10lf\n", mass_rigid);
		Matrix interior_rigid = rigid_interior[j];

		Coord r_rigid = nbc[pId].pos2 - rigid_pos[j];
		nbc[pId].v1 = r_rigid;

		Coord normal_ij = nbc[pId].pos2 - nbc[pId].pos1;
		Coord dv_i = 1.0f * normal_ij.normalize() * force[pId] / mass_rigid;
		Coord dtor_i = interior_rigid.inverse() * r_rigid.cross(dv_i) * mass_rigid;

		//AA[2 * j] += dv_i;
		//AA[2 * j + 1] += dv_i;

		//printf("rigid norm %.10lf\n", dv_i.norm() * dt);

		atomicAdd(&AA[j * 2][0], dv_i[0]);
		atomicAdd(&AA[j * 2][1], dv_i[1]);
		atomicAdd(&AA[j * 2][2], dv_i[2]);

		atomicAdd(&AA[j * 2 + 1][0], dtor_i[0]);
		atomicAdd(&AA[j * 2 + 1][1], dtor_i[1]);
		atomicAdd(&AA[j * 2 + 1][2], dtor_i[2]);
		
	}

	template <typename Coord, typename Matrix>
	__global__ void USFI_update_fluid_rigid(
		GArray<Real> force,
		GArray<Coord> tmp_rigid_velocity,
		GArray<Coord> tmp_rigid_angular_velocity,
		GArray<Coord> rigid_pos,
		GArray<Real> rigid_mass,
		GArray<Matrix> rigid_interior,
		GArray<NeighborConstraints> nbc,
		Real sampling_distance,
		Real restDensity,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;

		int i = nbc[pId].idx1;
		int j = nbc[pId].idx2;

		Real mass_rigid = rigid_mass[j];
		Matrix interior_rigid = rigid_interior[j];

		Coord r_rigid = nbc[pId].pos2 - rigid_pos[j];
		nbc[pId].v1 = r_rigid;
		//Coord r_rigid = nbc[pId].v1;
		

		Coord normal_ij = nbc[pId].pos2 - nbc[pId].pos1;
		Coord dv_i = 1.0f * normal_ij.normalize() * force[pId] * dt / mass_rigid;
		Coord dtor_i = interior_rigid.inverse() * r_rigid.cross(dv_i) * mass_rigid;

		tmp_rigid_velocity[j] += dv_i;
		tmp_rigid_angular_velocity[j] += dv_i;

	}

	template <typename Coord>
	__global__ void USFI_compute_gradient(
		GArray<Real> force,
		GArray<Coord> velocity_fluid_in_iteration,
		GArray<Coord> tmp_rigid_velocity,
		GArray<Coord> tmp_rigid_angular_velocity,
		GArray<Coord> rigid_pos,
		GArray<Real> gradient,
		GArray<NeighborConstraints> nbc,
		Real sampling_distance,
		Real restDensity,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;

		int i = nbc[pId].idx1;
		int j = nbc[pId].idx2;

		Real mass_i = restDensity * pow(sampling_distance, 3);

		Coord normal_ij = nbc[pId].pos2 - nbc[pId].pos1;
		normal_ij = normal_ij.normalize();
		Coord normal_ji = - normal_ij;

		Coord r_rigid = nbc[pId].pos2 - rigid_pos[j];
		//nbc[pId].v1 = r_rigid;
		//r_rigid /= r_rigid.norm();

		Real gradient_r_linear = tmp_rigid_velocity[j].dot(normal_ij) *  mass_i * 1.0f * dt;

		Real gradient_r_angular = 1.0f * dt * (tmp_rigid_angular_velocity[j].cross(r_rigid)).dot(normal_ij) * mass_i;
								

		Real gradient_fluid = velocity_fluid_in_iteration[i].dot(normal_ji) * 1.0f * dt * mass_i;

		gradient[pId] = gradient_r_linear + gradient_fluid + gradient_r_angular;

	}

	template <typename Real>
	__global__ void USFI_update_gradient(
		GArray<Real> force,
		GArray<Real> delta_force,
		GArray<Real> gradient_point,
		GArray<Real> gradient_edge,
		GArray<NeighborConstraints> nbc,
		Real sampling_distance,
		Real restDensity
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;

		int i = nbc[pId].idx1;
		int j = nbc[pId].idx2;

		//Real invAlpha_j = 1.0f / invRadius[j];

		Real r0 = (nbc[pId].pos1 - nbc[pId].pos2).norm();
		Real a_ij = (1.0f / r0);
		//force[pId] -= 1.0f * gradient_point[j] * a_ij;
		//Real force_new = force[pId] - 15.0f * gradient_edge[pId] * a_ij ;
		
		Real force_new = force[pId] - 1.2f * gradient_point[i] * a_ij ;
		if (force_new < 0) force_new = 0;
		force_new -= force[pId];
		force[pId] += force_new;
		delta_force[pId] = force_new;
	}

	template <typename Real, typename Coord>
	__global__ void USFI_collision_rigid_fluid(
		GArray<Coord> pos,
		GArray<Coord> vel,
		GArray<Box3D> boxes,
		GArray<Sphere3D> spheres,
		GArray<NeighborConstraints> nbc,
		int start_sphere,
		int start_box,
		Real radius,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;

		int i = nbc[pId].idx1;
		int j = nbc[pId].idx2;
		//return;
		Coord pos_i = pos[i];
		if (j >= start_box)
		{
			Point3D p3d(pos[i]);
			Box3D b3d(boxes[j - start_box]);

			if (p3d.distance(b3d) < radius)
			{
				b3d.extent[0] += radius;
				b3d.extent[1] += radius;
				b3d.extent[2] += radius;

				bool tmp;
				p3d = p3d.project(b3d);

				pos[i] = p3d.origin;
				vel[i] += (p3d.origin - pos_i) / dt;
			}
		}
		else
		{

		}

	}

	template<typename TDataType>
	UnifiedFluidRigidConstraint<TDataType>::UnifiedFluidRigidConstraint()
		: Node()
		
	{
		m_nbrQueryElement = this->template addComputeModule<NeighborElementQuery<TDataType>>("neighborhood");
		m_nbrQueryElement->nbr_cons.connect(&m_nbr_cons);
	}

	template<typename TDataType>
	UnifiedFluidRigidConstraint<TDataType>::~UnifiedFluidRigidConstraint()
	{
		
	}

	template<typename TDataType>
	void UnifiedFluidRigidConstraint<TDataType>::pretreat(Real dt)
	{
		if (tmp_rigid_angular_velocity.size() != m_rigid_angular_velocity.getElementCount())
			tmp_rigid_angular_velocity.resize(m_rigid_angular_velocity.getElementCount());

		if (tmp_rigid_velocity.size() != m_rigid_velocity.getElementCount())
			tmp_rigid_velocity.resize(m_rigid_velocity.getElementCount());

		Function1Pt::copy(tmp_rigid_angular_velocity, m_rigid_angular_velocity.getValue());
		Function1Pt::copy(tmp_rigid_velocity, m_rigid_velocity.getValue());

		
		m_nbrQueryElement->compute();
		
		//NEED COLLISION HANDLING
		Real radius = 0.0025;
		int p_num = m_nbrQueryElement->nbr_cons.getElementCount();

		printf("FROM SFI interface %d\n", p_num);

		err_last = -1.0f;

		Real zero = 0;
		if (p_num > 0)
		cuExecute(p_num,
			USFI_collision_rigid_fluid,
			m_particle_position.getValue(),
			m_particle_velocity.getValue(),
			m_shapes->getBoxes(),
			m_shapes->getSpheres(),
			m_nbrQueryElement->nbr_cons.getValue(),
			zero,
			m_shapes->getSpheres().size(),
			radius,
			dt);
		
		
		m_nbrQueryElement->compute();//update contacto point

		m_boundary_forces.setElementCount(m_nbrQueryElement->nbr_cons.getElementCount());
		m_gradient_boundary.setElementCount(m_nbrQueryElement->nbr_cons.getElementCount());
		
		if (m_nbrQueryElement->nbr_cons.getElementCount() == 0) return;

		m_boundary_forces.getValue().reset();
		m_gradient_boundary.getValue().reset();
		delta_force.resize(m_nbrQueryElement->nbr_cons.getElementCount());
		delta_force.reset();

		if (m_arithmetic)
		{
			delete m_arithmetic;
		}
		
		if((m_nbrQueryElement->nbr_cons.getElementCount()) > 0)
			m_arithmetic = Arithmetic<float>::Create(m_nbrQueryElement->nbr_cons.getElementCount());//

	}

	template<typename TDataType>
	void UnifiedFluidRigidConstraint<TDataType>::take_one_iteration_1(Real dt)
	{
		
		//USFI_update_rigid_rigid
		Function1Pt::copy(tmp_rigid_angular_velocity, m_rigid_angular_velocity.getValue());
		Function1Pt::copy(tmp_rigid_velocity, m_rigid_velocity.getValue());

		int b_num = m_nbrQueryElement->nbr_cons.getElementCount();

		/*
		//USFI_update_fluid_rigid
		
		if(b_num > 0)
		cuExecute(b_num,
			USFI_update_fluid_rigid,
			m_boundary_forces.getValue(),
			tmp_rigid_velocity,
			tmp_rigid_angular_velocity,
			m_rigid_position.getValue(),
			m_rigid_mass.getValue(),
			m_rigid_interior.getValue(),
			m_nbrQueryElement->nbr_cons.getValue(),
			sampling_distance,
			restDensity,
			dt);
		*/

		int r_num = this->AA.getElementCount();
		cuExecute(r_num,
			USFI_update_rigid_rigid,
			tmp_rigid_velocity,
			tmp_rigid_angular_velocity,
			AA.getValue(),
			dt);

		//USFI_compute_gradient
		if (b_num > 0)
		cuExecute(b_num,
			USFI_compute_gradient,
			m_boundary_forces.getValue(),
			m_fluid_tmp_vel.getValue(),
			tmp_rigid_velocity,
			tmp_rigid_angular_velocity,
			m_rigid_position.getValue(),
			m_gradient_boundary.getValue(),
			m_nbrQueryElement->nbr_cons.getValue(),
			sampling_distance,
			restDensity,
			dt);
		
	}

	template<typename TDataType>
	void UnifiedFluidRigidConstraint<TDataType>::take_one_iteration_2(Real dt)
	{

		int b_num = m_nbrQueryElement->nbr_cons.getElementCount();
		if (b_num > 0)
		{
			Real rr = m_arithmetic->Dot(m_gradient_boundary.getValue(), m_gradient_boundary.getValue());
			Real err = sqrt(rr / delta_force.size());
			//printf(" ^^^^^^^^^^^^^^^^^^^^^^^^^^^  %d  Boundary err: %.13lf\n:", b_num, err);

			if (err > err_last && err >= 0)
			{
				err_last = err;
				return;
			}
			err_last = err;
			
		}
		//USFI_update_gradient
		
		if (b_num > 0)
		cuExecute(b_num,
			USFI_update_gradient,
			m_boundary_forces.getValue(),
			delta_force,
			m_gradient_point.getValue(),
			m_gradient_boundary.getValue(),
			m_nbrQueryElement->nbr_cons.getValue(),
			sampling_distance,
			restDensity);
		
		if (b_num > 0)
			cuExecute(b_num,
				USFI_update_fluid_rigid,
				delta_force,
				AA.getValue(),
				m_rigid_position.getValue(),
				m_rigid_mass.getValue(),
				m_rigid_interior.getValue(),
				m_nbrQueryElement->nbr_cons.getValue(),
				sampling_distance,
				restDensity,
				dt);
				
	}

	template<typename TDataType>
	void UnifiedFluidRigidConstraint<TDataType>::update(Real dt)
	{
		/*
		//USFI_update_fluid_rigid
		int b_num = m_nbrQueryElement->nbr_cons.getElementCount();
		if (b_num > 0)
		cuExecute(b_num,
			USFI_update_fluid_rigid,
			m_boundary_forces.getValue(),
			AA.getValue(),
			m_rigid_position.getValue(),
			m_rigid_mass.getValue(),
			m_rigid_interior.getValue(),
			m_nbrQueryElement->nbr_cons.getValue(),
			sampling_distance,
			restDensity,
			dt);
			*/
	}


	template<typename TDataType>
	bool UnifiedFluidRigidConstraint<TDataType>::initialize()
	{
		//
		
		m_nbrQueryElement->setDiscreteSet(m_shapes);
		m_nbrQueryElement->inRadius()->setValue(0.075);
		this->m_particle_position.connect(m_nbrQueryElement->inPosition());

		

		return true;
	}
}