#include "GranularMedia.h"

namespace dyno
{
#define EPSILON 0.000001

	template<typename TDataType>
	GranularMedia<TDataType>::GranularMedia()
		: Node()
	{
		this->setDt(0.033f);

		auto heights = std::make_shared<HeightField<TDataType>>();
		this->stateHeightField()->setDataPtr(heights);
	}

	template<typename TDataType>
	GranularMedia<TDataType>::~GranularMedia()
	{
	}

	template<typename Real, typename Coord4D>
	__device__ Coord4D d_flux_x(Coord4D gpl, Coord4D gpr, Real GRAVITY)
	{
		Real h = maximum(0.5f * (gpl.x + gpr.x), 0.0f);
		Real b = 0.5f * (gpl.w + gpr.w);

		Real hl = maximum(gpl.x, 0.0f);
		Real hl4 = hl * hl * hl * hl;
		Real ul = sqrtf(2.0f) * hl * gpl.y / (sqrtf(hl4 + maximum(hl4, Real(EPSILON))));

		Real hr = max(gpr.x, 0.0f);
		Real hr4 = hr * hr * hr * hr;
		Real ur = sqrtf(2.0f) * hr * gpr.y / (sqrtf(hr4 + maximum(hr4, Real(EPSILON))));

		if (hl < EPSILON && hr < EPSILON)
		{

			return Coord4D(0.0f);
		}

		Real a_plus;
		Real a_minus;
		a_plus = maximum(maximum(Real(ul + sqrtf(GRAVITY * (gpl.x/*+gpl.w*/))), Real(ur + sqrtf(GRAVITY * (gpr.x/*+gpr.w*/)))), Real(0));
		a_minus = minimum(minimum(Real(ul - sqrtf(GRAVITY * (gpl.x/*+gpl.w*/))), Real(ur - sqrtf(GRAVITY * (gpr.x/*+gpr.w*/)))), Real(0));

		Coord4D delta_U = gpr - gpl;
		if (gpl.x > EPSILON && gpr.x > EPSILON)
		{
			delta_U.x += delta_U.w;
		}

		delta_U.w = 0.0f;

		Coord4D Fl = Coord4D(gpl.y, gpl.y * ul, gpl.z * ul, 0.0f);
		Coord4D Fr = Coord4D(gpr.y, gpr.y * ur, gpr.z * ur, 0.0f);

		Coord4D re = (a_plus * Fl - a_minus * Fr) / (a_plus - a_minus) + a_plus * a_minus / (a_plus - a_minus) * delta_U;

		if (ul == 0 && ur == 0)//abs(ul) <EPSILON && abs(ur) <EPSILON
		{
			re.x = 0;
			re.y = 0;
			re.z = 0;
		}
		return re;
	}

	template<typename Real, typename Coord4D>
	__device__ Coord4D d_flux_y(Coord4D gpl, Coord4D gpr, Real GRAVITY)
	{
		Real hl = max(gpl.x, 0.0f);
		Real hl4 = hl * hl * hl * hl;
		Real vl = sqrtf(2.0f) * hl * gpl.z / (sqrtf(hl4 + max(hl4, EPSILON)));

		Real hr = max(gpr.x, 0.0f);
		Real hr4 = hr * hr * hr * hr;
		Real vr = sqrtf(2.0f) * hr * gpr.z / (sqrtf(hr4 + max(hr4, EPSILON)));

		if (hl < EPSILON && hr < EPSILON)
		{
			return Coord4D(0.0f);
		}

		Real a_plus = maximum(maximum(Real(vl + sqrtf(GRAVITY * (gpl.x/* + gpl.w*/))), Real(vr + sqrtf(GRAVITY * (gpr.x/* + gpr.w*/)))), Real(0));
		Real a_minus = minimum(minimum(Real(vl - sqrtf(GRAVITY * (gpl.x/* + gpl.w*/))), Real(vr - sqrtf(GRAVITY * (gpr.x/* + gpr.w*/)))), Real(0));

		Real b = 0.5f * (gpl.w + gpr.w);

		Coord4D delta_U = gpr - gpl;
		if (gpl.x > EPSILON && gpr.x > EPSILON)
		{
			delta_U.x += delta_U.w;
		}
		delta_U.w = 0.0f;

		Coord4D Fl = Coord4D(gpl.z, gpl.y * vl, gpl.z * vl, 0.0f);
		Coord4D Fr = Coord4D(gpr.z, gpr.y * vr, gpr.z * vr, 0.0f);

		Coord4D re = (a_plus * Fl - a_minus * Fr) / (a_plus - a_minus) + a_plus * a_minus / (a_plus - a_minus) * delta_U;

		if (vl == 0 && vr == 0)
		{
			re.x = 0;
			re.y = 0;
			re.z = 0;
		}
		return re;
	}

	//Section 4.1
	template<typename Coord4D>
	__global__ void GM_Advection(
		DArray2D<Coord4D> grid_next,
		DArray2D<Coord4D> grid,
		Real GRAVITY,
		Real timestep)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		uint width = grid.nx();
		uint height = grid.ny();

		if (x < width - 2 && y < height - 2)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			Coord4D center = grid(gridx, gridy);
			Coord4D north = grid(gridx, gridy - 1);
			Coord4D west = grid(gridx - 1, gridy);
			Coord4D south = grid(gridx, gridy + 1);
			Coord4D east = grid(gridx + 1, gridy);

			Coord4D eastflux = d_flux_x(center, east, GRAVITY);
			Coord4D westflux = d_flux_x(west, center, GRAVITY);
			Coord4D southflux = d_flux_y(center, south, GRAVITY);
			Coord4D northflux = d_flux_y(north, center, GRAVITY);
			Coord4D flux = eastflux - westflux + southflux - northflux;
			Coord4D u_center = center - timestep * flux;

			if (u_center.x < EPSILON)
			{
				u_center.x = 0.0f;
				u_center.y = 0.0f;
				u_center.z = 0.0f;
			}

			Real totalH = u_center.x + center.w;
			if (u_center.x <= EPSILON)
			{
				u_center.y = 0;
				u_center.z = 0;
			}
			if ((east.w >= totalH) || (west.w >= totalH))
			{
				u_center.y = 0;
				//u_center.z = 0;
			}
			if ((north.w >= totalH) || (south.w >= totalH))
			{
				//u_center.y = 0;
				u_center.z = 0;
			}

			if (x == 0 || x == width - 1)
			{
				u_center.y = 0;
				//u_center.z = 0;
			}
			if (y == 0 || y == height - 1)
			{
				//u_center.y = 0;
				u_center.z = 0;
			}
			u_center.w = center.w;

			grid_next(gridx, gridy) = u_center;
		}
	}


	template<typename Coord4D>
	__global__ void GM_SetBoundaryCondition(DArray2D<Coord4D> grid)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		uint width = grid.nx();
		uint height = grid.ny();

		if (x >= width || y >= height) return;

		if (x == 0)	grid(0, y) = grid(1, y);

		if (x == width - 1)	grid(width - 1, y) = grid(width - 2, y);

		if (y == 0)	grid(x, 0) = grid(x, 1);

		if (y == height - 1) grid(x, height - 1) = grid(x, height - 2);
	}

	//Section 4.2
	template<typename Real, typename Coord4D>
	__global__ void GM_UpdateVelocity(
		DArray2D<Coord4D> grid_next,
		DArray2D<Coord4D> grid,
		Real timestep,
		Real depth,
		Real dragging,
		Real GRAVITY,
		Real mu)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		const Real grid_spacing2 = 2;

		uint width = grid.nx();
		uint height = grid.ny();

		if (x < width - 2 && y < height - 2)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			Coord4D center = grid(gridx, gridy);
			Coord4D north = grid(gridx, gridy - 1);
			Coord4D west = grid(gridx - 1, gridy);
			Coord4D south = grid(gridx, gridy + 1);
			Coord4D east = grid(gridx + 1, gridy);

			Real h = center.x;
			Real hu_new = center.y;
			Real hv_new = center.z;

			Real hu_old = hu_new;
			Real hv_old = hv_new;

			Real s = center.x + center.w;
			Real sw = west.x + west.w;
			Real se = east.x + east.w;
			Real sn = north.x + north.w;
			Real ss = south.x + south.w;


			Real  h_d = depth;

			float2 sliding_dir;
			sliding_dir.x = (sw - se) / grid_spacing2;
			sliding_dir.y = (sn - ss) / grid_spacing2;
			Real gradient = sqrtf(sliding_dir.x * sliding_dir.x + sliding_dir.y * sliding_dir.y);

			Real sliding_cos = 1 / sqrtf(1 + gradient * gradient);
			Real sliding_sin = abs(gradient) / sqrtf(1 + gradient * gradient);

			Real sliding_length = sqrtf(sliding_dir.x * sliding_dir.x + sliding_dir.y * sliding_dir.y);

			Real g = GRAVITY;
			Real hu_tmp = hu_old + timestep * g * maximum(minimum(h_d, center.x), Real(0)) * sliding_dir.x;
			Real hv_tmp = hv_old + timestep * g * maximum(minimum(h_d, center.x), Real(0)) * sliding_dir.y;

			Vector<Real, 2> vel_dir;

			Real vel_norm = sqrtf(hu_tmp * hu_tmp + hv_tmp * hv_tmp);
			if (vel_norm < EPSILON)
			{
				vel_dir.x = 0.0f;
				vel_dir.y = 0.0f;
			}
			else
			{
				vel_dir.x = hu_tmp / vel_norm;
				vel_dir.y = hv_tmp / vel_norm;
			}

			hu_new = hu_tmp - timestep * g * maximum(minimum(h_d, center.x), Real(0)) * vel_dir.x * mu;
			hv_new = hv_tmp - timestep * g * maximum(minimum(h_d, center.x), Real(0)) * vel_dir.y * mu;

			if (hu_new * hu_tmp + hv_new * hv_tmp < EPSILON && sliding_sin - mu * sliding_cos < 0)
			{
				hu_new = 0.0f;
				hv_new = 0.0f;
			}


			Coord4D u_center;
			u_center.x = center.x;
			u_center.y = hu_new * dragging;
			u_center.z = hv_new * dragging;
			Real totalH = u_center.x + center.w;
			if (u_center.x <= EPSILON)
			{
				//u_center.x = 0;
				u_center.y = 0;
				u_center.z = 0;
			}
			if ((east.w >= totalH) || (west.w >= totalH))
			{
				u_center.y = 0;
				//u_center.z = 0;
			}
			if ((north.w >= totalH) || (south.w >= totalH))
			{
				//u_center.y = 0;
				u_center.z = 0;
			}

			if (x == 0 || x == width - 1)
			{
				u_center.y = 0;
				//u_center.z = 0;
			}
			if (y == 0 || y == height - 1)
			{
				//u_center.y = 0;
				u_center.z = 0;
			}
			u_center.w = center.w;

			grid_next(gridx, gridy) = u_center;
		}
	}

	template<typename Coord3D, typename Coord4D>
	__global__ void UpdateHeightField(
		DArray2D<Coord3D> heightfield,
		DArray2D<Coord4D> grid)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		int width = grid.nx();
		int height = grid.ny();

		if (i < width - 2 && j < height - 2)
		{
			int gx = i + 1;
			int gy = j + 1;

			Coord4D gxy = grid(gx, gy);

			heightfield(i, j) = Coord3D(0, gxy.x, 0);
		}
	}

	template<typename TDataType>
	void GranularMedia<TDataType>::resetStates()
	{
		uint w = this->varWidth()->getValue();
		uint h = this->varHeight()->getValue();

		uint exW = w + 2;
		uint exH = h + 2;

		auto s = this->varSpacing()->getValue();
		auto o = this->varOrigin()->getValue();

		this->stateGrid()->resize(exW, exH);
		this->stateGridNext()->resize(exW, exH);

		CArray2D<Coord4D> initializer(exW, exH);

		for (uint i = 0; i < exW; i++)
		{
			for (uint j = 0; j < exH; j++)
			{
				if (abs(i - exW / 2) < 10 && abs(j - exW / 2) < 10)
				{
					initializer(i, j) = Coord4D(5, 0, 0, 0);
				}
				else
					initializer(i, j) = Coord4D(0, 0, 0, 0);
			}
		}

		this->stateGrid()->assign(initializer);
		this->stateGridNext()->assign(initializer);

		auto topo = this->stateHeightField()->getDataPtr();
		topo->setExtents(w, h);
		topo->setGridSpacing(s);
		topo->setOrigin(o);
		topo->update();

		auto hf = this->stateHeightField()->getDataPtr();

		auto& disp = hf->getDisplacement();

		auto& grid = this->stateGrid()->getData();

		cuExecute2D(make_uint2(grid.nx(), grid.ny()),
			UpdateHeightField,
			disp,
			grid);
	}

	template<typename TDataType>
	void GranularMedia<TDataType>::updateStates()
	{
		auto& grid = this->stateGrid()->getData();
		auto& grid_next = this->stateGridNext()->getData();

		uint2 dim = make_uint2(grid.nx(), grid.ny());

		Real dt = this->stateTimeStep()->getValue();
		Real depth = this->varDepth()->getValue();
		Real dragging = this->varCoefficientOfDragForce()->getValue();
		Real mu = this->varCoefficientOfFriction()->getValue();

		Real G = abs(this->varGravity()->getValue());

		cuExecute2D(dim,
			GM_Advection,
			grid_next,
			grid,
			G,
			dt);

		cuExecute2D(dim,
			GM_SetBoundaryCondition,
			grid_next);

		cuExecute2D(
			dim,
			GM_UpdateVelocity,
			grid,
			grid_next,
			dt,
			depth,
			dragging,
			G,
			mu);

		auto hf = this->stateHeightField()->getDataPtr();

		auto& disp = hf->getDisplacement();

		cuExecute2D(dim,
			UpdateHeightField,
			disp,
			grid_next);
	}

	DEFINE_CLASS(GranularMedia);
}

