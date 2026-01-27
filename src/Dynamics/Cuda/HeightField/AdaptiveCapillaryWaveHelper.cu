#include "AdaptiveCapillaryWaveHelper.h"
#include "HeightField/Module/NumericalScheme.h"

namespace dyno
{
	template <typename Coord>
	__device__ Coord ACWH_VerticalPotential(Coord gp, float GRAVITY)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));

		Coord G;
		G.x = v * h;
		G.y = uh * v;
		G.z = vh * v + GRAVITY * h * h;
		G.w = 0.0f;
		return G;
	}

	template <typename Coord>
	__device__ Coord ACWH_HorizontalPotential(Coord gp, float GRAVITY)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));

		Coord F;
		F.x = u * h;
		F.y = uh * u + GRAVITY * h * h;
		F.z = vh * u;
		F.w = 0.0f;
		return F;
	}

	template <typename Coord4D>
	__device__ Coord4D ACWH_Interpolation(Coord4D pi, Coord4D pj, AdaptiveGridNode2D ni, AdaptiveGridNode2D nj)
	{
		float w = 0.5f;
		w = ni.m_level > nj.m_level ? 2.0f / 3.0f : w;
		w = ni.m_level < nj.m_level ? 1.0f / 3.0f : w;

		return w * pi + (1 - w) * pj;
	}

	template <typename Coord4D>
	__global__ void ACWH_OneWaveStepVersion1(
		DArray<Coord4D> grid_next,
		DArray<Coord4D> grid,
		DArray<AdaptiveGridNode2D> leaves,
		DArrayList<int> neighbors,
		uint level_max,
		float GRAVITY,
		float timestep)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;

		if (i < grid.size())
		{
			Coord4D center = grid[i];

			AdaptiveGridNode2D node_i = leaves[i];

			Coord4D u_center = center;

			auto& list_x_minus = neighbors[4 * i];
			auto& list_x_plus = neighbors[4 * i + 1];
			auto& list_y_minus = neighbors[4 * i + 2];
			auto& list_y_plus = neighbors[4 * i + 3];

			float l_i = pow(2.0f, level_max - node_i.m_level);

			for (uint ne = 0; ne < list_x_minus.size(); ne++)
			{
				int j = list_x_minus[ne];
				Coord4D west = grid[j];

				AdaptiveGridNode2D node_j = leaves[j];

				float l_j = pow(2.0f, level_max - node_j.m_level);

				Coord4D u_west = ACWH_Interpolation(center, west, node_i, node_j) - timestep * (ACWH_HorizontalPotential(center, GRAVITY) - ACWH_HorizontalPotential(west, GRAVITY)) / (0.5f * l_i + 0.5f * l_j);

				Coord4D flux = list_x_plus.size() > 0 ? timestep * ACWH_HorizontalPotential(u_west, GRAVITY) / list_x_minus.size() : Coord4D(0);

				u_center += flux / l_i;
			}

			for (uint ne = 0; ne < list_x_plus.size(); ne++)
			{
				int j = list_x_plus[ne];
				Coord4D east = grid[j];

				AdaptiveGridNode2D node_j = leaves[j];

				float l_j = pow(2.0f, level_max - node_j.m_level);

				Coord4D u_east = ACWH_Interpolation(center, east, node_i, node_j) - timestep * (ACWH_HorizontalPotential(east, GRAVITY) - ACWH_HorizontalPotential(center, GRAVITY)) / (0.5f * l_i + 0.5f * l_j);

				Coord4D flux = list_x_minus.size() > 0 ? timestep * ACWH_HorizontalPotential(u_east, GRAVITY) / list_x_plus.size() : Coord4D(0);

				u_center -= flux / l_i;
			}

			for (uint ne = 0; ne < list_y_minus.size(); ne++)
			{
				int j = list_y_minus[ne];
				Coord4D north = grid[j];

				AdaptiveGridNode2D node_j = leaves[j];

				float l_j = pow(2.0f, level_max - node_j.m_level);

				Coord4D u_north = ACWH_Interpolation(center, north, node_i, node_j) - timestep * (ACWH_VerticalPotential(center, GRAVITY) - ACWH_VerticalPotential(north, GRAVITY)) / (0.5f * l_i + 0.5f * l_j);

				Coord4D flux = list_y_plus.size() > 0 ? timestep * ACWH_VerticalPotential(u_north, GRAVITY) / list_y_minus.size() : Coord4D(0);

				u_center += flux / l_i;
			}

			for (uint ne = 0; ne < list_y_plus.size(); ne++)
			{
				int j = list_y_plus[ne];
				Coord4D south = grid[j];

				AdaptiveGridNode2D node_j = leaves[j];

				float l_j = pow(2.0f, level_max - node_j.m_level);

				Coord4D u_south = ACWH_Interpolation(center, south, node_i, node_j) - timestep * (ACWH_VerticalPotential(south, GRAVITY) - ACWH_VerticalPotential(center, GRAVITY)) / (0.5f * l_i + 0.5f * l_j);

				Coord4D flux = list_y_minus.size() > 0 ? timestep * ACWH_VerticalPotential(u_south, GRAVITY) / list_y_plus.size() : Coord4D(0);

				u_center -= flux / l_i;
			}

			u_center.x = max(0.0f, u_center.x);

			grid_next[i] = u_center;
		}
	}

	template<typename TDataType>
	void AdaptiveCapillaryWaveHelper<TDataType>::ACWHelper_OneWaveStepVersion1(
		DArray<Coord4D>& grid_next,
		DArray<Coord4D>& grid,
		DArray<AdaptiveGridNode2D>& leaves,
		DArrayList<int>& neighbors,
		uint level_max,
		float GRAVITY,
		float timestep)
	{
		cuExecute(grid_next.size(),
			ACWH_OneWaveStepVersion1,
			grid_next,
			grid,
			leaves,
			neighbors,
			level_max,
			GRAVITY,
			timestep);
	}

	template <typename Coord>
	__device__ void ACWH_FixShore(Coord& c, Coord& r)
	{
		if (r.x <= 0)
		{
			r.x = 0;
			r.y = 0;
			r.z = 0;// r.w > c.x + c.w ? -c.z : 0;
			r.w = r.w > c.x + c.w ? c.x + c.w : r.w;
		}
	}

	template <typename Coord>
	__device__ Coord ACWH_HorizontalSlope(Coord gp, float H, float GRAVITY)
	{
		Coord F;
		F.x = 0.0f;// u* h;
		F.y = GRAVITY * H * (gp.x + gp.w);
		F.z = 0.0f;
		F.w = 0.0f;
		return F;
	}

	template <typename Coord>
	__device__ Coord ACWH_VerticalSlope(Coord gp, float H, float GRAVITY)
	{
		Coord G;
		G.x = 0.0f;
		G.y = 0.0f;
		G.z = GRAVITY * H * (gp.x + gp.w);
		G.w = 0.0f;
		return G;
	}

	template <typename Coord4D>
	__global__ void ACWH_OneWaveStepVersion2(
		DArray<Coord4D> grid_next,
		DArray<Coord4D> grid,
		DArray<AdaptiveGridNode2D> leaves,
		DArrayList<int> neighbors,
		uint level_max,
		float GRAVITY,
		float timestep)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;

		if (i < grid.size())
		{
			Coord4D center = grid[i];

			AdaptiveGridNode2D node_i = leaves[i];

			Real H = max(center.x, 0.0f);
			Coord4D u_center = center;

			auto& list_x_minus = neighbors[4 * i];
			auto& list_x_plus = neighbors[4 * i + 1];
			auto& list_y_minus = neighbors[4 * i + 2];
			auto& list_y_plus = neighbors[4 * i + 3];

			float l_i = pow(2.0f, level_max - node_i.m_level);

			for (uint ne = 0; ne < list_x_minus.size(); ne++)
			{
				int j = list_x_minus[ne];
				Coord4D west = grid[j];
				ACWH_FixShore(center, west);

				//AdaptiveGridNode2D node_j = leaves[j];

				//float l_j = pow(2.0f, level_max - node_j.m_level);

				Coord4D flux = list_x_plus.size() > 0 ? timestep * (CentralUpwindX(west, center, GRAVITY) + ACWH_HorizontalSlope(west, H, GRAVITY)) / list_x_minus.size() : Coord4D(0);

				u_center += flux / l_i;
			}

			for (uint ne = 0; ne < list_x_plus.size(); ne++)
			{
				int j = list_x_plus[ne];
				Coord4D east = grid[j];
				ACWH_FixShore(center, east);

				//AdaptiveGridNode2D node_j = leaves[j];

				//float l_j = pow(2.0f, level_max - node_j.m_level);

				Coord4D flux = list_x_minus.size() > 0 ? timestep * (CentralUpwindX(center, east, GRAVITY) + ACWH_HorizontalSlope(east, H, GRAVITY)) / list_x_plus.size() : Coord4D(0);

				u_center -= flux / l_i;
			}

			for (uint ne = 0; ne < list_y_minus.size(); ne++)
			{
				int j = list_y_minus[ne];
				Coord4D north = grid[j];
				ACWH_FixShore(center, north);

				//AdaptiveGridNode2D node_j = leaves[j];

				//float l_j = pow(2.0f, level_max - node_j.m_level);

				Coord4D flux = list_y_plus.size() > 0 ? timestep * (CentralUpwindY(north, center, GRAVITY) + ACWH_VerticalSlope(north, H, GRAVITY)) / list_y_minus.size() : Coord4D(0);

				u_center += flux / l_i;
			}

			for (uint ne = 0; ne < list_y_plus.size(); ne++)
			{
				int j = list_y_plus[ne];
				Coord4D south = grid[j];
				ACWH_FixShore(center, south);

				//AdaptiveGridNode2D node_j = leaves[j];

				//float l_j = pow(2.0f, level_max - node_j.m_level);

				Coord4D flux = list_y_minus.size() > 0 ? timestep * (CentralUpwindY(center, south, GRAVITY) + ACWH_VerticalSlope(south, H, GRAVITY)) / list_y_plus.size() : Coord4D(0);

				u_center -= flux / l_i;
			}

			if (u_center.x <= EPSILON)
			{
				u_center.x = 0;
				u_center.y = 0;
				u_center.z = 0;
			}
			grid_next[i] = u_center;
		}
	}

	template<typename TDataType>
	void AdaptiveCapillaryWaveHelper<TDataType>::ACWHelper_OneWaveStepVersion2(
		DArray<Coord4D>& grid_next,
		DArray<Coord4D>& grid,
		DArray<AdaptiveGridNode2D>& leaves,
		DArrayList<int>& neighbors,
		uint level_max,
		float GRAVITY,
		float timestep)
	{
		cuExecute(grid_next.size(),
			ACWH_OneWaveStepVersion2,
			grid_next,
			grid,
			leaves,
			neighbors,
			level_max,
			GRAVITY,
			timestep);
	}

	DEFINE_CLASS(AdaptiveCapillaryWaveHelper);
}