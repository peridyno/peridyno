// https://hwlang.de/algorithmen/sortieren/bitonic/oddn.htm
// https://poniesandlight.co.uk/reflect/bitonic_merge_sort

#define SORT_UP 0
#define SORT_DOWN 1

#define SORT_DIRECTION regs.direction
#define SORT_LOCAL_SIZE gl_WorkGroupSize.x*2

// fucntion point
void prepare_local(in int local_idx, in int global_idx);
void output_global(in int global_idx, in int local_idx);
void compare_swap_global(in int idx1, in int idx2);
void compare_swap_local(in int idx1, in int idx2);


layout(local_size_x = 64) in;
// ENUM for uniform::Parameters.algorithm:
#define eLocalBms      0
#define eLocalDisperse 1
#define eBigFlip       2
#define eBigDisperse   3

layout(push_constant) uniform Push {
	uint n;
    uint h;
    uint algorithm;
    uint direction;
} regs;

void global_compare_and_swap(ivec2 idx){
    if(idx.y >= regs.n || idx.x >= regs.n) return;
    compare_swap_global(idx.x, idx.y);
}

void local_compare_and_swap(ivec2 idx){
    uint offset = gl_WorkGroupSize.x * 2 * gl_WorkGroupID.x;
    if(idx.y + offset >= regs.n || idx.x + offset>= regs.n) return;
    compare_swap_local(idx.x, idx.y);
}

void big_flip(in uint h) {
	uint t_prime = gl_GlobalInvocationID.x;
	uint half_h = h >> 1;
	uint q       = ((2 * t_prime) / h) * h;
	uint x       = q     + (t_prime % half_h);
	uint y       = q + h - (t_prime % half_h) - 1; 
	global_compare_and_swap(ivec2(x,y));
}

void big_disperse( in uint h ) {
	uint t_prime = gl_GlobalInvocationID.x;
	uint half_h = h >> 1;
	uint q       = ((2 * t_prime) / h) * h;
	uint x       = q + (t_prime % (half_h));
	uint y       = q + (t_prime % (half_h)) + half_h;
	global_compare_and_swap(ivec2(x,y));

}

void local_flip(in uint h){
		uint t = gl_LocalInvocationID.x;
		barrier();
		uint half_h = h >> 1;
		ivec2 indices = 
			ivec2( h * ( ( 2 * t ) / h ) ) +
			ivec2( t % half_h, h - 1 - ( t % half_h ) );
		local_compare_and_swap(indices);
}

void local_disperse(in uint h){
	uint t = gl_LocalInvocationID.x;
	for ( ; h > 1 ; h /= 2 ) {
		barrier();
		uint half_h = h >> 1;
		ivec2 indices = 
			ivec2( h * ( ( 2 * t ) / h ) ) +
			ivec2( t % half_h, half_h + ( t % half_h ) );
		local_compare_and_swap(indices);
	}
}

void local_bitonic_merge_sort(uint h){
	uint t = gl_LocalInvocationID.x;
	for (uint hh = 2; hh <= h; hh <<= 1) {
		local_flip(hh);
		local_disperse(hh/2);
	}
}

void sort() {
   	int t = int(gl_LocalInvocationID.x);
	int offset = int(gl_WorkGroupSize.x * 2 * gl_WorkGroupID.x);

	if (regs.algorithm <= eLocalDisperse) {
		if (offset+t*2 < regs.n) {
            prepare_local(t*2, offset+t*2);
		}
		if (offset+t*2+1 < regs.n) {
            prepare_local(t*2+1, offset+t*2+1);
		}
	}
	switch (regs.algorithm){
		case eLocalBms:
			local_bitonic_merge_sort(regs.h);
		break;
		case eLocalDisperse:
			local_disperse(regs.h);
		break;
		case eBigFlip:
			big_flip(regs.h);
		break;
		case eBigDisperse:
			big_disperse(regs.h);
		break;
	}
    // Write local memory back to buffer in case we pulled in the first place.
	if (regs.algorithm <= eLocalDisperse){
		barrier();
		// push to global memory
		if (offset+t*2 < regs.n) {
            output_global(offset+t*2, t*2);
		}
		if (offset+t*2+1 < regs.n) {
            output_global(offset+t*2 + 1, t*2 + 1);
		}
	}
}