#ifdef USE_BUFFER_REFERENCE
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define sizeof(Type) (uint64_t(Type(uint64_t(0))+1))

layout(std430, buffer_reference, buffer_reference_align = 4) buffer BufRef
{
   uint64_t addr;
};

layout(std430, buffer_reference, buffer_reference_align = 4) buffer CharRef
{
   uint8_t buf[4];
};

void ref_buf_copy_align4(uint64_t src_addr, uint64_t dst_addr, uint size) {
   CharRef src = CharRef(src_addr);
   CharRef dst = CharRef(dst_addr);
   for(uint i = 0; i < size; i+=1,src+=1,dst+=1) {
      dst.buf = src.buf;
   }
}
#endif // GL_EXT_buffer_reference

#define CAT(A,B) A##_##B
#define CAT2(A,B) CAT(A,B)
#define CAT3(A,B) CAT2(A,B)
#define CAT4(A,B) CAT3(A,B)