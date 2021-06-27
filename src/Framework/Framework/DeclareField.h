#pragma once
//----------------------------------------------------------------------------
// Check for unsupported old compilers.
#if defined(_MSC_VER) && _MSC_VER < 1800
# error Peridyno requires MSVC++ 12.0 aka Visual Studio 2013 or newer
#endif

#if !defined(__clang__) && defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8))
# error Peridyno requires GCC 4.8 or newer
#endif

namespace dyno
{

#define DEF_VAR(T, name, value, desc) \
private:									\
	VarField<T> var_##name = VarField<T>(T(value), std::string(#name), desc, FieldTypeEnum::Param, this);			\
public:										\
	inline VarField<T>* var##name() {return &var_##name;}

#define DEF_VAR_IN(T, name, desc) \
private:									\
	VarField<T> in_##name = VarField<T>(std::string(#name), desc, FieldTypeEnum::In, this);			\
public:										\
	inline VarField<T>* in##name() {return &in_##name;}

#define DEF_VAR_OUT(T, name, desc) \
private:									\
	VarField<T> out_##name = VarField<T>(std::string(#name), desc, FieldTypeEnum::Out, this);			\
public:									\
	inline VarField<T>* out##name() {return &out_##name;}

/**
*	Macro definition for input/output of type Array
*/
#define DEF_ARRAY_IN(T, name, device, desc) \
private:									\
	ArrayField<T, device> in_##name = ArrayField<T, device>(std::string(#name), desc, FieldTypeEnum::In, this);	\
public:									\
	inline ArrayField<T, device>* in##name() {return &in_##name;}

#define DEF_ARRAY_OUT(T, name, device, desc) \
private:									\
	ArrayField<T, device> out_##name = ArrayField<T, device>(std::string(#name), desc, FieldTypeEnum::Out, this);	\
public:									\
	inline ArrayField<T, device>* out##name() {return &out_##name;}


/**
* Macro definition for input/output of type ArrayList
*/
#define DEF_ARRAYLIST_IN(T, name, device, desc)			\
private:												\
	ArrayListField<T, device> in_##name = ArrayListField<T, device>(std::string(#name), desc, FieldTypeEnum::In, this);	\
public:													\
	inline ArrayListField<T, device>* in##name() {return &in_##name;}

#define DEF_ARRAYLIST_OUT(T, name, device, desc)		\
private:												\
	ArrayListField<T, device> out_##name = ArrayListField<T, device>(std::string(#name), desc, FieldTypeEnum::Out, this);	\
public:													\
	inline ArrayListField<T, device>* out##name() {return &out_##name;}
}