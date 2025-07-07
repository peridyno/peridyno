#pragma once

#include <functional>
#include <iostream>
#include <array>
#include <sstream>
#include <string_view>
#include <utility>
#include <iomanip>
#include <cstdint>
#include <cstring> 

namespace Wt
{
	class Guid
	{
	public:
		explicit Guid(const std::array<unsigned char, 16>& bytes);
		explicit Guid(std::array<unsigned char, 16>&& bytes);

		Guid();

		Guid(const Guid& other) = default;
		Guid& operator=(const Guid& other) = default;
		Guid(Guid&& other) = default;
		Guid& operator=(Guid&& other) = default;

		bool operator==(const Guid& other) const;
		bool operator!=(const Guid& other) const;

		std::string str() const;
		operator std::string() const;
		const std::array<unsigned char, 16>& bytes() const;
		void swap(Guid& other);
		bool isValid() const;

	private:
		std::array<unsigned char, 16> _bytes;

		friend std::ostream& operator<<(std::ostream& s, const Guid& guid);
		friend bool operator<(const Guid& lhs, const Guid& rhs);
	};

	Guid newGuid();

	namespace details
	{
		template <typename...> struct hash;

		template<typename T>
		struct hash<T> : public std::hash<T>
		{
			using std::hash<T>::hash;
		};


		template <typename T, typename... Rest>
		struct hash<T, Rest...>
		{
			inline std::size_t operator()(const T& v, const Rest&... rest) {
				std::size_t seed = hash<Rest...>{}(rest...);
				seed ^= hash<T>{}(v)+0x9e3779b9 + (seed << 6) + (seed >> 2);
				return seed;
			}
		};
	}
}

namespace std
{
	// Template specialization for std::swap<Guid>() --
	// See guid.cpp for the function definition
	template <>
	void swap(Wt::Guid& guid0, Wt::Guid& guid1) noexcept;

	// Specialization for std::hash<Guid> -- this implementation
	// uses std::hash<std::string> on the stringification of the guid
	// to calculate the hash
	template <>
	struct hash<Wt::Guid>
	{
		std::size_t operator()(Wt::Guid const& guid) const
		{
			const uint64_t* p = reinterpret_cast<const uint64_t*>(guid.bytes().data());
			return Wt::details::hash<uint64_t, uint64_t>{}(p[0], p[1]);
		}
	};
}