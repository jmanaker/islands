#pragma once
#include <cstdint>
#include <bitset>
#include <functional>
#include <memory>
#include <valarray>
#include <vector>

#include "LinAlgCommon.hpp"
#include "Tweaks.hpp"

namespace lin_alg2
{
    using namespace lin_alg;
    template<std::size_t dim> struct slice_fixed_dim;
    typedef slice_fixed_dim<1> slice;
    struct nocopy_tag {};
    struct slice_axis
    {
        std::ptrdiff_t stride;
        std::size_t len;
        constexpr std::ptrdiff_t slice(all_t) noexcept { return 0; }
        constexpr std::ptrdiff_t slice(lin_alg2::slice const &) noexcept;
    };
    struct gslice
    {
        std::ptrdiff_t start{ 0 };
        std::size_t len;
#pragma warning(suppress: 4200) //"Nonstandard extension: zero-sized array in struct/union"
        slice_axis indices[];
    private:
        constexpr auto shrink_to(slice_axis *dest) const noexcept
        {
            for (slice_axis const *cur_loc{ indices }, *stop{ cur_loc + len }; cur_loc < stop; ++cur_loc)
                if (1 != cur_loc->len)
                    *dest++ = *cur_loc;
            return dest;
        }
        template<std::size_t dim>
        friend struct slice_fixed_dim;
    public:
        gslice(void) noexcept = default;
        constexpr gslice(std::ptrdiff_t start, std::size_t len,
            std::initializer_list<slice_axis> indices = {}) noexcept
            : start{ start }, len{ len }, indices{ indices }
        {
            assert(size(indices) <= len);
        }
        constexpr gslice(gslice const &other, nocopy_tag) noexcept : gslice(other.start, other.len) {}
        gslice(gslice const &other) : gslice(other, nocopy_tag{})
        {
            std::uninitialized_move_n(other.indices, len, indices);
        }
        constexpr auto shrink(void) noexcept { return std::exchange(len, shrink_to(indices) - indices); }
        constexpr void slice(auto mask_start)
        {
            for (slice_axis *cur_loc{ indices }, *stop{ cur_loc + len }; cur_loc < stop; ++cur_loc)
                start += cur_loc->slice(*mask_start++);
        }
        constexpr [[nodiscard]] operator std::size_t(void) const noexcept
        {
            for (slice_axis const *cur_loc{ indices }, *stop{ cur_loc + len }; cur_loc < stop; ++cur_loc)
                assert(1 == cur_loc->len);
            return start;
        }
    };
#define ALLOCA_GSLICE(start, len, ...) \
    new(_alloca(sizeof(gslice) + len * sizeof(slice_axis))) gslice{start, len, __VA_ARGS__ }
#define NEW_GSLICE(start, len, ...) \
    new(::operator new(sizeof(gslice) + len * sizeof(slice_axis))) gslice{start, len, __VA_ARGS__ }

    template<std::size_t const dim>
    struct slice_fixed_dim : private _internal::_insert_overloads<slice_fixed_dim<dim>, slice, dim>
    {
    private:
        union
        {
            gslice storage;
            char fig_leaf;
        };
    public:
        slice_axis data[dim];
        template<typename...Args>
        constexpr void slice(Args &&...args) noexcept
        {
            static_assert(dim == sizeof...(args));
            auto loc{ data };
            //Can't fold on + b/c then increments to loc are unsequenced
            (..., (storage.start += (loc++)->slice(args)));
        }
        template<std::size_t... indices>
        constexpr [[nodiscard]] auto op_bracket(auto &&masks, std::index_sequence<indices...>)
        {
            auto copy{ *this };
            copy.slice(get<indices>(masks)...);
            return copy;
        }
        constexpr explicit(dim != 0)
            slice_fixed_dim(std::ptrdiff_t start = 0, slice_axis(&&data)[dim] = {})
            noexcept
            : storage{start, dim}, data{ data }
        {}
        constexpr slice_fixed_dim(slice_fixed_dim const &other) noexcept : storage{ other } {}
        constexpr explicit slice_fixed_dim(gslice const &other) noexcept
            : slice_fixed_dim(other.start)
        {
            using std::end;
            auto const dest{ other.shrink_to(data) }, stop{ end(data) };
            assert(dest < stop);
            std::uninitialized_fill(dest, stop, { 0, 1 });
        }
        template<std::ptrdiff_t const... locs>
        friend constexpr [[nodiscard]] auto coerce(slice_fixed_dim const &me) noexcept
        {
            //Can't quite use shrink/etc., because might drop indices outside `locs...`
            slice_fixed_dim<dim - sizeof...(locs)> retval;
            std::ptrdiff_t index{ 0 };
            auto dest{ retval.data };
            for (auto const &axis : me.data)
            {
                if (((locs == index) || ...))
                    assert(1 == axis.len);
                else *dest++ = axis;
                ++index;
            }
            return retval;
        }

        constexpr [[nodiscard]] auto &as_gslice(void) noexcept { return storage; }
        constexpr [[nodiscard]] auto const &as_gslice(void) const noexcept
        {
            return const_cast<slice_fixed_dim *>(this)->as_gslice();
        }
        constexpr [[nodiscard]] operator gslice &(void) noexcept { return as_gslice(); }
        constexpr [[nodiscard]] operator gslice const &(void) const noexcept { return as_gslice(); }
        constexpr [[nodiscard]] operator std::size_t(void) const noexcept { return as_gslice(); }
    };
    constexpr inline [[nodiscard]] std::ptrdiff_t slice_axis::slice(lin_alg2::slice const &other) noexcept
    {
        gslice const &_other(other);
        auto const retval{ _other.start * stride };
        auto &other_sl{ *_other.indices };
        stride *= other_sl.stride;
        len = other_sl.len;
        return retval;
    }

    class gslice_iter : public containers::wrap_iter<gslice_iter>
    {
        std::ptrdiff_t offset{ 0 };
        slice_axis &axis;
        std::size_t old_len;
    protected:
        gslice &backing;
    private:
        friend struct wrap_iter<gslice_iter>;
        constexpr auto &op_shift(std::ptrdiff_t shift) & noexcept
        {
            offset += shift;
            assert(0 <= offset);
            backing.start += shift * axis.stride;
            old_len -= shift;
            assert(0 <= axis.len);
            return *this;
        }
    public:
        gslice_iter(gslice &backing, std::ptrdiff_t axis) noexcept 
            : backing{ backing }, axis{ backing.indices[axis] }, old_len{std::exchange(this->axis.len, 1)}
        {}
        ~gslice_iter(void)
        {
            *this -= offset;
            std::swap(axis.len, old_len);
        }
        [[nodiscard]] auto const &operator *(void) const { return backing; }
    };
}