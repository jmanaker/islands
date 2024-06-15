#pragma once
#include <cassert>
#include <cstdint>
#include <compare>
#include <utility>

#include "Tweaks.hpp"
#include "LinAlgCommon.hpp"

namespace lin_alg3
{
    using namespace lin_alg;
    namespace _internal
    {
        using namespace lin_alg::_internal;
        enum check { yes = true, no = false };
    }

    constexpr std::size_t const unknown_bound{ 0 };
    template<std::size_t dim=1> struct slice;
    struct slice_axis
    {
        std::size_t len;
        std::ptrdiff_t stride;
        constexpr slice_axis(void) noexcept
//#ifdef NDEBUG
            = default;
//#else
//            : slice_axis(1) {}
//#endif
        //Necessary for implicit conversion from std::ptrdiff_t
        constexpr slice_axis(std::size_t len, std::ptrdiff_t stride = 1) noexcept
            : len{ len }, stride{ stride }
        {}
        constexpr [[nodiscard]] std::ptrdiff_t subslice(all_t) noexcept { return 0; }
        constexpr [[nodiscard]] std::ptrdiff_t subslice(slice<> const &other) noexcept;
    };
    [[nodiscard]] bool operator==(slice_axis const &lhs, slice_axis const &rhs)
    {
        return !std::memcmp(&lhs, &rhs, sizeof(lhs));
    }
    template<std::size_t dim>
    class slice_iter;
    template<std::size_t dim>
    struct slice :
        public _internal::tensor_base<slice<dim>, dim>,
        //private <== friend declaration troubles
        _internal::_insert_overloads<slice<dim>, slice<>, dim>
    {
    private:
        template<std::size_t dim>
        friend class slice_iter;
    public:
        std::ptrdiff_t start;
        slice_axis indices[dim];
        //private:
        template<std::size_t... indices>
        constexpr /*[[nodiscard]]*/ auto op_bracket(auto &&masks, std::index_sequence<indices...>)
        {
            auto copy{ *this };
            copy.subslice(get<indices>(masks)...);
            return copy;
        }
    private:
        template<_internal::check should_check, std::ptrdiff_t... offsets>
        auto coerce_to(slice_axis *dest) const noexcept
        {
            //Can't quite use c'tor, because might drop indices outside `locs...`
            std::ptrdiff_t index{ 0 };
            for (auto const &axis : indices)
            {
                if (((offsets == index) || ...))
                {
                    if constexpr (should_check)
                        assert(1 == axis.len);
                }
                else
                    *dest++ = axis;
                ++index;
            }
            return start;
        }
    public:
        constexpr explicit slice(std::ptrdiff_t start) noexcept : start{ start } {}
        constexpr slice(void) noexcept : slice(0) {}
        //constexpr explicit slice(std::size_t start) noexcept : slice(static_cast<std::ptrdiff_t>(start)) {}
        constexpr slice(std::ptrdiff_t start, slice_axis(&&indices)[dim]) noexcept : slice(start)
        {
            using std::begin, std::end;
            std::move(begin(indices), end(indices), begin(this->indices));
        }
        constexpr slice(slice const &) noexcept = default;
        template<std::size_t old_dim>
        constexpr explicit slice(slice<old_dim> const &other) : start{ other.start }
        {
            using std::end;
            std::ptrdiff_t index{ 0 };
            auto dest{ indices };
            for (auto const &axis : other.indices)
                if (1 != axis.len)
                    *dest++ = axis;
            if constexpr (dim > 0)
            {
                auto const stop{ end(indices) };
                assert(dest < stop);
                std::fill(dest, stop, { 1, 0 });
            }
        }
        template<std::ptrdiff_t const... locs>
        friend constexpr [[nodiscard]] auto coerce(slice const &me) noexcept
        {
            using namespace _internal;
            static_assert(dim > std::max({ locs... }));
            constexpr auto const remaining_dim{ dim - sizeof...(locs) };
            slice<remaining_dim> retval;
            //union
            //{
            //    slice<remaining_dim> retval{};
            //    char fig_leaf; //If remaining_dim is 0, retval's last elem is a flexible array
            //};
            retval.start = me.coerce_to<check::yes, locs...>(retval.indices);
#pragma warning(suppress:4305) //Converting std::size_t to bool
            if constexpr (remaining_dim) return retval; else return retval.start;
        }
        template<typename...Args>
        constexpr void subslice(Args &&...args) noexcept
        {
            if constexpr(dim>0) //unknown bound ==> no check
                static_assert(dim == sizeof...(args));

            auto loc{ indices };
            //Can't fold on + b/c then increments to loc are unsequenced
            //Right fold matches right-associative comma operator
            (..., (start += (loc++)->subslice(args)));
        }
        constexpr [[nodiscard]] auto to_scalar(void) const noexcept
        {
            for (auto const &axis : indices)
                assert(1 == axis.len);
            return start;
        }
        constexpr explicit [[nodiscard]] operator std::ptrdiff_t(void) const noexcept
        {
            return to_scalar();
        }
        template<std::ptrdiff_t a, std::ptrdiff_t b>
        constexpr friend [[nodiscard]] auto transpose(slice obj) noexcept
        {
            std::swap(obj.indices[a], obj.indices[b]);
            return std::move(obj);
        }
        template<std::size_t offset=0>
        constexpr [[nodiscard]] auto begin(void) const { return slice_iter<dim>::make_at<offset>(*this); };
        template<std::size_t offset=0>
        constexpr [[nodiscard]] auto cbegin(void) const { return begin<offset>(); }
        template<std::size_t offset=0>
        constexpr [[nodiscard]] auto end(void) const { return begin<offset>() + indices[offset].len; }
        template<std::size_t offset=0>
        constexpr [[nodiscard]] auto cend(void) const { return end<offset>(); };
    };
    constexpr slice const all{ 0,{-1} };

    constexpr [[nodiscard]] std::ptrdiff_t slice_axis::subslice(slice<> const &other) noexcept
    {
        auto const retval{ other.start * stride };
        auto const &other_sl{ *other.indices };
        if (all.indices->len != other_sl.len)
        {
            auto const old_len{ std::exchange(len, other_sl.len) };
            assert(len <= old_len);
        }
        stride *= other_sl.stride;
        return retval;
    }
    constexpr [[nodiscard]] auto transpose(slice<2> matrix) noexcept
    {
        return transpose<0, 1>(std::move(matrix));
    }

    template<std::size_t dim>
    class slice_iter : public containers::wrap_iter<slice_iter<dim>>
    {
        typedef std::conditional_t<(1 < dim), slice<dim - 1>, std::ptrdiff_t> backing_t;
    public:
        template<std::ptrdiff_t offset>
        static [[nodiscard]] auto make_at(slice<dim> const &cur)
        {
            return slice_iter(cur, std::integral_constant<std::ptrdiff_t, offset>{});
        }
    private:
        backing_t backing;
        slice_axis axis;
        friend struct containers::wrap_iter<slice_iter>;
        [[nodiscard]] auto &bstart(void) noexcept
        {
            if constexpr (1 < dim) return backing.start; else return backing;
        }
        [[nodiscard]] auto bstart(void) const noexcept
        {
            return const_cast<slice_iter *>(this)->bstart();
        }
        [[nodiscard]] auto brest(void) noexcept
        {
            if constexpr (1 < dim) return backing.indices; else return nullptr;
        }
        [[nodiscard]] auto brest(void) const noexcept
        {
            return const_cast<slice_iter *>(this)->brest();
        }
        constexpr auto &op_shift(std::ptrdiff_t shift) & noexcept
        {
            bstart() += shift * axis.stride;
            axis.len -= shift;
            assert(0 <= axis.len);
            return *this;
        }
        template<std::ptrdiff_t offset>
        //Can't take cur by const ref, b/c need to tweak to prevent coerce assertion from firing
        constexpr slice_iter(slice<dim> cur, std::integral_constant<std::ptrdiff_t, offset>)
            noexcept : axis{ cur.indices[offset] }
        {
            using namespace _internal;
            bstart() = cur.coerce_to<check::no, offset>(brest());
        }
    public:
        friend [[nodiscard]] std::conditional_t<(1 < dim), std::partial_ordering, std::strong_ordering>
            operator<=>(slice_iter const &lhs, slice_iter const &rhs)
        {
            if constexpr (1 < dim)
                if (!memcmp(lhs.brest(), rhs.brest(), sizeof(lhs.backing.indices)))
                    return std::partial_ordering::unordered;
            return lhs.bstart() <=> rhs.bstart();
        }
        friend [[nodiscard]] auto operator==(slice_iter const &lhs, slice_iter const &rhs)
        {
            return (lhs <=> rhs) == 0;
        }
        [[nodiscard]] auto operator[](auto &data) const { return data[backing]; }
        [[nodiscard]] auto const &operator *(void) const { return backing; }
    };
}