#pragma once
#include <cstdint>
#include <bitset>
#include <memory>
#include <valarray>
#include <vector>

#include "Tweaks.hpp"
#include "LinAlgCommon.hpp"

namespace lin_alg
{
    struct index
    {
        std::ptrdiff_t start{ 0 };
        std::size_t len{ 1 };
        std::ptrdiff_t stride{ 1 };
    private:
        constexpr auto &operator*=(std::size_t scalar) noexcept
        {
            start *= scalar;
            stride *= scalar;
            return *this;
        }
        constexpr auto &operator+=(std::size_t base) noexcept
        {
            start += base;
            return *this;
        }
    public:
        //Necessary for implicit conversion from std::size_t
        constexpr index(std::ptrdiff_t start = 0, std::size_t len = 1, std::ptrdiff_t stride = 1) noexcept
            : start{ start }, len{ len }, stride{ stride }
        {}
        constexpr [[nodiscard]] auto operator[](index other) const noexcept
        {
            assert(other.len * other.stride <= this->len);
            other *= stride;
            other += start;
            return std::move(other);
        }
        constexpr [[nodiscard]] auto operator[](all_t) const noexcept { return *this; }
    };
    [[nodiscard]] bool operator==(index const &lhs, index const &rhs)
    {
        return !std::memcmp(&lhs, &rhs, sizeof(lhs));
    }

    class shape_error : public std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    typedef bool data_t;

    namespace _internal
    {
        template<typename T>
        struct holder abstract final
        {
            template<std::size_t count, typename...repetitions> struct repeat;
            template<typename...repetitions>
            struct repeat<0, repetitions...> abstract final
            {
                typedef std::tuple<repetitions...> type;
            };
            template<std::size_t count, typename... repetitions>
            struct repeat<count, repetitions...> abstract final
            {
                typedef repeat<count - 1, T, repetitions...>::type type;
            };
        };

        template<typename T, std::size_t count>
        using repeat_t = holder<T>::template repeat<count>::type;

        template<std::size_t count>
        struct nested_iter abstract final
        {
            typedef std::initializer_list<typename nested_iter<count - 1>::type> type;
        };
        template<> struct nested_iter<0> abstract final { typedef data_t type; };
    }

    template<std::size_t count> using nested_iter_t = _internal::nested_iter<count>::type;

    template<std::size_t const dim> class tensor;

    template<std::size_t dim, bool pts_to_const>
    struct tensor_iter : containers::wrap_iter<tensor_iter<dim, pts_to_const>>
    {
    public:
        typedef tensor<dim - 1> backing_t;
        typedef std::conditional_t<pts_to_const, backing_t const &, backing_t> ptd_to_t;
        backing_t slice;
        std::ptrdiff_t stride;
        constexpr auto &op_shift(std::ptrdiff_t shift) &
        {
            slice.indices[0].start += stride * shift;
            return *this;
        }
        constexpr [[nodiscard]] ptd_to_t operator*(void) const { return slice; }
        constexpr [[nodiscard]] auto operator<=>(tensor_iter const &other) const
        {
            assert(slice.backing == other.slice.backing);
            assert(stride == other.stride);
            return slice.indices[0] != other.slice.indices[0];
        }
    };

    class vector;

    template<std::size_t const dim>
    class tensor :
        public _internal::tensor_base<tensor<dim>, dim>,
        public/*friend declaration troubles*/ _internal::_insert_overloads<tensor<dim>, index, dim>
    {
    public:
        typedef _internal::repeat_t<data_t, dim> index_bundle;
        struct iteration_t
        {
            tensor<dim - 1> start, stop;
            std::ptrdiff_t stride;
            [[nodiscard]] auto cbegin(void) const { return tensor_iter<dim, true>{start, stride}; }
            [[nodiscard]] auto cend(void) const { return tensor_iter<dim, true>{stop, stride}; }
            [[nodiscard]] auto begin(void) const { return cbegin(); }
            [[nodiscard]] auto end(void) const { return cend(); }
            [[nodiscard]] auto begin(void) { return tensor_iter<dim, false>{start, stride}; }
            [[nodiscard]] auto end(void) { return tensor_iter<dim, true>{stop, stride}; }
        };
        index indices[dim]{ 0 };
        std::shared_ptr<data_t[]> backing;
    /*private:
        template<>
        template<std::uintmax_t code>
        friend struct loc::_overloads;*/
        template<std::size_t... unroll>
        [[nodiscard]] auto op_bracket(auto &&masks, std::index_sequence<unroll...>)
        {
            return tensor(backing, indices[unroll][get<unroll>(masks)]...);
        }
    private:
        template<typename il>
        index *init_sizes(std::initializer_list<il> const &data, index *idx)
        {
            using std::cend;
            assert(idx < cend(indices));
            using std::size;
            idx->len = std::max(size(data), idx->len);
            if constexpr (!std::is_same_v<il, data_t>)
            {
                index *retval{ nullptr };
                assert(size(data));
                for (auto const &list : data)
                    retval = init_sizes(list, idx - 1);
                return retval;
            }
            return idx;
        }
        template<typename il>
        constexpr auto copy_data(std::initializer_list<il> const &data, index *idx, data_t *loc)
        {
            using std::begin, std::end;
            if constexpr (std::is_same_v<il, data_t>)
                std::move(begin(data), end(data), loc);
            else
            {
                auto loc_cpy{ loc };
                for (auto const &list : data)
                {
                    copy_data(list, idx - 1, loc_cpy);
                    loc_cpy += idx->stride;
                }
            }
        }
    public:
        using tensor::recurse::operator[];
        constexpr tensor(void) = default;
        constexpr tensor(std::shared_ptr<data_t[]> backing, index(&&indices)[dim]) noexcept
            : backing{ std::move(backing) }
        {
            using std::begin, std::end;
            std::move(begin(indices), end(indices), begin(this->indices));
        }
        //Replacing with auto causes a compiler error
        template<typename... Args>
        constexpr tensor(std::shared_ptr<data_t[]> backing, Args&&... indices) noexcept
            : tensor(std::move(backing), { index{indices}... })
        {
            static_assert(dim == sizeof...(Args));
        }
        constexpr tensor(nested_iter_t<dim> data)
        {
            static_assert(dim > 0);
            using std::begin, std::cbegin, std::end, std::cend;
            auto loc{ init_sizes(data, end(this->indices) - 1) };
            assert(cbegin(indices) == loc);
            loc->stride = 1;
            std::size_t size(loc->stride);
            {
                auto const stop{ cend(indices) };
                decltype(loc) prev;
                while ((prev = loc++) != stop)
                    size *= (loc->stride = prev->len);
            }
            backing = std::make_shared<data_t[]>(size);
            copy_data(data, this->indices + dim - 1, this->backing.get());
        }
        constexpr [[nodiscard]] auto &to_scalar(void) noexcept
        {
            auto loc{ backing.get() };
            for (auto const &idx : indices)
            {
                assert(1 == idx.len);
                loc += idx.start;
            }
            return *loc;
        }
        constexpr explicit [[nodiscard]] operator data_t &(void) noexcept { return to_scalar(); }
        constexpr [[nodiscard]] auto const &to_scalar(void) const noexcept
        {
            return const_cast<tensor *>(this)->to_scalar();
        }
        constexpr explicit [[nodiscard]] operator data_t const &(void) const noexcept
        {
            return to_scalar();
        }
        template<std::ptrdiff_t a, std::ptrdiff_t b>
        constexpr friend [[nodiscard]] auto transpose(tensor obj) noexcept
        {
            std::swap(obj.indices[a], obj.indices[b]);
            return std::move(obj);
        }
        //TODO
        template<std::size_t dim> constexpr [[nodiscard]] auto cbegin(void) const;
        template<std::size_t dim> constexpr [[nodiscard]] auto begin(void) const { return cbegin<dim>(); };
        template<std::size_t dim> constexpr [[nodiscard]] auto begin(void);
        template<std::size_t dim> constexpr [[nodiscard]] auto cend(void) const;
        template<std::size_t dim> constexpr [[nodiscard]] auto end(void) const { return cend<dim>(); };
        template<std::size_t dim> constexpr [[nodiscard]] auto end(void);
    };

    typedef tensor<2> matrix_view;

    [[nodiscard]] auto transpose(matrix_view matrix) noexcept
    {
        return transpose<0, 1>(std::move(matrix));
    }

    class vector
    {
    public:
        std::shared_ptr<data_t[]> backing;
        index coords{ 0,0 };
        std::ptrdiff_t slice;
        vector() = default;
        explicit vector(matrix_view const &me) : backing{ me.backing }, slice{ 0 }
        {
            index const *loc{ nullptr };
            for (auto const &index : me.indices)
            {
                if (1 != index.len)
                {
                    if (!loc)
                        loc = &index;
#ifdef _DEBUG
                    else throw shape_error{ "Tried to coerce 2D matrix to vector" };
#endif
                }
                else slice += index.start;
            }
            /*if (loc)
                coords = *loc;
            else
                coords = { 0,1 };*/
            coords = loc ? *loc : decltype(*loc){0, 1};
        }
        [[nodiscard]] auto as_horiz(std::ptrdiff_t stride = 1) const
        {
            return matrix_view{ backing, coords, index{slice, 1, stride} };
        }
        [[nodiscard]] auto as_vert(std::ptrdiff_t stride = 1) const
        {
            return matrix_view{ backing, index{slice, 1, stride}, coords };
        }
        [[nodiscard]] data_t operator[](std::size_t index) const
        {
            return (backing)[slice + coords[index].start];
        }
        [[nodiscard]] auto size(void) const { return coords.len; };
        template<typename this_t>
        struct common_iter : public containers::wrap_iter<common_iter<this_t>>
        {
            this_t &backing;
            std::ptrdiff_t _iter;
            constexpr auto &op_shift(std::ptrdiff_t shift) &
            {
                _iter += shift;
                return *this;
            }
            constexpr common_iter(this_t &backing, std::ptrdiff_t offset)
                : backing{ backing }, _iter{ offset }
            {}
            [[nodiscard]] constexpr auto operator*(void) const { return backing[_iter]; }
            [[nodiscard]] constexpr auto operator!=(common_iter const &other) const
            {
                assert(backing.backing == other.backing.backing);
                return _iter != other._iter;
            }
        };
        constexpr [[nodiscard]] auto cbegin(void) const { return common_iter{*this, 0}; }
        constexpr [[nodiscard]] auto cend(void) const { return cbegin() + size(); }
        constexpr [[nodiscard]] auto begin(void) const { return cbegin(); }
        constexpr [[nodiscard]] auto end(void) const { return cend(); }
        constexpr [[nodiscard]] auto begin(void) { return common_iter{*this, 0}; }
        constexpr [[nodiscard]] auto end(void) { return begin() + size(); }
    };
    using std::size;
    using std::cbegin;
    using std::cend;
    using std::begin;
    using std::end;
}